"""
Walk-forward backtest for the sector model.

Structure:
  - Warm-up period = train_window days (no trading)
  - Every rebalance_freq days: predict → construct → hold → record
  - Every retrain_every_n rebalances: re-fit LightGBM on trailing data

This simulates realistic operation: the model is retrained periodically
(not continuously) and positions are held for the full rebalance window
with transaction costs paid on entry.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from signals.cross_section import CrossSectionalModel
from signals.sector import SectorRegimeModel
from portfolio.optimizer import construct_weights


class SectorBacktest:
    def __init__(
        self,
        stock_returns:  pd.DataFrame,
        residuals:      pd.DataFrame,
        features:       pd.DataFrame,
        targets:        pd.DataFrame,
        sector_returns: pd.Series,
        betas:          pd.DataFrame,
        regime_model:   SectorRegimeModel,
        alpha_model:    CrossSectionalModel,
        cfg:            dict,
    ):
        self.stock_returns  = stock_returns
        self.residuals      = residuals
        self.features       = features
        self.targets        = targets
        self.sector_returns = sector_returns
        self.betas          = betas
        self.regime_model   = regime_model
        self.alpha_model    = alpha_model
        self.cfg            = cfg

    def run(self) -> pd.DataFrame:
        rebal_freq      = self.cfg.get("rebalance_freq",   20)
        train_window    = self.cfg.get("train_window",    504)
        retrain_every   = self.cfg.get("retrain_every_n",   3)
        trans_cost      = self.cfg.get("transaction_cost", 0.001)
        initial_capital = self.cfg.get("initial_capital", 1_000_000)

        port_cfg = self.cfg.get("portfolio", {})
        n_long   = port_cfg.get("n_long",  8)
        n_short  = 0   # long-only: no short positions
        max_pos  = port_cfg.get("max_position_size", 0.20)
        bear_scale = 0.35   # handled inside construct_weights via REGIME_EXPOSURE

        dates    = self.stock_returns.index
        regimes  = self.regime_model.predict_state(self.sector_returns)
        vol_21d  = self.residuals.rolling(21).std()

        start_idx   = train_window
        rebal_dates = dates[start_idx::rebal_freq]

        capital          = float(initial_capital)
        current_weights  = pd.Series(0.0, index=self.stock_returns.columns)
        records: List[Dict] = []
        last_fit_rebal = -retrain_every  # trigger fit on first rebalance

        for rebal_num, rebal_date in enumerate(rebal_dates):
            date_idx = dates.get_loc(rebal_date)

            # Re-fit LightGBM every `retrain_every` rebalances
            if rebal_num - last_fit_rebal >= retrain_every:
                train_start = dates[max(0, date_idx - train_window)]
                train_end   = dates[date_idx - 1]
                self.alpha_model.fit(
                    self.features, self.targets, train_start, train_end
                )
                last_fit_rebal = rebal_num

            # Predict alpha scores
            try:
                scores = self.alpha_model.predict(self.features, rebal_date)
            except Exception as e:
                logger.warning(f"{rebal_date}: prediction failed ({e}), holding flat")
                scores = pd.Series(0.0, index=self.stock_returns.columns)

            regime = int(regimes.loc[rebal_date]) if rebal_date in regimes.index else 1
            vol_now = vol_21d.loc[rebal_date].reindex(scores.index).fillna(0.02)
            scores  = scores.reindex(self.stock_returns.columns).fillna(0.0)

            new_weights = construct_weights(
                scores, vol_now, regime, n_long, n_short, max_pos, bear_scale
            ).reindex(self.stock_returns.columns).fillna(0.0)

            # Transaction cost on turnover
            turnover = (new_weights - current_weights).abs().sum()
            capital *= (1.0 - turnover * trans_cost)

            # Hold for rebal_freq days
            end_idx    = min(date_idx + rebal_freq, len(dates) - 1)
            hold_slice = self.stock_returns.iloc[date_idx:end_idx]
            period_ret = float((hold_slice * new_weights).sum(axis=1).sum())
            capital   *= (1.0 + period_ret)

            # IC: rank correlation between alpha scores and realized period return
            realized = hold_slice.sum()
            ic = float(scores.reindex(realized.index).corr(realized, method="spearman")) if len(realized) else float("nan")

            # Long leg vs short leg P&L attribution
            long_mask  = new_weights > 0
            short_mask = new_weights < 0
            long_ret   = float((hold_slice * new_weights.where(long_mask,  0)).sum(axis=1).sum())
            short_ret  = float((hold_slice * new_weights.where(short_mask, 0)).sum(axis=1).sum())

            records.append({
                "date":            rebal_date,
                "capital":         capital,
                "period_return":   period_ret,
                "long_return":     long_ret,
                "short_return":    short_ret,
                "ic":              ic,
                "turnover":        float(turnover),
                "cost":            float(turnover * trans_cost),
                "regime":          regime,
                "gross_exposure":  float(new_weights.abs().sum()),
                "n_long":          int((new_weights > 0).sum()),
                "n_short":         int((new_weights < 0).sum()),
            })
            current_weights = new_weights

        results = pd.DataFrame(records).set_index("date")
        total_ret = capital / initial_capital - 1
        logger.info(
            f"Backtest complete | ${capital:,.0f} | total return: {total_ret*100:.1f}%"
        )
        return results

    @staticmethod
    def performance_stats(results: pd.DataFrame, periods_per_year: float = 12.0) -> Dict:
        """Annualized stats. periods_per_year = 252/rebalance_freq (e.g. 252/20=12.6)."""
        r = results["period_return"]
        if len(r) < 4 or r.std() < 1e-10:
            return {}
        sharpe   = (r.mean() / r.std()) * np.sqrt(periods_per_year)
        vals     = results["capital"]
        max_dd   = float((vals / vals.cummax() - 1).min())
        total    = float(vals.iloc[-1] / vals.iloc[0] - 1)
        ann_ret  = float((1 + total) ** (periods_per_year / len(r)) - 1)
        ic       = results["ic"].dropna()
        total_cost = float(results["cost"].sum())
        return {
            "total_return":       total,
            "annualized_return":  ann_ret,
            "annualized_sharpe":  float(sharpe),
            "max_drawdown":       max_dd,
            "mean_ic":            float(ic.mean()),
            "ic_hit_rate":        float((ic > 0).mean()),
            "long_return_total":  float(results["long_return"].sum()),
            "short_return_total": float(results["short_return"].sum()),
            "total_cost":         total_cost,
            "n_rebalances":       len(r),
            "avg_turnover":       float(results["turnover"].mean()),
            "avg_gross_exposure": float(results["gross_exposure"].mean()),
            "pct_time_bear":      float((results["regime"] == 0).mean()),
        }
