"""
Walk-forward backtest for the sector model.

Structure:
  - Warm-up period = train_window days (no trading)
  - Every rebalance_freq days: predict → construct → hold → record
  - Every retrain_every_n rebalances: re-fit LightGBM on trailing data

Exposure management:
  Gross exposure is determined by volatility targeting (Moreira & Muir 2017):
      exposure = clip( target_vol / sector_vol_ann, min_exposure, max_exposure )
  The HMM regime model is retained as a feature source for LightGBM but no
  longer drives binary exposure decisions.
"""

from typing import Dict, List, Optional, Tuple

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
        second_model:   Optional[CrossSectionalModel] = None,
        second_model_name: str = "composite",
    ):
        self.stock_returns  = stock_returns
        self.residuals      = residuals
        self.features       = features
        self.targets        = targets
        self.sector_returns = sector_returns
        self.betas          = betas
        self.regime_model   = regime_model
        self.alpha_model    = alpha_model
        self.second_model   = second_model
        self.second_model_name = second_model_name
        self.cfg            = cfg

    def run(self) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run backtest with primary model. If second_model is set, returns tuple of
        (primary_results, secondary_results).
        """
        rebal_freq      = self.cfg.get("rebalance_freq",   20)
        train_window    = self.cfg.get("train_window",    504)
        retrain_every   = self.cfg.get("retrain_every_n",   3)
        trans_cost      = self.cfg.get("transaction_cost", 0.001)
        initial_capital = self.cfg.get("initial_capital", 1_000_000)

        port_cfg     = self.cfg.get("portfolio", {})
        n_long       = port_cfg.get("n_long",            8)
        max_pos      = port_cfg.get("max_position_size", 0.20)
        target_vol   = port_cfg.get("target_vol",        0.15)
        min_exposure = port_cfg.get("min_exposure",      0.50)
        max_exposure = port_cfg.get("max_exposure",      1.00)

        dates   = self.stock_returns.index
        vol_21d = self.residuals.rolling(21).std()

        # Downside semideviation (annualised): clip positive returns to zero before
        # computing rolling std.  Only days where the sector falls contribute, so
        # exposure scales back up faster during V-shaped recoveries than with total vol.
        # The √2 factor normalises so that for a symmetric distribution this equals
        # total vol, keeping the target_vol=0.15 calibration consistent.
        downside_ret     = self.sector_returns.clip(upper=0)
        sector_vol_daily = downside_ret.rolling(21).std() * np.sqrt(252) * np.sqrt(2)

        # HMM state series for diagnostics (precomputed, no look-ahead concern
        # since it's only used as a feature in LightGBM, not for trading decisions)
        regime_proba  = self.regime_model.predict_proba(self.sector_returns)
        bull_prob_all = regime_proba["regime_2"]

        start_idx   = train_window
        rebal_dates = dates[start_idx::rebal_freq]

        capital         = float(initial_capital)
        second_capital  = float(initial_capital)
        current_weights = pd.Series(0.0, index=self.stock_returns.columns)
        records: List[Dict] = []
        second_records: List[Dict] = [] if self.second_model else None
        last_fit_rebal  = -retrain_every

        for rebal_num, rebal_date in enumerate(rebal_dates):
            date_idx = dates.get_loc(rebal_date)

            # Re-fit LightGBM every `retrain_every` rebalances
            if rebal_num - last_fit_rebal >= retrain_every:
                train_start = dates[max(0, date_idx - train_window)]
                train_end   = dates[date_idx - 1]
                self.alpha_model.fit(
                    self.features, self.targets, train_start, train_end
                )
                if self.second_model:
                    self.second_model.fit(
                        self.features, self.targets, train_start, train_end
                    )
                last_fit_rebal = rebal_num

            # Current sector vol for exposure scaling
            sector_vol_ann = float(sector_vol_daily.loc[rebal_date]) \
                             if rebal_date in sector_vol_daily.index else target_vol

            # Predict alpha scores from primary model
            try:
                scores = self.alpha_model.predict(self.features, rebal_date)
            except Exception as e:
                logger.warning(f"{rebal_date}: prediction failed ({e}), holding flat")
                scores = pd.Series(0.0, index=self.stock_returns.columns)

            stock_vol_now = vol_21d.loc[rebal_date].reindex(scores.index).fillna(0.02)
            scores        = scores.reindex(self.stock_returns.columns).fillna(0.0)

            new_weights = construct_weights(
                scores, stock_vol_now, sector_vol_ann,
                n_long, max_pos, target_vol, min_exposure, max_exposure,
            ).reindex(self.stock_returns.columns).fillna(0.0)

            # If second model is available, also run it
            if self.second_model:
                try:
                    second_scores = self.second_model.predict(self.features, rebal_date)
                except Exception as e:
                    logger.warning(f"{rebal_date}: second model prediction failed ({e})")
                    second_scores = pd.Series(0.0, index=self.stock_returns.columns)
                
                second_scores = second_scores.reindex(self.stock_returns.columns).fillna(0.0)
                second_weights = construct_weights(
                    second_scores, stock_vol_now, sector_vol_ann,
                    n_long, max_pos, target_vol, min_exposure, max_exposure,
                ).reindex(self.stock_returns.columns).fillna(0.0)
            else:
                second_weights = None

            # Transaction cost on turnover (use primary model weights)
            turnover = (new_weights - current_weights).abs().sum()
            capital *= (1.0 - turnover * trans_cost)

            # If second model exists, also deduct transaction costs
            if self.second_model:
                second_capital *= (1.0 - turnover * trans_cost)

            # Hold for rebal_freq days
            end_idx    = min(date_idx + rebal_freq, len(dates) - 1)
            hold_slice = self.stock_returns.iloc[date_idx:end_idx]
            period_ret = float((hold_slice * new_weights).sum(axis=1).sum())
            capital   *= (1.0 + period_ret)

            # If second model exists, also compute its returns
            if self.second_model and second_weights is not None:
                second_period_ret = float((hold_slice * second_weights).sum(axis=1).sum())
                second_capital *= (1.0 + second_period_ret)
            else:
                second_period_ret = None

            # Benchmark: sector ETF return over same hold period
            bench_ret = float(self.sector_returns.iloc[date_idx:end_idx].sum())

            # IC: rank correlation of scores vs realised period returns
            realized = hold_slice.sum()
            ic = float(scores.reindex(realized.index).corr(realized, method="spearman")) \
                 if len(realized) else float("nan")

            if self.second_model:
                second_ic = float(second_scores.reindex(realized.index).corr(realized, method="spearman")) \
                     if len(realized) else float("nan")
            else:
                second_ic = None

            long_mask = new_weights > 0
            long_ret  = float((hold_slice * new_weights.where(long_mask, 0)).sum(axis=1).sum())

            if self.second_model and second_weights is not None:
                second_long_mask = second_weights > 0
                second_long_ret = float((hold_slice * second_weights.where(second_long_mask, 0)).sum(axis=1).sum())
            else:
                second_long_ret = None

            bull_prob = float(bull_prob_all.loc[rebal_date]) \
                        if rebal_date in bull_prob_all.index else float("nan")

            records.append({
                "date":              rebal_date,
                "capital":           capital,
                "period_return":     period_ret,
                "benchmark_return":  bench_ret,
                "long_return":       long_ret,
                "short_return":      0.0,
                "ic":                ic,
                "sector_vol_ann":    sector_vol_ann,
                "bull_prob":         bull_prob,
                "turnover":          float(turnover),
                "cost":              float(turnover * trans_cost),
                "gross_exposure":    float(new_weights.abs().sum()),
                "n_long":            int((new_weights > 0).sum()),
            })

            if self.second_model and second_records is not None:
                second_records.append({
                    "date":              rebal_date,
                    "capital":           second_capital,
                    "period_return":     second_period_ret,
                    "benchmark_return":  bench_ret,
                    "long_return":       second_long_ret,
                    "short_return":      0.0,
                    "ic":                second_ic,
                    "sector_vol_ann":    sector_vol_ann,
                    "bull_prob":         bull_prob,
                    "turnover":          float(turnover),  # Use same turnover estimate as primary
                    "cost":              float(turnover * trans_cost),
                    "gross_exposure":    float(second_weights.abs().sum()),
                    "n_long":            int((second_weights > 0).sum()),
                })

            current_weights = new_weights

        results = pd.DataFrame(records).set_index("date")
        total_ret = capital / initial_capital - 1
        logger.info(
            f"Primary model backtest complete | ${capital:,.0f} | total return: {total_ret*100:.1f}%"
        )

        if self.second_model and second_records:
            second_results = pd.DataFrame(second_records).set_index("date")
            second_capital = second_results["capital"].iloc[-1]
            second_total_ret = second_capital / initial_capital - 1
            logger.info(
                f"Secondary model ({self.second_model_name}) backtest complete | ${second_capital:,.0f} | total return: {second_total_ret*100:.1f}%"
            )
            return results, second_results
        
        return results

    @staticmethod
    def performance_stats(results: pd.DataFrame, periods_per_year: float = 12.0) -> Dict:
        """Annualized stats. periods_per_year = 252/rebalance_freq."""
        r = results["period_return"]
        if len(r) < 4 or r.std() < 1e-10:
            return {}

        sharpe     = (r.mean() / r.std()) * np.sqrt(periods_per_year)
        vals       = results["capital"]
        max_dd     = float((vals / vals.cummax() - 1).min())
        total      = float(vals.iloc[-1] / vals.iloc[0] - 1)
        ann_ret    = float((1 + total) ** (periods_per_year / len(r)) - 1)
        ic         = results["ic"].dropna()
        total_cost = float(results["cost"].sum())

        bench_total = float((1 + results["benchmark_return"]).prod() - 1)
        active      = results["period_return"] - results["benchmark_return"]
        info_ratio  = float((active.mean() / active.std()) * np.sqrt(periods_per_year)) \
                      if active.std() > 1e-10 else 0.0

        return {
            "total_return":           total,
            "benchmark_total_return": bench_total,
            "excess_return":          total - bench_total,
            "annualized_return":      ann_ret,
            "annualized_sharpe":      float(sharpe),
            "information_ratio":      info_ratio,
            "max_drawdown":           max_dd,
            "mean_ic":                float(ic.mean()),
            "ic_hit_rate":            float((ic > 0).mean()),
            "long_return_total":      float(results["long_return"].sum()),
            "total_cost":             total_cost,
            "n_rebalances":           len(r),
            "avg_turnover":           float(results["turnover"].mean()),
            "avg_gross_exposure":     float(results["gross_exposure"].mean()),
            "avg_sector_vol_ann":     float(results["sector_vol_ann"].mean()),
            "avg_bull_prob":          float(results["bull_prob"].mean()),
        }
