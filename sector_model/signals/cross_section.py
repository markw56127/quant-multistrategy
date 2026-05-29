"""
Cross-sectional alpha model: predicts forward idiosyncratic return per stock.

Why LightGBM instead of a neural net or the existing SAC/PINN stack:
  - At 25 stocks × ~2,500 training dates ≈ 62,500 rows per training window,
    LightGBM consistently outperforms deep learning on tabular finance data
    (Gu, Kelly & Xiu 2020 JFE; documented in multiple Numerai/Kaggle benchmarks).
  - Gradient boosting handles missing features (gaps from rolling windows)
    and feature interactions without manual engineering.
  - Training is fast enough to re-fit quarterly in a walk-forward loop.
  - Feature importance via `.feature_importance()` makes the model debuggable.

Walk-forward discipline:
  Train on [t - train_window, t - 5], predict at t.
  The 5-day gap is an execution buffer — we use only features that could
  realistically be acted on by the time we're actually trading.
"""

from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

FEATURE_COLS = [
    # ── Per-stock technical ───────────────────────────────────────────────
    "idio_mom_1m",        # 21-day idiosyncratic return
    "idio_mom_3m",        # 63-day
    "idio_mom_6m",        # 126-day
    "total_mom_1m",       # 21-day stock return minus sector return
    "total_mom_3m",
    "reversal_1w",        # 5-day return reversal
    "vol_ratio",          # stock realized vol / sector realized vol
    "beta",               # rolling OLS beta to sector
    "mom_12_1",           # 12-month return minus last month (Jegadeesh & Titman 1993)
    # ── HMM regime probabilities ─────────────────────────────────────────
    "regime_0",           # bear probability
    "regime_1",           # chop probability
    "regime_2",           # bull probability
    # ── Cross-sectional market context (same value for all stocks each day) ─
    "cs_dispersion",      # rolling 21d avg of cross-sectional return std.
                          # High = stocks diverging = more alpha opportunity.
    "breadth_200d",       # fraction of universe above 200d cumulative return.
                          # Low breadth + high index = narrow/fragile rally.
    "sector_vol_pctile",  # current 21d sector vol as percentile of trailing 252d.
                          # Lets model distinguish calm bull (low pctile) from
                          # crisis vol spike (high pctile) — Item 6 proxy without
                          # splitting into separate models.
    # ── Volume signals (per-stock) ────────────────────────────────────────
    "rel_volume",         # 5d avg volume / 63d avg volume. Rising volume = attention.
    "price_vol_corr",     # 21d correlation between daily return and log-vol change.
                          # Positive = price moves confirmed by volume (accumulation).
                          # Negative = price moves on declining volume (weak signal).
    # ── Fundamentals (available ~2020+) ──────────────────────────────────
    "eps_surprise",
    "eps_ttm",
    "eps_growth_yoy",
    "eps_acceleration",
    "eps_revision",
    "eps_revision_trend",
    "trailing_pe",
    "peg_ratio",
    "pe_rank_cs",
]


def build_features(
    stock_returns:   pd.DataFrame,
    residuals:       pd.DataFrame,
    betas:           pd.DataFrame,
    regime_proba:    pd.DataFrame,
    sector_returns:  pd.Series,
    fundamentals:    Optional[pd.DataFrame] = None,
    volumes:         Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a long-format panel with (date, ticker) MultiIndex.
    All features lagged ≥1 day — no forward-looking bias.
    """
    sector_vol  = sector_returns.rolling(21).std()
    stock_vol   = stock_returns.rolling(21).std()

    # ── Cross-sectional market context (broadcast to all tickers) ────────
    # cs_dispersion: when stocks are diverging, alpha is more available
    cs_disp = stock_returns.std(axis=1).rolling(21).mean().shift(1)
    # breadth_200d: fraction of universe above 200d cumulative return level
    breadth = (stock_returns.rolling(200).sum() > 0).mean(axis=1).shift(1)
    # sector_vol_pctile: is current vol historically high or low?
    # Proxies the vol-regime split (Item 6) without needing two separate models
    s_vol_pctile = sector_vol.rolling(252, min_periods=63).rank(pct=True).shift(1)

    records: List[pd.DataFrame] = []
    for ticker in residuals.columns:
        r   = residuals[ticker]
        ret = stock_returns[ticker] if ticker in stock_returns.columns else None
        b   = betas[ticker]

        df = pd.DataFrame(index=r.index)

        # Per-stock technical
        df["idio_mom_1m"] = r.rolling(21).sum().shift(1)
        df["idio_mom_3m"] = r.rolling(63).sum().shift(1)
        df["idio_mom_6m"] = r.rolling(126).sum().shift(1)

        if ret is not None:
            df["total_mom_1m"] = (ret - sector_returns).rolling(21).sum().shift(1)
            df["total_mom_3m"] = (ret - sector_returns).rolling(63).sum().shift(1)
            df["reversal_1w"]  = ret.rolling(5).sum().shift(1)
            df["vol_ratio"]    = (stock_vol[ticker] / (sector_vol + 1e-8)).shift(1)
            df["mom_12_1"]     = (ret.rolling(252).sum() - ret.rolling(21).sum()).shift(1)
        else:
            df[["total_mom_1m", "total_mom_3m", "reversal_1w",
                "vol_ratio", "mom_12_1"]] = np.nan

        df["beta"] = b.shift(1)
        df = df.join(regime_proba.shift(1))

        # Cross-sectional market context (same for all tickers each day)
        df["cs_dispersion"]    = cs_disp
        df["breadth_200d"]     = breadth
        df["sector_vol_pctile"] = s_vol_pctile

        # Volume signals
        if volumes is not None and ticker in volumes.columns:
            vol_s = volumes[ticker].reindex(r.index).ffill()
            vol_s = vol_s.clip(lower=1)
            df["rel_volume"] = (
                vol_s.rolling(5).mean() / (vol_s.rolling(63).mean() + 1e-6)
            ).shift(1)
            log_dvol = np.log(vol_s).diff()
            df["price_vol_corr"] = ret.rolling(21).corr(log_dvol).shift(1) \
                                   if ret is not None else np.nan
        else:
            df[["rel_volume", "price_vol_corr"]] = np.nan

        df["ticker"] = ticker
        records.append(df)

    panel = pd.concat(records).sort_index()
    panel.index = pd.MultiIndex.from_arrays(
        [panel.index, panel["ticker"]], names=["date", "ticker"]
    )
    panel = panel.drop(columns=["ticker"])

    # Fundamental features
    if fundamentals is not None:
        fund_cols = [c for c in fundamentals.columns if c in FEATURE_COLS]
        panel = panel.join(fundamentals[fund_cols], how="left")

    # Cross-sectional P/E rank (computed across all tickers at each date)
    if "trailing_pe" in panel.columns:
        panel["pe_rank_cs"] = (
            panel["trailing_pe"]
            .groupby(level="date")
            .rank(pct=True, na_option="keep")
        )

    return panel


class CrossSectionalModel:
    """Walk-forward LightGBM predictor trained on stacked (date, ticker) panels."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model: Optional[lgb.Booster] = None

    def _extract_Xy(
        self,
        features: pd.DataFrame,
        targets:  pd.DataFrame,
        start: pd.Timestamp,
        end:   pd.Timestamp,
    ):
        """
        Merge features and targets for dates in [start, end].
        targets is a (T, N) DataFrame; we stack it to match the feature MultiIndex.
        """
        feat_window = features.loc[start:end]

        tgt_stacked = (
            targets.loc[start:end]
            .stack()
            .rename("target")
        )
        tgt_stacked.index.names = ["date", "ticker"]

        merged = feat_window.join(tgt_stacked, how="inner").dropna(subset=["target"])
        X = merged[FEATURE_COLS].values.astype(np.float32)
        y = merged["target"].values.astype(np.float32)
        return X, y

    def fit(
        self,
        features:    pd.DataFrame,
        targets:     pd.DataFrame,
        train_start: pd.Timestamp,
        train_end:   pd.Timestamp,
    ) -> None:
        X, y = self._extract_Xy(features, targets, train_start, train_end)
        if len(X) < 100:
            logger.warning(f"Only {len(X)} training samples — skipping fit")
            return

        params: Dict = {
            "objective":            "regression",
            "metric":               "rmse",
            "num_leaves":           self.cfg.get("num_leaves", 31),
            "learning_rate":        self.cfg.get("learning_rate", 0.05),
            "feature_fraction":     self.cfg.get("feature_fraction", 0.8),
            "bagging_fraction":     self.cfg.get("bagging_fraction", 0.8),
            "bagging_freq":         self.cfg.get("bagging_freq", 5),
            "min_child_samples":    self.cfg.get("min_child_samples", 20),
            "verbosity":            -1,
        }
        n_rounds = self.cfg.get("n_estimators", 300)
        early    = self.cfg.get("early_stopping_rounds", 50)

        # 10% validation split for early stopping; no shuffling (time-series)
        n_val = max(1, int(len(X) * 0.10))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(early, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        logger.info(
            f"LightGBM trained on {len(X_tr)} rows | "
            f"best iteration: {self.model.best_iteration} | "
            f"val RMSE: {self.model.best_score['valid_0']['rmse']:.5f}"
        )

    def predict(self, features: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """Alpha score for each stock at `date`. Higher = more attractive long."""
        if self.model is None:
            raise RuntimeError("Model not fitted — call .fit() first")
        try:
            day = features.xs(date, level="date")
        except KeyError:
            return pd.Series(dtype=float)

        X = day[FEATURE_COLS].values.astype(np.float32)
        scores = self.model.predict(X)
        return pd.Series(scores, index=day.index, name="alpha_score")

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        imp = self.model.feature_importance(importance_type="gain")
        return pd.Series(imp, index=FEATURE_COLS).sort_values(ascending=False)
