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
    # ── Technical / price-based ──────────────────────────────────────────
    "idio_mom_1m",    # 21-day lagged idiosyncratic return
    "idio_mom_3m",    # 63-day
    "idio_mom_6m",    # 126-day
    "total_mom_1m",   # 21-day stock return minus sector return
    "total_mom_3m",
    "reversal_1w",    # 5-day return reversal (mean-reversion signal)
    "vol_ratio",      # stock realized vol / sector realized vol
    "beta",           # current rolling OLS beta
    "regime_0",       # HMM bear probability
    "regime_1",       # HMM chop probability
    "regime_2",       # HMM bull probability
    # ── Fundamental (available ~2020+; NaN for earlier dates) ────────────
    "eps_surprise",       # % EPS surprise vs analyst estimate at most recent earnings
    "eps_ttm",            # trailing 12-month EPS
    "eps_growth_yoy",     # TTM EPS growth vs same TTM one year prior
    "eps_acceleration",   # change in YoY growth rate (is growth speeding up?)
    "eps_revision",       # analyst EPS estimate change vs prior quarter's estimate
    "trailing_pe",        # price / TTM EPS
    "peg_ratio",          # trailing P/E / (EPS growth × 100)
]


def build_features(
    stock_returns:   pd.DataFrame,
    residuals:       pd.DataFrame,
    betas:           pd.DataFrame,
    regime_proba:    pd.DataFrame,
    sector_returns:  pd.Series,
    fundamentals:    Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a long-format panel with (date, ticker) MultiIndex.

    All features are lagged by at least 1 day so there is no forward-looking
    bias when the DataFrame is sliced at any rebalance date.

    Args:
        fundamentals : optional (date, ticker) MultiIndex DataFrame from
                       data.fundamentals.fetch_fundamental_features.
                       Joined on matching (date, ticker) pairs; NaN where
                       earnings history is unavailable.
    """
    sector_vol = sector_returns.rolling(21).std()
    stock_vol  = stock_returns.rolling(21).std()

    records: List[pd.DataFrame] = []
    for ticker in residuals.columns:
        r   = residuals[ticker]
        ret = stock_returns[ticker] if ticker in stock_returns.columns else None
        b   = betas[ticker]

        df = pd.DataFrame(index=r.index)
        df["idio_mom_1m"]  = r.rolling(21).sum().shift(1)
        df["idio_mom_3m"]  = r.rolling(63).sum().shift(1)
        df["idio_mom_6m"]  = r.rolling(126).sum().shift(1)

        if ret is not None:
            df["total_mom_1m"] = (ret - sector_returns).rolling(21).sum().shift(1)
            df["total_mom_3m"] = (ret - sector_returns).rolling(63).sum().shift(1)
            df["reversal_1w"]  = ret.rolling(5).sum().shift(1)
            df["vol_ratio"]    = (stock_vol[ticker] / (sector_vol + 1e-8)).shift(1)
        else:
            df[["total_mom_1m", "total_mom_3m", "reversal_1w", "vol_ratio"]] = np.nan

        df["beta"] = b.shift(1)
        df = df.join(regime_proba.shift(1))
        df["ticker"] = ticker
        records.append(df)

    panel = pd.concat(records).sort_index()
    panel.index = pd.MultiIndex.from_arrays(
        [panel.index, panel["ticker"]], names=["date", "ticker"]
    )
    panel = panel.drop(columns=["ticker"])

    if fundamentals is not None:
        fund_cols = [c for c in fundamentals.columns if c in FEATURE_COLS]
        panel = panel.join(fundamentals[fund_cols], how="left")

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
