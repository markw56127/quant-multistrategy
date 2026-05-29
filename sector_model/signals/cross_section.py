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
    # ── Fundamentals ─────────────────────────────────────────────────────
    "eps_surprise",          # beat/miss vs naive trailing avg (SEC EDGAR)
    "eps_ttm",               # trailing-12-month diluted EPS
    "eps_growth_yoy",        # YoY TTM EPS growth
    "eps_acceleration",      # change in YoY growth vs 1yr ago
    "eps_revision",          # analyst estimate revision (yfinance, sparse)
    "eps_revision_trend",    # rolling sign of revisions (yfinance, sparse)
    "revenue_growth_yoy",    # YoY quarterly revenue growth (SEC EDGAR)
    "log_market_cap",        # log(price × shares_outstanding) — SEC EDGAR, point-in-time
    "ipo_age_days",          # calendar days since first trade — captures post-IPO momentum
    "high_52w_ratio",        # price / 52-week high — breakout signal (George & Hwang 2004)
                             # stocks near 52w high break through analyst/institution anchoring
    "revenue_acceleration",  # QoQ change in YoY revenue growth — catches inflection points
                             # NVDA: +2%→+101% YoY in one quarter; MU cycle trough flip
    "gross_margin",          # trailing gross margin level
    "gross_margin_expansion",# YoY change in gross margin — pricing power signal
                             # NVDA 53%→78% GM showed up before price went parabolic
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

    # First-trade dates: first non-NaN date per stock — used for ipo_age feature
    first_trade = stock_returns.apply(lambda s: s.first_valid_index())

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

            # 52-week high proximity: price / max(price, 252d).
            # Near 1.0 = stock near annual high → breakout candidate.
            # George & Hwang (2004): strongest predictor among momentum variants.
            cum_ret    = ret.cumsum().apply(np.exp)   # price index (ratio scale)
            rolling_hi = cum_ret.rolling(252, min_periods=63).max()
            df["high_52w_ratio"] = (cum_ret / rolling_hi).shift(1).clip(0, 1)
        else:
            df[["total_mom_1m", "total_mom_3m", "reversal_1w",
                "vol_ratio", "mom_12_1", "high_52w_ratio"]] = np.nan

        df["beta"] = b.shift(1)
        df = df.join(regime_proba.shift(1))

        # IPO age: log(trading days since first listing).
        # Captures two documented effects:
        #   1. Post-IPO momentum: newly listed stocks outperform for 12-18 months
        #      (Ritter & Welch 2002) — high ipo_age_score early on
        #   2. Avoids treating pre-IPO NaN as a signal — the feature is NaN
        #      before the stock exists, which LightGBM handles as missing
        # For stocks with full history, ipo_age is large and effectively neutral
        # in cross-sectional z-scoring.
        t0 = first_trade.get(ticker)
        if t0 is not None and pd.notna(t0):
            days_listed = pd.Series(
                [(d - t0).days for d in r.index], index=r.index, dtype=float
            )
            df["ipo_age_days"] = days_listed.where(days_listed > 0)
        else:
            df["ipo_age_days"] = np.nan

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


class CompositeFactorModel:
    """
    Simple weighted composite of cross-sectional z-scored factors.

    Why this outperforms LightGBM on small universes:
      LightGBM with 200 trees and 15 leaves has thousands of degrees of
      freedom. On a 25-70 stock cross-section it memorises noise that doesn't
      generalise. This model has exactly k parameters (the factor weights) and
      can't overfit because there's nothing to fit — weights are fixed from the
      academic literature, not estimated from the data.

      The tradeoff: it can't learn non-linear interactions or adapt to regime
      changes. But for small-N cross-sections the bias-variance tradeoff strongly
      favours the simpler model.

    Default factor weights (configurable via composite_weights in config):
      mom_12_1          0.35  — Jegadeesh & Titman (1993) momentum
      eps_revision_trend 0.35  — persistent analyst upgrades (Stickel 1991)
      pe_rank_cs_neg    0.30  — relative value (cheap vs peers; Fama-French 1992)

    Factors with NaN values (e.g. fundamentals before 2020) are skipped for that
    date and the remaining weights are renormalised automatically.
    """

    _DEFAULT_WEIGHTS = {
        "mom_12_1":           0.35,
        "eps_revision_trend": 0.35,
        "pe_rank_cs_neg":     0.30,  # special key: uses -pe_rank_cs
    }

    def __init__(self, cfg: dict):
        self.cfg = cfg
        w = cfg.get("composite_weights", {})
        self.weights: Dict[str, float] = w if w else dict(self._DEFAULT_WEIGHTS)
        self.model = True   # sentinel: always "fitted", enables feature_importance logging

    def fit(self, *args, **kwargs) -> None:
        """No-op: weights are fixed, not estimated."""
        pass

    def predict(self, features: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        """Cross-sectional composite score for each stock. Higher = stronger long."""
        try:
            day = features.xs(date, level="date")
        except KeyError:
            return pd.Series(dtype=float)

        composite   = pd.Series(0.0, index=day.index)
        total_w     = 0.0

        for factor_name, weight in self.weights.items():
            col  = "pe_rank_cs" if factor_name == "pe_rank_cs_neg" else factor_name
            sign = -1.0         if factor_name == "pe_rank_cs_neg" else 1.0

            if col not in day.columns:
                continue
            vals = day[col].dropna()
            if len(vals) < 3 or vals.std() < 1e-8:
                continue

            z = sign * (vals - vals.mean()) / vals.std()
            composite = composite.add(weight * z, fill_value=0.0)
            total_w  += weight

        if total_w > 0:
            composite /= total_w   # renormalise when some factors are NaN

        return composite.rename("alpha_score")

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.weights).sort_values(ascending=False)


class MomentumRankModel:
    """
    Within-sector stock ranking by 12-1 month price momentum.

    Why use this instead of LightGBM for within-sector selection:
      - Momentum (Jegadeesh & Titman 1993) is the most replicated factor in
        finance. Within a sector, the top-momentum names outperform over
        1-12 month horizons with t-stats > 3 in most studies.
      - Very low turnover: momentum is persistent, so the same stocks stay
        near the top for multiple rebalance periods.
      - No training data required — pure signal, zero overfitting risk.
      - Combined with sector rotation (Layer 1), this targets the best stocks
        in the best sectors: the intersection of sector momentum and
        within-sector stock momentum.

    The model also blends in EPS growth and revenue growth (when available
    from SEC EDGAR) to distinguish sustained fundamental momentum from
    pure price momentum. Price-only momentum can reverse on earnings misses;
    fundamental confirmation reduces this whipsaw.
    """

    _WEIGHTS = {
        "mom_12_1":            0.35,  # 12-1 month price momentum
        "high_52w_ratio":      0.25,  # proximity to 52-week high — breakout signal.
                                      # NVDA was at 60% of 52w high in Jan 2023,
                                      # then broke through after Feb earnings and ran.
                                      # George & Hwang (2004): strongest momentum variant.
        "revenue_acceleration":0.20,  # second derivative of revenue growth.
                                      # Fires at the inflection (NVDA +2%→+101% YoY).
                                      # The signal that was actually available early.
        "eps_growth_yoy":      0.15,  # fundamental confirmation (slower, lagging)
        "vol_ratio":          -0.20,  # quality filter: penalise 2× sector vol names
    }

    def __init__(self, cfg: dict = None):
        self.model = True  # sentinel: always "fitted"

    def fit(self, *args, **kwargs) -> None:
        pass

    def predict(self, features: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        try:
            day = features.xs(date, level="date")
        except KeyError:
            return pd.Series(dtype=float)

        composite = pd.Series(0.0, index=day.index)
        total_w   = 0.0

        for col, w in self._WEIGHTS.items():
            if col not in day.columns:
                continue
            vals = day[col].dropna()
            if len(vals) < 3 or vals.std() < 1e-8:
                continue
            z = (vals - vals.mean()) / vals.std()
            composite = composite.add(w * z, fill_value=0.0)
            total_w += w

        if total_w > 0:
            composite /= total_w

        return composite.rename("alpha_score")

    def feature_importance(self) -> pd.Series:
        return pd.Series(self._WEIGHTS).sort_values(ascending=False)
