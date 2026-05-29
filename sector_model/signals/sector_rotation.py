"""
Sector rotation model — top-down allocation across GICS sectors.

Two-factor composite (no ML — only 11 sectors, same small-N problem):

  1. Sector momentum (12-1 month):
     Moskowitz & Grinblatt (1999) showed sector momentum is one of the
     strongest and most persistent factors in equity markets.  Past 12-month
     return skipping last month predicts next-month relative sector performance.

  2. Macro tilt:
     Each sector has a known directional relationship with the yield curve
     and market volatility (VIX).  These relationships are grounded in
     economic theory, not fitted to data:
       - Financials benefit from a steep yield curve (net interest margin)
       - Utilities/Real Estate are hurt by rising rates (rate-sensitive)
       - Defensives (staples, healthcare) outperform when VIX spikes
       - Cyclicals (tech, discretionary) outperform in low-VIX risk-on regimes

Final sector weight = softmax( momentum_score + macro_score ), floored at
a minimum allocation so no sector is ever fully zeroed out.

Macro data fetched from yfinance:
  ^VIX  — CBOE Volatility Index
  ^TNX  — 10-year Treasury yield
  ^IRX  — 3-month Treasury yield
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


# Sector macro sensitivities: (yield_curve_beta, vix_beta)
# yield_curve_beta > 0 → benefits from steep (10y > 3m) yield curve
# vix_beta > 0 → benefits from high VIX (defensive), < 0 → risk-on
_MACRO_BETAS: Dict[str, tuple] = {
    "Information Technology":  ( 0.0, -1.0),
    "Financials":              (+1.5, -0.5),
    "Health Care":             ( 0.0, +0.8),
    "Industrials":             ( 0.0, -0.6),
    "Consumer Discretionary":  ( 0.0, -0.8),
    "Consumer Staples":        ( 0.0, +1.0),
    "Energy":                  ( 0.0, -0.4),
    "Materials":               ( 0.0, -0.6),
    "Real Estate":             (-1.5, +0.5),
    "Utilities":               (-1.0, +1.0),
    "Communication Services":  ( 0.0, -0.5),
}

_MACRO_TICKERS = ["^VIX", "^TNX", "^IRX"]


def fetch_macro_data(
    start: str,
    end: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download daily VIX, 10y yield, and 3m yield.  Returns a DataFrame with
    columns: vix, yield_10y, yield_3m, yield_curve (10y - 3m).
    """
    from pathlib import Path
    import pyarrow  # noqa: ensure parquet backend available

    if cache_dir:
        cache_path = Path(cache_dir) / f"macro_{start}_{end}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached macro data from {cache_path}")
            return pd.read_parquet(cache_path)

    logger.info("Fetching macro data (VIX, yields)...")
    raw = yf.download(_MACRO_TICKERS, start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    macro = pd.DataFrame(index=close.index)
    macro["vix"]         = close.get("^VIX")
    macro["yield_10y"]   = close.get("^TNX")
    macro["yield_3m"]    = close.get("^IRX")
    macro["yield_curve"] = macro["yield_10y"] - macro["yield_3m"]
    macro = macro.ffill(limit=5).dropna()

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        macro.to_parquet(cache_path)

    logger.info(f"Macro data: {len(macro)} days  "
                f"VIX mean={macro['vix'].mean():.1f}  "
                f"yield curve mean={macro['yield_curve'].mean():.2f}%")
    return macro


class SectorRotationModel:
    """
    Composite sector allocation model.  Scores each sector at every rebalance
    date and converts to portfolio weights via softmax.
    """

    def __init__(
        self,
        sector_etf_returns: pd.DataFrame,   # (T, n_sectors) daily log-returns
        macro: pd.DataFrame,                 # output of fetch_macro_data()
        sectors: List[str],                  # ordered list matching etf_returns cols
        momentum_weight: float = 0.60,
        macro_weight:    float = 0.40,
        min_sector_alloc: float = 0.05,      # floor: no sector goes to zero
    ):
        self.etf_ret    = sector_etf_returns
        self.macro      = macro
        self.sectors    = sectors
        self.mom_w      = momentum_weight
        self.macro_w    = macro_weight
        self.min_alloc  = min_sector_alloc

    def _sector_momentum(self, date: pd.Timestamp) -> pd.Series:
        """12-1 month sector ETF return (Jegadeesh-Titman applied to sectors)."""
        try:
            t = self.etf_ret.index.get_loc(date)
        except KeyError:
            return pd.Series(0.0, index=self.etf_ret.columns)
        lo_12 = max(0, t - 252)
        lo_1  = max(0, t - 21)
        mom_12 = self.etf_ret.iloc[lo_12:t].sum()
        mom_1  = self.etf_ret.iloc[lo_1:t].sum()
        return (mom_12 - mom_1).reindex(self.etf_ret.columns).fillna(0.0)

    def _macro_score(self, date: pd.Timestamp) -> pd.Series:
        """Yield-curve and VIX tilt per sector."""
        try:
            macro_row = self.macro.loc[:date].iloc[-1]
        except (KeyError, IndexError):
            return pd.Series(0.0, index=self.sectors)

        # z-score macro features relative to trailing 252-day distribution
        hist = self.macro.loc[:date].tail(252)
        yc_z   = (macro_row["yield_curve"] - hist["yield_curve"].mean()) \
                 / (hist["yield_curve"].std() + 1e-8)
        vix_z  = (macro_row["vix"] - hist["vix"].mean()) \
                 / (hist["vix"].std() + 1e-8)

        scores = {}
        for sector in self.sectors:
            yc_beta, vix_beta = _MACRO_BETAS.get(sector, (0.0, 0.0))
            scores[sector] = yc_beta * yc_z + vix_beta * vix_z
        return pd.Series(scores)

    def predict(self, date: pd.Timestamp) -> pd.Series:
        """
        Return sector allocation weights at `date`. Weights sum to 1.
        Indexed by sector name (must match keys in SECTOR_ETFS).
        """
        mom   = self._sector_momentum(date)
        macro = self._macro_score(date)

        # Align momentum to sector names via etf_returns columns
        # etf_returns columns are ETF tickers; map back to sector names
        etf_to_sector = {v: k for k, v in
                         {s: self.etf_ret.columns[i]
                          for i, s in enumerate(self.sectors)}.items()}
        mom_by_sector = pd.Series(
            {etf_to_sector.get(col, col): val for col, val in mom.items()}
        ).reindex(self.sectors).fillna(0.0)

        # Cross-sectional z-score both factors
        def _z(s: pd.Series) -> pd.Series:
            std = s.std()
            return (s - s.mean()) / std if std > 1e-8 else s * 0

        composite = self.mom_w * _z(mom_by_sector) + self.macro_w * _z(macro)

        # Softmax → weights
        exp_s   = np.exp(composite - composite.max())   # subtract max for stability
        weights = exp_s / exp_s.sum()

        # Apply minimum floor and renormalise
        weights = weights.clip(lower=self.min_alloc)
        weights = weights / weights.sum()

        return weights

    @staticmethod
    def log_weights(weights: pd.Series, date: pd.Timestamp) -> None:
        top = weights.sort_values(ascending=False)
        logger.debug(
            f"Sector weights @ {date.date()}: "
            + "  ".join(f"{s[:4]}={w:.2f}" for s, w in top.items())
        )
