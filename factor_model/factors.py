"""
Factor computation with sector-neutralisation.

Five factors, each grounded in the academic literature:

  VALUE      cheap stocks outperform expensive ones
             composite of Book/Price, Earnings/Price, Sales/Price
             Fama & French (1992)
  MOMENTUM   recent winners keep winning (12-1 month)
             Jegadeesh & Titman (1993)
  QUALITY    profitable, efficient firms outperform
             composite of gross profitability and ROE
             Novy-Marx (2013)
  LOW_VOL    low-volatility stocks have higher risk-adjusted returns
             Ang, Hodrick, Xing & Zhang (2006)
  SIZE       small-caps carry a premium (used with caution)
             Banz (1981)

Each raw factor is winsorised, cross-sectionally z-scored, and
SECTOR-NEUTRALISED (demeaned within GICS sector) so the composite is a pure
stock-selection bet rather than an implicit sector tilt. Without this, the
value factor alone would load heavily into financials and energy.

The composite is an EQUAL-WEIGHT average of the available factor z-scores.
Equal weighting is deliberately not optimised — the "1/N" result (DeMiguel,
Garlappi & Uppal 2009) shows naive diversification is hard to beat
out-of-sample and avoids fitting factor weights to the backtest.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


# Factors and their orientation (+1: higher raw value = more attractive)
FACTORS = ["value", "momentum", "quality", "low_vol", "size"]


def _winsorize(s: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    """Clip to [2%, 98%] quantiles to limit the influence of outliers."""
    if s.notna().sum() < 5:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std < 1e-12 or s.notna().sum() < 5:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _sector_neutralize(values: pd.Series, sectors: pd.Series) -> pd.Series:
    """Demean within sector, then z-score the residual cross-sectionally."""
    df = pd.DataFrame({"v": values, "sector": sectors.reindex(values.index)})
    # Subtract the sector mean from each stock's value
    df["resid"] = df["v"] - df.groupby("sector")["v"].transform("mean")
    return _zscore(df["resid"])


def compute_factor_scores(
    date: pd.Timestamp,
    prices: pd.DataFrame,         # (T, N) adjusted close
    fundamentals: pd.DataFrame,   # (date, ticker) MultiIndex panel
    sectors: pd.Series,           # ticker -> GICS sector
    members: Optional[set] = None,         # point-in-time index members at `date`
    composite_factors: Optional[List[str]] = None,  # which factors enter composite
) -> pd.DataFrame:
    """
    Compute sector-neutral factor z-scores for every stock as of `date`.

    Returns a DataFrame indexed by ticker with columns:
      value, momentum, quality, low_vol, size, composite
    Stocks with insufficient data are dropped.

    `members`: if given, restrict the universe to point-in-time index members
    (survivorship-bias correction). `composite_factors`: subset of FACTORS that
    enter the composite (default all five). All factors are still computed and
    returned for attribution.
    """
    composite_factors = composite_factors or FACTORS

    # Universe at this date: stocks with a valid price
    px_hist = prices.loc[:date]
    if len(px_hist) < 252:
        return pd.DataFrame()
    last_px = px_hist.iloc[-1]
    universe = last_px.dropna().index
    if members is not None:
        universe = universe.intersection(pd.Index(list(members)))
    if len(universe) < 20:
        return pd.DataFrame()

    # ── Fundamentals snapshot as of `date` ────────────────────────────────
    try:
        fund = fundamentals.xs(date, level="date")
    except KeyError:
        # Use the most recent available fundamentals on/before date
        avail_dates = fundamentals.index.get_level_values("date")
        prior = avail_dates[avail_dates <= date]
        if len(prior) == 0:
            return pd.DataFrame()
        fund = fundamentals.xs(prior.max(), level="date")

    fund = fund.reindex(universe)
    market_cap = last_px.reindex(universe) * fund["shares_outstanding"]

    raw = pd.DataFrame(index=universe)

    # ── VALUE: book/price, earnings/price, sales/price ────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        bp = fund["book_equity"]     / market_cap
        ep = fund["net_income_ttm"]  / market_cap
        sp = fund["revenue_ttm"]     / market_cap
    value_parts = [_zscore(_winsorize(x)) for x in (bp, ep, sp)]
    raw["value"] = pd.concat(value_parts, axis=1).mean(axis=1)

    # ── MOMENTUM: 12-1 month return ───────────────────────────────────────
    if len(px_hist) >= 252:
        p_now = px_hist.iloc[-21]          # skip the most recent month
        p_12m = px_hist.iloc[-252]
        with np.errstate(divide="ignore", invalid="ignore"):
            mom = (p_now / p_12m - 1).reindex(universe)
        raw["momentum"] = _zscore(_winsorize(mom))
    else:
        raw["momentum"] = 0.0

    # ── QUALITY: gross profitability + ROE ────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        gp_assets = fund["gross_profit_ttm"] / fund["total_assets"]
        roe       = fund["net_income_ttm"]   / fund["book_equity"]
    quality_parts = [_zscore(_winsorize(x)) for x in (gp_assets, roe)]
    raw["quality"] = pd.concat(quality_parts, axis=1).mean(axis=1)

    # ── LOW_VOL: negative trailing 1y volatility ──────────────────────────
    rets = np.log(px_hist / px_hist.shift(1)).iloc[-252:]
    vol  = rets.std().reindex(universe) * np.sqrt(252)
    raw["low_vol"] = _zscore(_winsorize(-vol))   # negate: low vol = high score

    # ── SIZE: negative log market cap (small-cap premium) ─────────────────
    raw["size"] = _zscore(_winsorize(-np.log(market_cap.clip(lower=1e6))))

    # ── Sector-neutralise each factor ─────────────────────────────────────
    out = pd.DataFrame(index=universe)
    for f in FACTORS:
        out[f] = _sector_neutralize(raw[f], sectors)

    # ── Equal-weight composite over the SELECTED factors only ─────────────
    out["composite"] = out[composite_factors].mean(axis=1)
    out = out.dropna(subset=["composite"])
    return out
