"""
Point-in-time S&P 500 membership — survivorship-bias correction.

The sector_model and factor_model v0 used *current* S&P 500 constituents,
which silently selects survivors: stocks that are in today's index are the
ones that didn't go bankrupt or get removed. This made the size factor look
like a free +187% (positive every single year — impossible for a real factor).

This module reconstructs which stocks were in the index ON EACH DATE using the
fja05680/sp500 historical-components dataset (derived from S&P's own change
announcements). The universe at each rebalance is then the actual membership at
that time, not today's.

Irreducible limitation with free data:
  ~55% of dropped names are fully delisted (acquisitions, bankruptcies) and
  have no yfinance price history. We recover the ~45% that still trade. This
  reduces but does not eliminate survivorship bias — bankruptcies (the worst
  outcomes) are disproportionately the unrecoverable ones. Absolute returns
  remain mildly optimistic; relative factor rankings are trustworthy.
"""

from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
import yfinance as yf
from loguru import logger

_COMPONENTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv"
)


def load_membership(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Return DataFrame with columns [date, tickers] where tickers is a
    comma-separated string of members on that date. Cached locally.
    """
    if cache_dir:
        p = Path(cache_dir) / "sp500_historical_components.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["date"] = pd.to_datetime(df["date"])
            return df

    logger.info("Fetching historical S&P 500 components (fja05680)...")
    r = requests.get(_COMPONENTS_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["date"] = pd.to_datetime(df["date"])
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(cache_dir) / "sp500_historical_components.csv", index=False)
    return df


def _clean(ticker: str) -> str:
    """Strip date suffixes like 'ADT-201604' → 'ADT' and normalise for yfinance."""
    return ticker.split("-")[0].replace(".", "-").strip()


def historical_universe(
    start: str, end: str, cache_dir: Optional[str] = None
) -> List[str]:
    """All distinct tickers that were S&P 500 members at any point in [start, end]."""
    df = load_membership(cache_dir)
    mask = (df["date"] >= start) & (df["date"] <= pd.Timestamp(end) + pd.Timedelta(days=400))
    sub = df[mask] if mask.any() else df
    universe: Set[str] = set()
    for row in sub["tickers"]:
        for tk in str(row).split(","):
            c = _clean(tk)
            if c:
                universe.add(c)
    return sorted(universe)


def membership_matrix(
    trading_dates: pd.DatetimeIndex, cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Boolean DataFrame [trading_dates × tickers]: True where the ticker was an
    index member on that date. The components file is sparse (rows only on
    change dates), so we forward-fill the most recent membership snapshot.
    """
    df = load_membership(cache_dir).sort_values("date")

    # Build membership sets per change date
    snapshots = {row["date"]: {_clean(t) for t in str(row["tickers"]).split(",")}
                 for _, row in df.iterrows()}
    snap_dates = sorted(snapshots)

    all_tickers = sorted(set().union(*snapshots.values()))
    mat = pd.DataFrame(False, index=trading_dates, columns=all_tickers)

    for d in trading_dates:
        prior = [sd for sd in snap_dates if sd <= d]
        if not prior:
            continue
        members = snapshots[prior[-1]]
        mat.loc[d, list(members & set(all_tickers))] = True
    return mat


# yfinance .info sector names → GICS sector names (for consistency with the
# Wikipedia GICS labels used for current members)
_YF_TO_GICS = {
    "Technology":             "Information Technology",
    "Financial Services":     "Financials",
    "Healthcare":             "Health Care",
    "Consumer Cyclical":      "Consumer Discretionary",
    "Consumer Defensive":     "Consumer Staples",
    "Communication Services": "Communication Services",
    "Industrials":            "Industrials",
    "Energy":                 "Energy",
    "Basic Materials":        "Materials",
    "Real Estate":            "Real Estate",
    "Utilities":              "Utilities",
}


def fetch_prices_survivorship_free(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch adjusted-close prices KEEPING partial-history names. Unlike
    sector_model's fetch_prices (which drops >20% missing), this retains
    delisted stocks — they have valid prices until delisting, then NaN. The
    factor computation handles NaN per-date, so a stock simply leaves the
    investable universe once it stops trading. This is essential for
    survivorship-free backtesting.

    Keeps any column with ≥60 valid observations (enough to be scored at all).
    """
    if cache_dir:
        p = Path(cache_dir) / f"prices_sf_{start}_{end}.parquet"
        if p.exists():
            logger.info(f"Loading cached survivorship-free prices from {p}")
            return pd.read_parquet(p)

    logger.info(f"Fetching {len(tickers)} tickers (survivorship-free) [{start} → {end}]...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.ffill(limit=5)
    keep = prices.columns[prices.notna().sum() >= 60]
    prices = prices[keep]
    logger.info(f"Kept {prices.shape[1]} of {len(tickers)} tickers (≥60 obs)")

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        prices.to_parquet(Path(cache_dir) / f"prices_sf_{start}_{end}.parquet")
    return prices


def fetch_sectors(
    tickers: List[str],
    seed: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
) -> pd.Series:
    """
    GICS sector per ticker. `seed` provides already-known GICS labels (current
    members from Wikipedia). Only tickers missing from both seed and cache are
    looked up via yfinance .info (delisted names), mapped to GICS. Unresolved →
    'Unknown' (their own neutralisation group, harmless).
    """
    cache_path = Path(cache_dir) / "historical_sectors.csv" if cache_dir else None
    known: Dict[str, str] = dict(seed) if seed else {}
    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0)["sector"].to_dict()
        known = {**cached, **known}   # seed (Wikipedia GICS) takes precedence

    missing = [t for t in tickers if t not in known]
    if missing:
        logger.info(f"Looking up {len(missing)} delisted-name sectors via yfinance.info...")
        for i, t in enumerate(missing):
            try:
                raw = yf.Ticker(t).info.get("sector")
                known[t] = _YF_TO_GICS.get(raw, "Unknown")
            except Exception:
                known[t] = "Unknown"
            if (i + 1) % 50 == 0:
                logger.info(f"  sectors: {i+1}/{len(missing)}")
                if cache_path:
                    pd.Series(known, name="sector").to_frame().to_csv(cache_path)
        if cache_path:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            pd.Series(known, name="sector").to_frame().to_csv(cache_path)

    return pd.Series({t: known.get(t, "Unknown") for t in tickers}, name="sector")
