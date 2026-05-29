"""
S&P 500 universe — constituent list with GICS sector labels and ETF benchmarks.

Universe is fetched from Wikipedia on first call and cached as CSV.  This
introduces mild survivorship bias (only current constituents are included) but
is acceptable for a first-pass backtest; any stock that joined after 2015 will
be dropped by the >20% missing-data filter in fetch_prices automatically.

GICS sector → sector ETF mapping used as the beta benchmark for OLS
decomposition and performance attribution.
"""

from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

# GICS sector name → SPDR Select Sector ETF
SECTOR_ETFS: Dict[str, str] = {
    "Information Technology":  "XLK",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Industrials":             "XLI",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Energy":                  "XLE",
    "Materials":               "XLB",
    "Real Estate":             "XLRE",
    "Utilities":               "XLU",
    "Communication Services":  "XLC",
}

# Sectors where the fundamental/momentum model is most likely to add alpha.
# Energy is excluded: returns are dominated by oil price, not fundamentals.
# Utilities and Real Estate are rate-sensitive bond-proxies with less
# cross-sectional dispersion from earnings signals.
ACTIVE_SECTORS: List[str] = [
    "Information Technology",
    "Financials",
    "Health Care",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Materials",
    "Communication Services",
]

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_universe(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Return a DataFrame with columns: Symbol, Security, GICS Sector, GICS Sub-Industry.
    Fetches from Wikipedia; caches to sp500_universe.csv.
    """
    if cache_dir:
        cache_path = Path(cache_dir) / "sp500_universe.csv"
        if cache_path.exists():
            logger.info(f"Loading cached S&P 500 universe from {cache_path}")
            return pd.read_csv(cache_path)

    logger.info("Fetching S&P 500 constituent list from Wikipedia...")
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(_WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_html(StringIO(r.text))[0]
    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    # Normalise ticker format (BRK.B → BRK-B for yfinance)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    # Drop Wikipedia parsing artifacts — single-character or non-alpha tickers
    # are never real S&P 500 symbols; they come from footnote markers in the table
    df = df[df["Symbol"].str.match(r"^[A-Z]{2,5}(-[A-Z])?$", na=False)]

    logger.info(
        f"S&P 500: {len(df)} stocks across {df['GICS Sector'].nunique()} sectors"
    )

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached universe to {cache_path}")

    return df


def get_sector_tickers(universe: pd.DataFrame, sector: str) -> List[str]:
    """Return tickers for a given GICS sector name."""
    return universe[universe["GICS Sector"] == sector]["Symbol"].tolist()


def build_sector_cfg(
    tickers: List[str],
    sector_etf: str,
    base_cfg: dict,
    cache_subdir: str,
) -> dict:
    """
    Build a sector-specific config dict from a base config.
    Only data.universe, data.sector_etf, and data.cache_dir are overridden;
    all modelling parameters (LightGBM, portfolio, backtest) are inherited.
    """
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["universe"]   = tickers
    cfg["data"]["sector_etf"] = sector_etf
    cfg["data"]["cache_dir"]  = cache_subdir
    # 3 high-conviction picks per sector — concentration is the point
    cfg["portfolio"]["n_long"] = 3
    return cfg
