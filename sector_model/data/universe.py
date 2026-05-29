"""
Semiconductor universe data loader.

SMH (VanEck Semiconductor ETF) top constituents define our stock universe.
The list is roughly stable over the backtest period; any delistings or late
IPOs surface as NaNs and are dropped column-wise before modeling.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

SEMI_UNIVERSE: List[str] = [
    "NVDA", "AVGO", "TSM",  "ASML", "AMD",
    "QCOM", "AMAT", "LRCX", "MU",   "KLAC",
    "MRVL", "INTC", "TXN",  "ADI",  "NXPI",
    "MCHP", "ON",   "MPWR", "SWKS", "TER",
    "ENTG", "ACLS", "ONTO", "FORM", "COHU",
]
SECTOR_ETF = "SMH"


def fetch_prices(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices. Caches to parquet so re-runs are instant.
    Forward-fills up to 5 days to handle market holidays; drops columns with
    more than 20% missing (late IPOs, extended halts).
    """
    if cache_dir:
        cache_path = Path(cache_dir) / f"prices_{start}_{end}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached prices from {cache_path}")
            return pd.read_parquet(cache_path)

    logger.info(f"Fetching {len(tickers)} tickers [{start} → {end}]")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    prices = prices.ffill(limit=5)
    missing_frac = prices.isna().mean()
    drop = missing_frac[missing_frac > 0.20].index.tolist()
    if drop:
        logger.warning(f"Dropping {drop} — >20% missing")
        prices = prices.drop(columns=drop)
    prices = prices.dropna()

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
        logger.info(f"Cached to {cache_path}")

    return prices


def load_sector_data(
    cfg: dict,
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Returns:
        stock_returns  : (T, N) daily log-returns for universe stocks
        sector_returns : (T,)  daily log-returns for sector ETF
        prices         : (T, N+1) raw adjusted close prices
    """
    all_tickers = cfg["universe"] + [cfg["sector_etf"]]
    prices = fetch_prices(all_tickers, cfg["start_date"], cfg["end_date"], cache_dir)

    log_ret = np.log(prices / prices.shift(1)).dropna()

    sector_col = cfg["sector_etf"]
    sector_ret = log_ret[sector_col]
    stock_ret  = log_ret[[t for t in cfg["universe"] if t in log_ret.columns]]

    logger.info(
        f"Loaded {stock_ret.shape[1]} stocks × {len(stock_ret)} days "
        f"({stock_ret.index[0].date()} → {stock_ret.index[-1].date()})"
    )
    return stock_ret, sector_ret, prices
