"""
Sector universe data loader.

The universe and sector ETF are read from config, making this pipeline
sector-agnostic. See config/sectors/ for pre-built sector configurations.

Any tickers with >20% missing data (late IPOs, delistings) are dropped
automatically before modeling.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


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

    # Two-tier filter:
    # >80% missing → not enough history to score reliably even with IPO handling
    # 20-80% missing → keep but flag as partial-history (IPO or late addition)
    # <20% missing → full history, standard treatment
    hard_drop = missing_frac[missing_frac > 0.80].index.tolist()
    if hard_drop:
        logger.warning(f"Dropping {hard_drop} — >80% missing (too short to score)")
        prices = prices.drop(columns=hard_drop)

    partial = missing_frac[(missing_frac > 0.20) & (missing_frac <= 0.80)].index.tolist()
    if partial:
        logger.info(f"Partial-history tickers (IPO/late addition): {partial}")

    # Keep rows where ≥50% of full-history stocks have data — preserves the full
    # date range even when partial-history tickers are present, avoiding the
    # prior bug where one late IPO cut everyone's history to its listing date.
    full_hist = missing_frac[missing_frac <= 0.20].index
    if len(full_hist) > 0:
        min_cols = max(1, int(0.5 * len(full_hist)))
        prices = prices.dropna(subset=full_hist, thresh=min_cols)
    else:
        prices = prices.dropna(how="all")

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
        logger.info(f"Cached to {cache_path}")

    return prices


def fetch_volumes(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch daily trading volumes for stock tickers. Cached separately from prices.
    Columns aligned to the same tickers/dates as fetch_prices.
    """
    if cache_dir:
        cache_path = Path(cache_dir) / f"volumes_{start}_{end}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached volumes from {cache_path}")
            return pd.read_parquet(cache_path)

    logger.info(f"Fetching volumes for {len(tickers)} tickers [{start} → {end}]")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        volumes = raw["Volume"]
    else:
        volumes = raw[["Volume"]].rename(columns={"Volume": tickers[0]}) \
                  if "Volume" in raw.columns else pd.DataFrame(index=raw.index)

    volumes = volumes.ffill(limit=5)
    missing_frac = volumes.isna().mean()
    drop = missing_frac[missing_frac > 0.20].index.tolist()
    if drop:
        volumes = volumes.drop(columns=drop)
    volumes = volumes.fillna(0)

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        volumes.to_parquet(cache_path)
        logger.info(f"Cached volumes to {cache_path}")

    return volumes


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

    sector_col = cfg["sector_etf"]

    # If the sector ETF was dropped by the missing-data filter (e.g., XLC
    # launched Dec 2018 so it has >20% missing on a 2015 start), fetch it
    # separately over its actual available range and join it in.
    if sector_col not in prices.columns:
        logger.info(f"{sector_col} missing from prices — fetching separately")
        import yfinance as yf
        etf_raw = yf.download(sector_col, start=cfg["start_date"],
                              end=cfg["end_date"], auto_adjust=True, progress=False)
        close = etf_raw["Close"]
        # yfinance returns DataFrame with MultiIndex for single ticker in newer versions
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        etf_close = close.rename(sector_col)
        prices = prices.join(etf_close, how="left")

    log_ret = np.log(prices / prices.shift(1)).dropna(subset=[sector_col])
    log_ret = log_ret.dropna(how="all")

    sector_ret = log_ret[sector_col]
    stock_ret  = log_ret[[t for t in cfg["universe"] if t in log_ret.columns]].dropna(how="all")

    logger.info(
        f"Loaded {stock_ret.shape[1]} stocks × {len(stock_ret)} days "
        f"({stock_ret.index[0].date()} → {stock_ret.index[-1].date()})"
    )
    return stock_ret, sector_ret, prices
