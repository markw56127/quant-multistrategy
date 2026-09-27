"""
Point-in-time market capitalisation, free of split/dividend lookahead.

Why this exists (SPLIT_FINDING.md): the obvious construction
`adjusted_price x EDGAR shares_outstanding` is wrong in a way that leaks the future.
yfinance adjusted prices divide every past price by all LATER splits and dividends,
while EDGAR's share count is as-reported. A stock that will split k:1 shows a past
market cap k times too small. It looks cheap on every price multiple, and since
stocks split after rising, a value factor built this way sorts partly on future
returns. In this repo that artifact WAS the development-period value premium.

Construction:
    raw_close(t)  = split-adjusted Close(t) x prod(splits after t)   # Close: not dividend-adjusted
    shares_now(t) = shares_reported(filing f) x prod(splits in (f, t])
    market_cap(t) = raw_close(t) x shares_now(t)

Filing dates f are the change points of the forward-filled EDGAR share series.
Names with no raw price data on Yahoo (fully delisted) fall back to the old
construction and are counted in the log, because leaving them out would change
the universe.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


def fetch_raw_close_splits(tickers, start: str, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Split-adjusted (not dividend-adjusted) Close and split events, downloaded through
    today so every split in the Close adjustment basis is also in the event list."""
    p = Path(cache_dir) / "raw_close_splits.parquet" if cache_dir else None
    if p is not None and p.exists():
        cached = pd.read_parquet(p)
        if set(tickers) <= set(cached["close"].columns):
            return cached
    logger.info(f"Fetching raw Close + splits for {len(tickers)} tickers ...")
    raw = yf.download(list(tickers), start=start, auto_adjust=False, actions=True, progress=False)
    out = pd.concat({"close": raw["Close"], "splits": raw["Stock Splits"]}, axis=1)
    if p is not None:
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(p)
    return out


def _log_split_cum(splits: pd.Series):
    sp = splits.replace(0.0, np.nan).dropna()
    if sp.empty:
        return None
    return np.log(sp).cumsum()


def _cum_at(logC, x) -> np.ndarray:
    """log prod of splits on or before each date in x."""
    if logC is None:
        return np.zeros(len(x))
    i = np.searchsorted(logC.index.values, np.asarray(x, dtype="datetime64[ns]"), side="right")
    return np.where(i > 0, logC.values[np.maximum(i - 1, 0)], 0.0)


def market_cap_panel(shares: pd.DataFrame, adj_prices: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """shares: date x ticker as-reported EDGAR shares (forward-filled). Returns date x ticker market cap."""
    close, splits = raw["close"], raw["splits"]
    dates = shares.index
    out, fallback = {}, []
    for t in shares.columns:
        s = shares[t]
        if t not in close.columns or close[t].notna().sum() < 60:
            out[t] = adj_prices[t].reindex(dates) * s if t in adj_prices.columns else s * np.nan
            fallback.append(t)
            continue
        logC = _log_split_cum(splits[t]) if t in splits.columns else None
        total = logC.iloc[-1] if logC is not None else 0.0
        filed = pd.Series(np.where(s.ne(s.shift()) & s.notna(), s.index, pd.NaT), index=dates).ffill()
        raw_close = close[t].reindex(dates).ffill(limit=5) * np.exp(total - _cum_at(logC, dates))
        shares_now = s * np.exp(_cum_at(logC, dates) - _cum_at(logC, pd.DatetimeIndex(filed.values)))
        out[t] = raw_close * shares_now
    if fallback:
        logger.warning(f"market cap: {len(fallback)} names without raw Yahoo data use the "
                       f"adjusted-price construction: {fallback}")
    return pd.DataFrame(out)


def add_market_cap(fundamentals: pd.DataFrame, adj_prices: pd.DataFrame, start: str,
                   cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Add a `market_cap` column to a (date, ticker) fundamentals panel."""
    shares = fundamentals["shares_outstanding"].unstack("ticker")
    raw = fetch_raw_close_splits(list(shares.columns), start, cache_dir)
    mc = market_cap_panel(shares, adj_prices, raw)
    fundamentals = fundamentals.copy()
    fundamentals["market_cap"] = mc.stack(future_stack=True).reindex(fundamentals.index)
    return fundamentals
