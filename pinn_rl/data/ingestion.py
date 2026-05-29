import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


class DataIngestion:
    """
    Fetches and caches OHLCV data for a universe of tickers.
    Supports hierarchical sector grouping and earnings calendar fetching.
    """

    def __init__(self, cache_dir: str = "cache/data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_universe(
        self,
        tickers_by_sector: Dict[str, List[str]],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Returns {ticker: ohlcv_df} for every ticker in the universe.
        Results are disk-cached keyed by (tickers, start, end, interval).
        """
        all_tickers = [t for tickers in tickers_by_sector.values() for t in tickers]
        cache_key = self._cache_key(all_tickers, start_date, end_date, interval)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading OHLCV data from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        logger.info(f"Fetching OHLCV for {len(all_tickers)} tickers from {start_date} to {end_date}")
        data = self._batch_download(all_tickers, start_date, end_date, interval)

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

        return data

    def fetch_sector_etfs(
        self,
        etfs: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        return self._batch_download(etfs, start_date, end_date, interval)

    def fetch_benchmark(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        data = self._batch_download([ticker], start_date, end_date, interval)
        return data.get(ticker, pd.DataFrame())

    def build_return_matrix(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        price_col: str = "Adj Close",
        log_returns: bool = True,
    ) -> pd.DataFrame:
        """Returns a DataFrame of (log-)returns: rows=dates, cols=tickers."""
        prices = {}
        for ticker, df in ohlcv_data.items():
            col = price_col if price_col in df.columns else "Close"
            prices[ticker] = df[col]

        price_df = pd.DataFrame(prices).dropna(how="all")
        if log_returns:
            return np.log(price_df / price_df.shift(1)).dropna()
        return price_df.pct_change().dropna()

    def build_sector_return_matrix(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        tickers_by_sector: Dict[str, List[str]],
        log_returns: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Returns {sector: return_df} for each sector group."""
        all_returns = self.build_return_matrix(ohlcv_data, log_returns=log_returns)
        sector_returns = {}
        for sector, tickers in tickers_by_sector.items():
            valid = [t for t in tickers if t in all_returns.columns]
            if valid:
                sector_returns[sector] = all_returns[valid]
        return sector_returns

    def fetch_earnings_calendar(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with earnings surprise data where available.
        Columns: ticker, date, eps_actual, eps_estimate, surprise_pct
        """
        rows = []
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        for ticker in tickers:
            try:
                info = yf.Ticker(ticker)
                earnings = info.earnings_history
                if earnings is None or earnings.empty:
                    continue
                earnings = earnings.reset_index()
                # Normalize date column
                date_col = next((c for c in earnings.columns if "date" in c.lower()), None)
                if date_col is None:
                    continue
                earnings[date_col] = pd.to_datetime(earnings[date_col])
                mask = (earnings[date_col] >= start) & (earnings[date_col] <= end)
                for _, row in earnings[mask].iterrows():
                    rows.append({
                        "ticker": ticker,
                        "date": row[date_col],
                        "eps_actual": row.get("epsActual", np.nan),
                        "eps_estimate": row.get("epsEstimate", np.nan),
                        "surprise_pct": row.get("surprisePercent", np.nan),
                    })
            except Exception as e:
                logger.warning(f"Earnings fetch failed for {ticker}: {e}")

        if not rows:
            return pd.DataFrame(columns=["ticker", "date", "eps_actual", "eps_estimate", "surprise_pct"])
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _batch_download(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str,
        batch_size: int = 50,
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            try:
                raw = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    for ticker in batch:
                        try:
                            # yfinance 1.x uses (Ticker, Price) MultiIndex; older used (Price, Ticker)
                            if ticker in raw.columns.get_level_values(0):
                                df = raw[ticker].dropna(how="all")
                            else:
                                df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                            if not df.empty:
                                result[ticker] = df
                        except KeyError:
                            logger.warning(f"No data returned for {ticker}")
                else:
                    # Single ticker
                    if not raw.empty:
                        result[batch[0]] = raw.dropna(how="all")
            except Exception as e:
                logger.error(f"Batch download failed for {batch}: {e}")

        logger.info(f"Successfully fetched data for {len(result)}/{len(tickers)} tickers")
        return result

    @staticmethod
    def _cache_key(tickers: List[str], start: str, end: str, interval: str) -> str:
        raw = f"{'_'.join(sorted(tickers))}_{start}_{end}_{interval}"
        return hashlib.md5(raw.encode()).hexdigest()
