"""
Earnings-surprise events from SEC EDGAR.

Builds, per ticker, a table of (announcement_date, SUE) where SUE is
Standardised Unexpected Earnings using the seasonal-random-walk model:

    expected_EPS(q)   = EPS(q-4)                    # same quarter, prior year
    unexpected_EPS(q) = EPS(q) - expected_EPS(q)
    SUE(q)            = unexpected_EPS(q) / std(unexpected_EPS, trailing 8q)

This is the Foster-Olsen-Shevlin (1984) / Bernard-Thomas (1989) formulation.
It predates analyst consensus data and is the standard PEAD signal when
I/B/E/S estimates are unavailable. High positive SUE → earnings beat the
naive seasonal expectation → stock tends to drift up for 20-60 days.

Announcement date = SEC filing date (exact, no look-ahead). We use the 10-Q/
10-K filing date as the event timestamp; the price reaction and drift are
measured from the following trading day.

Reuses sec_edgar primitives from sector_model.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sector_model"))
from data.sec_edgar import _fetch_facts, _quarterly_series, fetch_cik_map  # noqa: E402

_DELAY = 0.22
_EPS_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
_MIN_HISTORY = 6   # need ≥6 quarters before SUE is meaningful


def _ticker_events(ticker: str, facts: dict) -> pd.DataFrame:
    """
    Return DataFrame [announcement_date, eps, sue] for one ticker.
    announcement_date is the RAW SEC filing date. NOTE (2026-06): contrary to
    what this docstring used to claim, _quarterly_series does NOT apply a +1
    day availability shift (that shift happens in sector_model's feature
    builder, not in _quarterly_series itself). Consumers must therefore treat
    ann_date as not-yet-tradable on the day itself — run.py requires
    ann_date < rebalance_date.
    """
    eps = _quarterly_series(facts, _EPS_TAGS, unit="USD/shares")
    if eps.empty or len(eps) < _MIN_HISTORY:
        return pd.DataFrame()

    eps = eps.sort_index()
    df = pd.DataFrame({"eps": eps})
    # Seasonal random walk: expected = 4 quarters ago
    df["expected"]   = df["eps"].shift(4)
    df["unexpected"] = df["eps"] - df["expected"]
    # Standardise by trailing 8-quarter std of unexpected earnings
    df["sue_std"] = df["unexpected"].rolling(8, min_periods=4).std()
    df["sue"] = df["unexpected"] / df["sue_std"].replace(0, np.nan)
    df = df.dropna(subset=["sue"])
    df["sue"] = df["sue"].clip(-8, 8)   # cap extreme values
    return df[["eps", "sue"]].reset_index().rename(columns={"filed": "ann_date", "index": "ann_date"})


def build_events(
    tickers: List[str],
    cache_dir: Optional[str] = None,
    cik_map: Optional[Dict[str, str]] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the full earnings-events table across all tickers.
    Returns DataFrame with columns [ticker, ann_date, eps, sue].

    `end_date` (the backtest's requested coverage) triggers a STALENESS CHECK
    (2026-06), the PEAD counterpart to the factor-fundamentals guard: EDGAR
    returns all available history, but this table is cached unconditionally,
    so a cache built against an older end_date would silently omit newer
    earnings and leave the OOS period (2025+) with no events to trade. If the
    newest cached announcement is well behind the requested end_date, rebuild.
    """
    if cik_map is None:
        cik_map = fetch_cik_map(cache_dir)

    cache_path = Path(cache_dir) / "pead_events.parquet" if cache_dir else None
    if cache_path and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        stale = False
        if end_date is not None and len(cached):
            last = pd.to_datetime(cached["ann_date"]).max()
            # 90-day slack: the newest real filing always lags end_date a bit
            # (and end_date may sit slightly in the future).
            if last < pd.Timestamp(end_date) - pd.Timedelta(days=90):
                logger.info(
                    f"PEAD events cache stale (newest {last.date()} ≪ "
                    f"{end_date}) — rebuilding from EDGAR")
                stale = True
        if not stale:
            logger.info(f"Loading cached PEAD events from {cache_path}")
            return cached

    rows = []
    n = len(tickers)
    for i, ticker in enumerate(tickers):
        cik = cik_map.get(ticker.upper())
        if cik is None:
            continue
        if i > 0:
            time.sleep(_DELAY)
        facts = _fetch_facts(cik)
        if facts is None:
            continue
        ev = _ticker_events(ticker, facts)
        if not ev.empty:
            ev["ticker"] = ticker
            rows.append(ev)
        if (i + 1) % 50 == 0:
            logger.info(f"PEAD events: {i+1}/{n} tickers, {sum(len(r) for r in rows)} events so far")

    if not rows:
        return pd.DataFrame()
    events = pd.concat(rows, ignore_index=True)
    events["ann_date"] = pd.to_datetime(events["ann_date"])
    events = events.sort_values("ann_date").reset_index(drop=True)

    if cache_path:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        events.to_parquet(cache_path)
        logger.info(f"Cached {len(events)} events from {events.ticker.nunique()} tickers → {cache_path}")
    return events
