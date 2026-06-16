"""
Balance-sheet and income fundamentals from SEC EDGAR for factor construction.

Extends what sector_model/data/sec_edgar.py provides (EPS, revenue, shares)
with the items needed for proper value and quality factors:
  - StockholdersEquity  → book value (for Book-to-Price)
  - NetIncomeLoss       → ROE, net profit margin
  - Assets              → return on assets, gross profitability denominator
  - GrossProfit         → gross profitability (Novy-Marx 2013)

Flow items (net income, gross profit) are reported as quarterly amounts in
10-Q and as full-year amounts in 10-K. We separate them by the reporting
period duration (≈90 days = quarterly, ≈365 days = annual) so TTM sums don't
double-count the annual figure. Balance-sheet items (equity, assets) are
point-in-time snapshots and need no such handling.

All series are indexed by FILING date (+1 day) so there is no look-ahead bias.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Reuse the SEC fetching primitives from sector_model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sector_model"))
from data.sec_edgar import _fetch_facts, fetch_cik_map, _shares_outstanding  # noqa: E402

_DELAY = 0.22  # SEC rate limit courtesy

# Flow items: reported as quarterly (10-Q) and annual (10-K) amounts
_NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss"]
_GROSS_PROFIT_TAGS = ["GrossProfit"]
_REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
]
# Point-in-time balance sheet items
_EQUITY_TAGS = ["StockholdersEquity"]
_ASSETS_TAGS = ["Assets"]


def _flow_quarterly(facts: dict, tags: List[str]) -> pd.Series:
    """
    Extract a quarterly FLOW series (net income, gross profit, revenue) indexed
    by filing date. Separates true quarterly entries (~90-day duration) from
    annual entries (~365 days) and keeps only quarterly amounts so a trailing
    4-quarter sum gives a clean TTM figure.

    Tags are MERGED across all candidates rather than picking the first
    non-empty one — GAAP tag names change over time (e.g. ASC 606 in 2018
    switched revenue from `Revenues` to `RevenueFromContractWith...`). Merging
    by period_end and preferring the most recently filed value gives continuous
    coverage across the transition.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for tag in tags:
        if tag not in gaap:
            continue
        for e in gaap[tag].get("units", {}).get("USD", []):
            if e.get("form", "") not in ("10-Q", "10-K"):
                continue
            start, end = e.get("start"), e.get("end")
            if start is None or end is None:
                continue
            dur = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if not (60 <= dur <= 120):   # quarterly durations only
                continue
            rows.append({
                "period_end": pd.Timestamp(end),
                "filed":      pd.Timestamp(e["filed"]),
                "val":        float(e["val"]),
            })
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows).sort_values("filed")
    df = df.drop_duplicates(subset="period_end", keep="last")
    df = df.drop_duplicates(subset="filed", keep="last")
    s = df.set_index("filed")["val"].sort_index()
    s.index = s.index + pd.Timedelta(days=1)
    return s


def _instant_quarterly(facts: dict, tags: List[str]) -> pd.Series:
    """
    Extract a point-in-time BALANCE-SHEET series (equity, assets) indexed by
    filing date. These are instants (no duration), reported each 10-Q/10-K.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for tag in tags:
        if tag not in gaap:
            continue
        for e in gaap[tag].get("units", {}).get("USD", []):
            if e.get("form", "") not in ("10-Q", "10-K"):
                continue
            if e.get("end") is None:
                continue
            rows.append({
                "period_end": pd.Timestamp(e["end"]),
                "filed":      pd.Timestamp(e["filed"]),
                "val":        float(e["val"]),
            })
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows).sort_values("filed")
    df = df.drop_duplicates(subset="period_end", keep="last")
    df = df.drop_duplicates(subset="filed", keep="last")
    s = df.set_index("filed")["val"].sort_index()
    s.index = s.index + pd.Timedelta(days=1)
    return s


def _ttm(quarterly: pd.Series, trading_dates: pd.DatetimeIndex) -> pd.Series:
    """Trailing-12-month sum of a quarterly flow series, as of each trading date."""
    if quarterly.empty:
        return pd.Series(np.nan, index=trading_dates)
    ann = sorted(quarterly.index)
    out: dict = {}
    for d in trading_dates:
        past = [dt for dt in ann if dt <= d]
        if len(past) >= 4:
            out[d] = float(quarterly.reindex(past).iloc[-4:].sum())
        elif len(past) >= 1:
            out[d] = float(quarterly.reindex(past).sum()) * (4.0 / len(past))  # annualise
    return pd.Series(out, name="ttm").reindex(trading_dates).astype(float)


def _build_factor_fundamentals(
    ticker: str,
    facts: dict,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Per-ticker raw fundamental series needed by factors.py."""
    out = pd.DataFrame(index=trading_dates, dtype=np.float64)

    # Point-in-time balance sheet (forward-fill between filings)
    equity = _instant_quarterly(facts, _EQUITY_TAGS)
    assets = _instant_quarterly(facts, _ASSETS_TAGS)
    out["book_equity"] = equity.reindex(trading_dates).ffill() if not equity.empty else np.nan
    out["total_assets"] = assets.reindex(trading_dates).ffill() if not assets.empty else np.nan

    # TTM flow items
    out["net_income_ttm"]  = _ttm(_flow_quarterly(facts, _NET_INCOME_TAGS), trading_dates)
    out["gross_profit_ttm"] = _ttm(_flow_quarterly(facts, _GROSS_PROFIT_TAGS), trading_dates)
    out["revenue_ttm"]     = _ttm(_flow_quarterly(facts, _REVENUE_TAGS), trading_dates)

    # Shares outstanding (point-in-time, for market cap)
    shares = _shares_outstanding(facts)
    out["shares_outstanding"] = (
        shares.reindex(trading_dates).ffill().bfill() if not shares.empty else np.nan
    )

    return out


def fetch_factor_fundamentals(
    tickers: List[str],
    trading_dates: pd.DatetimeIndex,
    cache_dir: Optional[str] = None,
    cik_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Fetch balance-sheet/income fundamentals for all tickers.
    Returns a (date, ticker) MultiIndex DataFrame.
    """
    if cik_map is None:
        cik_map = fetch_cik_map(cache_dir)

    frames = []
    n = len(tickers)
    for i, ticker in enumerate(tickers):
        cache_path = Path(cache_dir) / f"{ticker}_factor_fund.parquet" if cache_dir else None
        feat = None
        if cache_path and cache_path.exists():
            feat = pd.read_parquet(cache_path)
            # STALENESS CHECK (2026-06): cached frames are reindexed to the
            # trading calendar at BUILD time. If the requested calendar now
            # extends well past the cached frame's last date, the cache was
            # built against an older end_date and must be rebuilt — otherwise
            # 2025+ rows would silently carry missing/stale fundamentals.
            if len(feat) == 0 or feat.index.max() < trading_dates.max() - pd.Timedelta(days=60):
                logger.debug(f"  {ticker}: fundamentals cache stale — rebuilding from EDGAR")
                feat = None
        if feat is None:
            cik = cik_map.get(ticker.upper())
            if cik is None:
                continue
            if i > 0:
                time.sleep(_DELAY)
            facts = _fetch_facts(cik)
            if facts is None:
                continue
            feat = _build_factor_fundamentals(ticker, facts, trading_dates)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                feat.to_parquet(cache_path)
            if (i + 1) % 25 == 0:
                logger.info(f"Factor fundamentals: {i+1}/{n} tickers")

        feat = feat.copy()
        feat["ticker"] = ticker
        frames.append(feat)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames)
    panel.index = pd.MultiIndex.from_arrays(
        [panel.index, panel["ticker"]], names=["date", "ticker"]
    )
    return panel.drop(columns=["ticker"])
