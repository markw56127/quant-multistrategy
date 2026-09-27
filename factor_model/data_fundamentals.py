"""
Balance-sheet and income fundamentals from SEC EDGAR for factor construction.

Columns (daily, point-in-time, one row per trading date):
  book_equity        StockholdersEquity (instant)
  total_assets       Assets (instant)
  net_income_ttm     NetIncomeLoss / ProfitLoss (trailing 4 quarters)
  gross_profit_ttm   GrossProfit (trailing 4 quarters)
  revenue_ttm        Revenues / ASC 606 revenue tags (trailing 4 quarters)
  shares_outstanding dei:EntityCommonStockSharesOutstanding, AS REPORTED. Use
                     shared/market_cap.py for market cap, never price x this
                     (SPLIT_FINDING.md).

Extraction is shared/edgar_pit.py: the earliest filing per period, available the day
after filing, and TTM flows built by differencing each tag's year-to-date chain.

History (2026-09-26): the original version kept only ~90-day flow facts. Q4 usually
exists only inside the 10-K's full-year figure, so its "last four quarters" often
spanned 15+ months or mixed periods (AMZN TTM net income read $0.2B at 2023-09
against ~$20B actual). It also back-filled shares before a company's first filing.
Both are fixed here; see SPLIT_FINDING.md, "Also found".
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sector_model"))
sys.path.insert(0, str(REPO / "shared"))
from data.sec_edgar import fetch_cik_map  # noqa: E402,F401  (re-exported for callers)
from edgar_pit import fetch as fetch_facts, ttm_flow, instant  # noqa: E402

NET_INCOME = ["NetIncomeLoss", "ProfitLoss"]
GROSS_PROFIT = ["GrossProfit"]
REVENUE = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
           "RevenueFromContractWithCustomerIncludingAssessedTax"]
EQUITY = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
ASSETS = ["Assets"]
SHARES = ["dei:SharesOutstanding"]


def _build(facts: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "book_equity": instant(facts, EQUITY, dates),
        "total_assets": instant(facts, ASSETS, dates),
        "net_income_ttm": ttm_flow(facts, NET_INCOME, dates),
        "gross_profit_ttm": ttm_flow(facts, GROSS_PROFIT, dates),
        "revenue_ttm": ttm_flow(facts, REVENUE, dates),
        "shares_outstanding": instant(facts, SHARES, dates),
    }, index=dates)


def fetch_factor_fundamentals(
    tickers: List[str],
    trading_dates: pd.DatetimeIndex,
    cache_dir: Optional[str] = None,
    cik_map: Optional[Dict[str, str]] = None,        # kept for signature compatibility
) -> pd.DataFrame:
    """(date, ticker) MultiIndex panel of point-in-time fundamentals.

    `cache_dir` is the per-ticker factor cache (e.g. factor_model/cache/factor_fund). The
    corrected frames live in the sibling `factor_fund_pit/` so they never mix with frames
    built by the old extraction; the CIK map is read from the parent directory."""
    root = Path(cache_dir).parent if cache_dir else REPO / "factor_model" / "cache"
    out_dir = root / "factor_fund_pit"
    out_dir.mkdir(parents=True, exist_ok=True)
    facts = fetch_facts(tickers, str(root))
    frames = []
    for t in tickers:
        p = out_dir / f"{t}.parquet"
        feat = pd.read_parquet(p) if p.exists() else None
        if feat is not None and (len(feat) == 0 or feat.index.max() < trading_dates.max() - pd.Timedelta(days=60)):
            feat = None                                   # built for an older calendar
        if feat is None:
            if facts[t].empty:
                continue
            feat = _build(facts[t], trading_dates)
            feat.to_parquet(p)
        frames.append(feat.assign(ticker=t))
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames)
    panel.index = pd.MultiIndex.from_arrays([panel.index, panel.pop("ticker")], names=["date", "ticker"])
    logger.info(f"Fundamentals (point-in-time): {panel.index.get_level_values('ticker').nunique()} tickers")
    return panel
