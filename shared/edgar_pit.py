"""
Point-in-time EDGAR fundamentals, shared by factor_model and alpha_research. Facts
are cached in long format, so new characteristics don't need a re-fetch.

Replaces factor_model's original TTM extraction, which kept only ~90-day facts. Q4
usually exists only inside the 10-K's full-year figure, so the "last four quarters"
silently spanned 15+ months. For example, AMZN's TTM net income read $0.2B at
2023-09 when it was ~$20B.

Per ticker, every USD fact for the tags below is stored as rows of
(tag, start, end, filed, val, form). Two extractors turn them into point-in-time
series:

  ttm_flow(...)  trailing-12-month FLOW (cash flow, R&D, net income), from each
                 filing's own numbers: a 10-K's fiscal year, or a 10-Q's
                 last FY + YTD - prior-year YTD (the 10-Q's comparative column, so
                 both YTDs share one basis through restatements). Available the day
                 after that filing.
  instant(...)   balance-sheet INSTANT (assets), available the day after filing.

Point-in-time rule: nothing is used before the day after it was filed. A TTM uses the
numbers as known at its filing date, so a later restatement never reaches back into
history.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sector_model"))
from data.sec_edgar import _fetch_facts, fetch_cik_map  # noqa: E402

CACHE = REPO / "factor_model" / "cache" / "edgar_pit"
TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    "ResearchAndDevelopmentExpense",
    "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    "NetIncomeLoss", "ProfitLoss",
    "Assets",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "DepreciationDepletionAndAmortization",
    "PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
    "PaymentsForRepurchaseOfCommonStock",
    "LongTermDebt", "LongTermDebtNoncurrent",
    "AssetsCurrent", "LiabilitiesCurrent", "CashAndCashEquivalentsAtCarryingValue",
    "SellingGeneralAndAdministrativeExpense", "InventoryNet",
    # inputs to the existing value/quality factors, re-derived point-in-time
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "GrossProfit", "CostOfRevenue", "CostOfGoodsAndServicesSold",
    "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}
DELAY = 0.22


def _extract(facts: dict) -> pd.DataFrame:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for tag in TAGS:
        for e in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if e.get("form") not in FORMS or e.get("end") is None:
                continue
            rows.append({"tag": tag, "start": e.get("start"), "end": e["end"],
                         "filed": e["filed"], "val": float(e["val"]), "form": e["form"]})
    # reported common shares outstanding (cover page), with its real filing date
    for e in facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}) \
            .get("units", {}).get("shares", []):
        if e.get("form") in FORMS and e.get("end") is not None:
            rows.append({"tag": "dei:SharesOutstanding", "start": None, "end": e["end"],
                         "filed": e["filed"], "val": float(e["val"]), "form": e["form"]})
    df = pd.DataFrame(rows, columns=["tag", "start", "end", "filed", "val", "form"])
    for c in ["start", "end", "filed"]:
        df[c] = pd.to_datetime(df[c])
    return df


def fetch(tickers, cik_cache_dir) -> dict:
    """Fetch (or load cached) long-format facts for each ticker."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cik_map = fetch_cik_map(cache_dir=cik_cache_dir)
    out, missing = {}, []
    todo = [t for t in tickers if not (CACHE / f"{t}.parquet").exists()]
    if todo:
        logger.info(f"EDGAR: fetching {len(todo)} tickers ({len(tickers) - len(todo)} cached)")
    for i, t in enumerate(todo):
        cik = cik_map.get(t) or cik_map.get(t.replace("-", "."))
        facts = _fetch_facts(cik) if cik else None
        time.sleep(DELAY)
        df = _extract(facts) if facts else pd.DataFrame(columns=["tag", "start", "end", "filed", "val", "form"])
        df.to_parquet(CACHE / f"{t}.parquet")
        if i % 50 == 0:
            logger.info(f"  {i}/{len(todo)}")
    for t in tickers:
        df = pd.read_parquet(CACHE / f"{t}.parquet")
        if df.empty:
            missing.append(t)
        out[t] = df
    if missing:
        logger.warning(f"EDGAR: no facts for {len(missing)} tickers")
    return out


def _first_filed(df: pd.DataFrame, tags) -> pd.DataFrame:
    """Rows for `tags`, one per (start, end): earliest filing; earlier tag in list wins ties."""
    d = df[df.tag.isin(tags)].copy()
    if d.empty:
        return d
    d["prio"] = d.tag.map({t: i for i, t in enumerate(tags)})
    d = d.sort_values(["filed", "prio"])
    return d.drop_duplicates(subset=["start", "end"], keep="first")


def _ttm_by_filing(d: pd.DataFrame) -> pd.DataFrame:
    """TTM for ONE tag at each filing, from that filing's own numbers:
        10-K:  TTM = the fiscal year it reports
        10-Q:  TTM = last fiscal year + current YTD - prior-year YTD
    Both YTD figures come from the same 10-Q (the comparative column), so they share one
    basis even when the company has restated (JNJ after the Kenvue spin-off). The fiscal
    year is the latest-filed value for it known at that filing. If a 10-Q lacks the
    comparative, the latest-filed prior-year YTD known by then is used."""
    d = d.dropna(subset=["start"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["filed", "end", "ttm"])
    d["dur"] = (d["end"] - d["start"]).dt.days
    d = d[(d.dur >= 60) & (d.dur <= 380)]
    out = []
    for F, rows in d.groupby("filed"):
        cur = rows[rows.end == rows.end.max()]
        annual = cur[cur.dur >= 350]
        if len(annual):
            out.append((F, cur.end.iloc[0], annual.val.iloc[0]))
            continue
        ytd = cur.sort_values("dur").iloc[-1]                          # longest current-period duration = YTD
        known = d[d.filed <= F]
        def latest(mask):
            m = known[mask]
            return m.sort_values("filed").val.iloc[-1] if len(m) else None
        prior = latest((abs((known.end - (ytd.end - YEAR)).dt.days) <= 10) & (abs(known.dur - ytd.dur) <= 10))
        fy = latest((abs((known.end - (ytd.start - pd.Timedelta(days=1))).dt.days) <= 10) & (known.dur >= 350))
        if prior is not None and fy is not None:
            out.append((F, ytd.end, fy + ytd.val - prior))
    return pd.DataFrame(out, columns=["filed", "end", "ttm"])


YEAR = pd.Timedelta(days=365)


def ttm_flow(df: pd.DataFrame, tags, dates: pd.DatetimeIndex, stale_days: int = 400) -> pd.Series:
    """Trailing-12-month flow as of each date, available the day after the filing that
    completes it. When several tags yield a value at the same filing, the tag the company
    reports in the most filings wins, then list order. JNJ files a small 10-K-only
    `ResearchAndDevelopmentExpense` beside its main R&D line, so list order alone picks
    the wrong one."""
    parts = []
    for prio, tag in enumerate(tags):
        d = df[df.tag == tag]
        if d.empty:
            continue
        t = _ttm_by_filing(d)
        if t.empty:
            continue
        t["freq"], t["prio"] = -d.filed.nunique(), prio
        parts.append(t)
    if not parts:
        return pd.Series(np.nan, index=dates)
    t = pd.concat(parts).sort_values(["filed", "freq", "prio"]).drop_duplicates("filed", keep="first")
    t = t[t.end >= t.end.cummax()]                                      # never step back to an older period
    s = pd.Series(t.ttm.values, index=t.filed + pd.Timedelta(days=1))
    s = s[~s.index.duplicated(keep="last")]
    out = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    age = pd.Series(s.index, index=s.index).reindex(s.index.union(dates)).ffill().reindex(dates)
    return out.where((pd.Series(dates, index=dates) - age).dt.days <= stale_days)


def instant(df: pd.DataFrame, tags, dates: pd.DatetimeIndex, stale_days: int = 400) -> pd.Series:
    """Balance-sheet value as of each date: latest period end among filings made by then."""
    d = _first_filed(df, tags)
    if d.empty:
        return pd.Series(np.nan, index=dates)
    d = d.assign(avail=d.filed + pd.Timedelta(days=1)).sort_values(["avail", "end"])
    # a filing can include prior-period comparatives: keep, per availability date, the latest period end
    d = d.drop_duplicates("avail", keep="last")
    d = d[d.end >= d.end.cummax()]                                  # never step back to an older period
    s = d.set_index("avail")["val"]
    out = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    age = pd.Series(s.index, index=s.index).reindex(s.index.union(dates)).ffill().reindex(dates)
    return out.where((pd.Series(dates, index=dates) - age).dt.days <= stale_days)
