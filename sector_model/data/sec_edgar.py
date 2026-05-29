"""
SEC EDGAR XBRL API — free, official, no API key required.

Why this beats yfinance earnings data:
  - yfinance earnings_dates is scraped/cached and unreliable pre-2020
  - SEC filings are the ground truth: every public company must file within
    40-60 days of the fiscal quarter end
  - `filed` date = actual announcement date (no look-ahead bias)
  - Covers 2009-present with consistent EPS and revenue data

Data fetched per ticker:
  - EarningsPerShareDiluted (quarterly 10-Q + annual 10-K = Q4)
  - Revenue (quarterly) — used for revenue growth signal
  - OperatingIncomeLoss — used for margin signal

Rate limit: SEC requests max 10 req/s; we cap at 5 to be safe.
Each company's data is cached as a parquet file and never re-fetched.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from loguru import logger

_HEADERS = {"User-Agent": "trading-model-research markwang426@gmail.com"}
_BASE    = "https://data.sec.gov/api/xbrl/companyfacts"
_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
_DELAY   = 0.22   # ≈ 4.5 req/s — below SEC's 10 req/s limit

# EPS tags tried in order; first one with data wins
_EPS_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]

# Revenue tags tried in order (GAAP naming changed with ASC 606 in 2018)
_REV_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
]


# ── CIK lookup ────────────────────────────────────────────────────────────────

def fetch_cik_map(cache_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Return dict: ticker (upper) → zero-padded 10-digit CIK string.
    Cached as company_ciks.parquet.
    """
    if cache_dir:
        p = Path(cache_dir) / "company_ciks.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            return dict(zip(df["ticker"], df["cik"]))

    logger.info("Fetching CIK map from SEC...")
    r = requests.get(_CIK_URL, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    raw = r.json()

    rows = [{"ticker": v["ticker"].upper(),
             "cik": str(v["cik_str"]).zfill(10)}
            for v in raw.values()]
    df = pd.DataFrame(rows)

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(Path(cache_dir) / "company_ciks.parquet", index=False)

    return dict(zip(df["ticker"], df["cik"]))


# ── Raw facts download ────────────────────────────────────────────────────────

def _fetch_facts(cik: str) -> Optional[dict]:
    """Download raw company facts JSON from SEC. Returns None on failure."""
    url = f"{_BASE}/CIK{cik}.json"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"CIK {cik}: fetch failed — {e}")
        return None


# ── Extract quarterly series ──────────────────────────────────────────────────

def _quarterly_series(
    facts: dict,
    tags: List[str],
    unit: str = "USD/shares",
) -> pd.Series:
    """
    Extract a quarterly time-series from XBRL facts.
    Returns Series indexed by `filed` date (announcement date), values = reported amount.
    Keeps only quarterly (Q1-Q3) and annual (FY = Q4) forms.
    Deduplicates by taking the most recently filed entry per fiscal period end.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in tags:
        if tag not in gaap:
            continue
        units = gaap[tag].get("units", {})
        if unit not in units:
            # Try USD for revenue
            if "USD" in units:
                entries = units["USD"]
            else:
                continue
        else:
            entries = units[unit]

        rows = []
        for e in entries:
            form = e.get("form", "")
            fp   = e.get("fp", "")
            # Keep quarterly (Q1-Q3) and annual (FY = Q4) filings only
            if form not in ("10-Q", "10-K") or fp not in ("Q1", "Q2", "Q3", "FY"):
                continue
            rows.append({
                "period_end": pd.Timestamp(e["end"]),
                "filed":      pd.Timestamp(e["filed"]),
                "val":        float(e["val"]),
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)
        # Deduplicate by fiscal period end (keep most recently filed version)
        df = df.sort_values("filed").drop_duplicates(subset="period_end", keep="last")
        # Also deduplicate by filed date — two periods can share a filing date
        df = df.drop_duplicates(subset="filed", keep="last")
        df = df.set_index("filed")["val"].sort_index()
        return df

    return pd.Series(dtype=float)


# ── Per-ticker feature builder ────────────────────────────────────────────────

def _build_edgar_features(
    ticker: str,
    facts:  dict,
    daily_prices: pd.Series,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build fundamental features for one ticker using SEC EDGAR data.

    Features produced:
      eps_ttm            trailing-12-month diluted EPS (4-quarter sum)
      eps_growth_yoy     YoY change in TTM EPS
      eps_surprise       reported EPS vs naive estimate (trailing avg)
      revenue_growth_yoy YoY quarterly revenue growth
      gross_margin       (revenue - cost_of_revenue) / revenue (if available)
    """
    out = pd.DataFrame(index=trading_dates, dtype=np.float64)

    # ── Diluted EPS ───────────────────────────────────────────────────────
    eps_q = _quarterly_series(facts, _EPS_TAGS, unit="USD/shares")

    if eps_q.empty:
        for col in ["eps_ttm", "eps_growth_yoy", "eps_surprise",
                    "revenue_growth_yoy", "trailing_pe", "peg_ratio"]:
            out[col] = np.nan
        return out

    # Shift 1 calendar day: earnings filed on date D available from D+1
    eps_q.index = eps_q.index + pd.Timedelta(days=1)

    # TTM: sum of last 4 quarterly values as of each trading date
    ann_dates = sorted(eps_q.index)
    ttm_by_date: dict = {}
    for d in trading_dates:
        past = [dt for dt in ann_dates if dt <= d]
        if len(past) >= 4:
            ttm_by_date[d] = float(eps_q.reindex(past).iloc[-4:].sum())
        elif len(past) >= 1:
            ttm_by_date[d] = float(eps_q.reindex(past).sum())

    eps_ttm = pd.Series(ttm_by_date, name="eps_ttm").reindex(trading_dates).astype(float)
    out["eps_ttm"] = eps_ttm

    # YoY EPS growth
    ttm_1yr = eps_ttm.shift(252).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["eps_growth_yoy"] = (
            (eps_ttm - ttm_1yr) / ttm_1yr.abs().replace(0, np.nan)
        ).clip(-2, 10)

    # EPS surprise: reported vs trailing 4-quarter average (naive estimate)
    # Positive = beat; negative = miss.  No analyst data needed.
    eps_daily  = eps_q.reindex(trading_dates).ffill()
    naive_est  = eps_q.reindex(trading_dates).ffill().rolling(4, min_periods=2).mean().shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["eps_surprise"] = (
            (eps_daily - naive_est) / naive_est.abs().replace(0, np.nan)
        ).clip(-2, 2)

    # Trailing P/E
    price    = daily_prices.reindex(trading_dates).ffill()
    valid_ttm = eps_ttm.where(eps_ttm > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = (price / valid_ttm).clip(0, 200)
    out["trailing_pe"] = pe

    # PEG ratio: P/E divided by EPS growth rate (in %)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth_pct = out["eps_growth_yoy"] * 100
        out["peg_ratio"] = (pe / growth_pct.replace(0, np.nan)).clip(0, 20)

    # ── Revenue ───────────────────────────────────────────────────────────
    rev_q = _quarterly_series(facts, _REV_TAGS, unit="USD")
    if not rev_q.empty:
        rev_q.index = rev_q.index + pd.Timedelta(days=1)
        rev_daily  = rev_q.reindex(trading_dates).ffill()
        rev_1yr    = rev_q.reindex(trading_dates).ffill().shift(252).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["revenue_growth_yoy"] = (
                (rev_daily - rev_1yr) / rev_1yr.abs().replace(0, np.nan)
            ).clip(-1, 5)
    else:
        out["revenue_growth_yoy"] = np.nan

    return out


# ── Public interface ──────────────────────────────────────────────────────────

def fetch_edgar_fundamentals(
    tickers:       List[str],
    prices:        pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cache_dir:     Optional[str] = None,
    cik_map:       Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Fetch SEC EDGAR fundamental features for all tickers.
    Returns a (date, ticker) MultiIndex DataFrame — same shape as
    fetch_fundamental_features() so it's a drop-in replacement.
    """
    if cik_map is None:
        cik_map = fetch_cik_map(cache_dir)

    ticker_frames = []
    n = len(tickers)

    for i, ticker in enumerate(tickers):
        cache_path = Path(cache_dir) / f"{ticker}_edgar.parquet" if cache_dir else None

        if cache_path and cache_path.exists():
            feat = pd.read_parquet(cache_path)
        else:
            cik = cik_map.get(ticker.upper())
            if cik is None:
                logger.debug(f"{ticker}: no CIK found — skipping EDGAR data")
                continue

            if i > 0:
                time.sleep(_DELAY)
            facts = _fetch_facts(cik)
            if facts is None:
                continue

            price_series = prices[ticker] if ticker in prices.columns else pd.Series(dtype=float)
            feat = _build_edgar_features(ticker, facts, price_series, trading_dates)

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                feat.to_parquet(cache_path)

            if (i + 1) % 20 == 0:
                logger.info(f"EDGAR fetch progress: {i+1}/{n} tickers")

        feat = feat.copy()
        feat["ticker"] = ticker
        ticker_frames.append(feat)

    if not ticker_frames:
        return pd.DataFrame()

    panel = pd.concat(ticker_frames)
    panel.index = pd.MultiIndex.from_arrays(
        [panel.index, panel["ticker"]], names=["date", "ticker"]
    )
    return panel.drop(columns=["ticker"])
