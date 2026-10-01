"""
Daily snapshots of data whose HISTORY is not free, banked now so it exists later.

  stock_options  option chains for 50 liquid single stocks, expiries ≤ 6 months
                 → snapshots/{date}/stocks/{TICKER}.parquet (same schema as the index chains)
                 Run PRE-CLOSE (with the index snapshot) so all chains share a timestamp regime.
  estimates      analyst data for current S&P 500 members: EPS and revenue consensus,
                 EPS trend (7/30/60/90-day-ago values), revisions, recommendation counts,
                 price targets, next earnings date. The point-in-time history is a paid
                 product (I/B/E/S).
                 → estimates/{date}.parquet, long format (ticker, table, period, field, value)
  etf_aum        total assets, shares outstanding, NAV and price for ~60 ETFs. Changes in
                 AUM net of returns = fund flows.
                 → etf_aum.csv, appended daily

Every part is isolated (one failing does not stop the others), every ticker gets retries
with backoff, and every part is idempotent per date (--force rewrites).

    python collect_extras.py --part stock_options
    python collect_extras.py --part estimates etf_aum
"""

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from collect_snapshots import snapshot_ticker

ROOT = Path(__file__).resolve().parent
SNAP, EST, AUM = ROOT / "snapshots", ROOT / "estimates", ROOT / "etf_aum.csv"
SP500_CACHE = ROOT.parent / "factor_model" / "cache" / "sp500_universe.csv"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

STOCK_OPTIONS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V", "MA", "UNH", "XOM", "LLY",
    "JNJ", "WMT", "PG", "HD", "COST", "NFLX", "AMD", "CRM", "ORCL", "ADBE", "INTC", "QCOM", "CSCO", "BAC",
    "WFC", "C", "GS", "MS", "BA", "CAT", "DIS", "KO", "PEP", "MRK", "PFE", "ABBV", "T", "VZ", "CVX", "COP",
    "UBER", "PYPL", "MU", "PLTR", "SMCI", "COIN",
]
ETFS = [
    "SPY", "IVV", "VOO", "VTI", "QQQ", "IWM", "DIA", "MDY", "RSP", "EFA", "VEA", "EEM", "VWO", "FXI", "EWJ",
    "AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "EMB", "MUB", "BIL",
    "GLD", "SLV", "USO", "UNG", "DBC", "VNQ",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC", "SMH", "KRE", "XBI", "ARKK",
    "TQQQ", "SQQQ", "UPRO", "SPXU", "SSO", "SDS", "TMF", "TMV", "SOXL", "SOXS", "UVXY", "SVXY", "VXX",
]


def retry(fn, what, tries=3, base=2.0):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                logger.warning(f"{what}: failed after {tries} tries ({e})")
                return None
            time.sleep(base * (2 ** k))


def today():
    return dt.date.today().isoformat()


# ─────────────────────────────────────────────────────────── stock options

def stock_options(force):
    d = SNAP / today() / "stocks"
    d.mkdir(parents=True, exist_ok=True)
    ok = 0
    for t in STOCK_OPTIONS:
        if retry(lambda: snapshot_ticker(t, d, force, max_days=183) or True, f"options {t}"):
            ok += 1
        time.sleep(0.3)
    logger.info(f"stock_options: {ok}/{len(STOCK_OPTIONS)} tickers → {d.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────── analyst estimates

def sp500_members():
    try:
        df = pd.read_html(WIKI, storage_options={"User-Agent": "Mozilla/5.0"})[0]
        syms = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        if len(syms) > 450:
            return syms
    except Exception as e:
        logger.warning(f"S&P 500 list from Wikipedia failed ({e}); using cached list")
    return pd.read_csv(SP500_CACHE)["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()


def _long(t, table, df):
    if df is None or (hasattr(df, "empty") and df.empty):
        return []
    if isinstance(df, dict):
        return [(t, table, "", k, v) for k, v in df.items()]
    df = df.reset_index()
    key = df.columns[0]
    rows = []
    for _, r in df.iterrows():
        for c in df.columns[1:]:
            rows.append((t, table, str(r[key]), str(c), r[c]))
    return rows


def _estimates_one(t):
    tk = yf.Ticker(t)
    rows = []
    for table in ["earnings_estimate", "revenue_estimate", "eps_trend", "eps_revisions", "recommendations"]:
        rows += _long(t, table, getattr(tk, table))
    rows += _long(t, "price_targets", tk.analyst_price_targets)
    cal = tk.calendar or {}
    for k in ["Earnings Date", "Earnings Average", "Earnings Low", "Earnings High"]:
        if k in cal:
            v = cal[k]
            rows.append((t, "calendar", "", k, ";".join(map(str, v)) if isinstance(v, list) else v))
    return rows


def estimates(force):
    EST.mkdir(exist_ok=True)
    out = EST / f"{today()}.parquet"
    if out.exists() and not force:
        logger.info("estimates: already collected today, skipped")
        return
    members = sp500_members()
    rows, bad = [], []
    for i, t in enumerate(members):
        r = retry(lambda: _estimates_one(t), f"estimates {t}")
        (rows.extend(r) if r else bad.append(t))
        time.sleep(0.25)
        if i % 100 == 0:
            logger.info(f"  estimates {i}/{len(members)}")
    df = pd.DataFrame(rows, columns=["ticker", "table", "period", "field", "value"])
    df["value"] = df["value"].astype(str)                       # mixed types: keep exact text
    df["fetched_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    df.to_parquet(out, index=False)
    logger.info(f"estimates: {df.ticker.nunique()}/{len(members)} tickers, {len(df):,} rows → {out.name}"
                + (f"; failed: {bad[:20]}{'…' if len(bad) > 20 else ''}" if bad else ""))


# ─────────────────────────────────────────────────────────── ETF assets / flows

def etf_aum(force):
    d = today()
    prev = pd.read_csv(AUM) if AUM.exists() else pd.DataFrame()
    if not prev.empty and (prev.date == d).any() and not force:
        logger.info("etf_aum: already collected today, skipped")
        return
    rows = []
    for t in ETFS:
        info = retry(lambda: yf.Ticker(t).info, f"aum {t}") or {}
        rows.append({"date": d, "ticker": t, "total_assets": info.get("totalAssets"),
                     "shares_outstanding": info.get("sharesOutstanding"), "nav": info.get("navPrice"),
                     "price": info.get("regularMarketPrice") or info.get("previousClose"),
                     "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        time.sleep(0.25)
    new = pd.DataFrame(rows)
    out = pd.concat([prev[prev.date != d] if not prev.empty else prev, new], ignore_index=True)
    out.to_csv(AUM, index=False)
    logger.info(f"etf_aum: {new.total_assets.notna().sum()}/{len(ETFS)} with total assets, "
                f"{new.shares_outstanding.notna().sum()} with shares outstanding → {AUM.name}")


PARTS = {"stock_options": stock_options, "estimates": estimates, "etf_aum": etf_aum}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=list(PARTS), choices=list(PARTS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for p in a.part:
        t0 = time.time()
        try:
            PARTS[p](a.force)
        except Exception as e:
            logger.error(f"{p}: {e}")
        logger.info(f"{p}: {time.time() - t0:.0f}s")
