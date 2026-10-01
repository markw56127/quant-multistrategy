"""
Daily 1-minute bar collector. Yahoo serves 1-minute bars only for the last ~8 trading
days, so this history disappears unless it is banked.

Each run downloads the last 7 days for every ticker and writes one file per COMPLETED
trading day that is not already on disk:
    intraday/{YYYY-MM-DD}.parquet   long format: ts_utc, ticker, open, high, low, close, volume

Because each run looks back a week, a missed run (Mac asleep or off) is filled on the next
run, as long as the gap is shorter than ~7 trading days. Today's bars are written only after
the close (16:00 ET). Scheduled after the close by launchd (com.markwang.intraday-bars).

Universe: the broad ETFs, rates/credit/commodities, sector SPDRs, and leveraged/inverse ETFs.
The leveraged funds must rebalance near the close every day, which is the intraday
forced-flow mechanism this data exists to test.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "intraday"
TICKERS = [
    # broad equity / rates / credit / commodities
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VNQ", "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "UNG",
    # sector SPDRs
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC", "SMH",
    # leveraged / inverse: daily rebalancing near the close
    "TQQQ", "SQQQ", "UPRO", "SPXU", "SSO", "SDS", "TMF", "TMV", "SOXL", "SOXS",
    # volatility ETPs
    "UVXY", "SVXY", "VXX",
]


def fetch(tickers):
    raw = yf.download(tickers, period="7d", interval="1m", auto_adjust=False, prepost=False,
                      progress=False, group_by="column", threads=True)
    if raw.empty:
        return pd.DataFrame()
    df = raw[["Open", "High", "Low", "Close", "Volume"]].stack(level=1, future_stack=True).dropna(how="all")
    df.columns = [c.lower() for c in df.columns]
    df.index.names = ["ts_utc", "ticker"]
    df = df.reset_index()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["date_et"] = df["ts_utc"].dt.tz_convert("America/New_York").dt.date.astype(str)
    return df


def main(force=False):
    OUT.mkdir(exist_ok=True)
    df = fetch(TICKERS)
    if df.empty:
        logger.error("no intraday data returned")
        return
    now_et = datetime.now(timezone.utc).astimezone(pd.Timestamp.now(tz="America/New_York").tz)
    today_et = now_et.date().isoformat()
    closed_today = now_et.hour >= 16
    written = []
    for d, g in df.groupby("date_et"):
        if d == today_et and not closed_today:
            continue                                         # day not complete yet
        p = OUT / f"{d}.parquet"
        if p.exists() and not force:
            continue
        g.drop(columns="date_et").sort_values(["ticker", "ts_utc"]).to_parquet(p, index=False)
        written.append(f"{d} ({g.ticker.nunique()} tickers, {len(g):,} bars)")
    missing = sorted(set(TICKERS) - set(df.ticker.unique()))
    if missing:
        logger.warning(f"no bars for: {missing}")
    logger.info("wrote " + (", ".join(written) if written else "nothing new (all days already on disk)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rewrite days already on disk")
    main(ap.parse_args().force)
