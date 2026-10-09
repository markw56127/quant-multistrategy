"""
Daily options-chain snapshot collector.

Historical options data is not free (OptionMetrics/ORATS/CBOE DataShop), so
this script banks our own: run it once per trading day and in six months a
real surface dataset exists — today's collection is next year's data. It has
no consumers yet by design; it exists so future projects (surface fitting,
vol-risk-premium research) have point-in-time data nobody can question.

What it stores, per ticker, per day:
  snapshots/{YYYY-MM-DD}/{ticker}.parquet — every listed expiry's full chain
  (calls+puts: strike, bid/ask/last, volume, open interest, implied vol) with
  the underlying spot and a UTC fetch timestamp on every row.
  snapshots/vix_term_structure.csv — one row per day of ^VIX9D/^VIX/^VIX3M
  closes (appended, deduped), since the indices are the cheap always-useful
  companion series.

Idempotent: a ticker already snapshotted today is skipped (--force overrides).
Yahoo's chains are 15-minute-delayed retail data — fine for building surface
history, not for market-making. Run in the last hour of the session if you
can (--force refreshes an earlier partial run); any consistent daily time
works, the timestamp column records the truth.

Schedule it (macOS, 15:30 ET example — adjust to your local time):
  crontab -e
  30 15 * * 1-5 cd /Users/markwang/Desktop/trading_model/options_data && /usr/bin/env python collect_snapshots.py >> collect.log 2>&1

Run manually:
    python collect_snapshots.py
    python collect_snapshots.py --tickers SPY QQQ --force
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "^SPX", "^VIX"]
VIX_INDICES = ["^VIX9D", "^VIX", "^VIX3M"]
ROOT = Path(__file__).resolve().parent / "snapshots"


def snapshot_ticker(ticker: str, day_dir: Path, force: bool, max_days: int = None) -> None:
    """max_days: keep only expiries within this many calendar days (None = all listed)."""
    out = day_dir / f"{ticker.replace('^', '_')}.parquet"
    if out.exists() and not force:
        logger.info(f"{ticker}: already snapshotted today — skipped (--force to refresh)")
        return
    t = yf.Ticker(ticker)
    expiries = t.options
    if max_days is not None:
        cutoff = dt.date.today() + dt.timedelta(days=max_days)
        expiries = [e for e in expiries if dt.date.fromisoformat(e) <= cutoff]
    if not expiries:
        logger.warning(f"{ticker}: no listed expiries returned — skipped")
        return
    spot = None
    for key in ("last_price", "lastPrice"):
        try:
            spot = float(t.fast_info[key])
            break
        except Exception:
            continue
    if spot is None:                       # index tickers often lack fast_info
        h = t.history(period="1d")
        spot = float(h["Close"].iloc[-1]) if not h.empty else None
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    frames = []
    for exp in expiries:
        ch = None
        for attempt in range(3):                       # transient Yahoo timeouts are common
            try:
                ch = t.option_chain(exp)
                break
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"{ticker} {exp}: fetch failed after 3 tries ({e}) — expiry skipped")
                else:
                    import time; time.sleep(2 * (attempt + 1))
        if ch is None:
            continue
        for side, df in (("call", ch.calls), ("put", ch.puts)):
            if df is None or df.empty:
                continue
            df = df.copy()
            df["type"], df["expiry"] = side, exp
            frames.append(df)
    if not frames:
        logger.warning(f"{ticker}: no chain data — nothing written")
        return
    full = pd.concat(frames, ignore_index=True)
    full["underlying"], full["spot"], full["fetched_utc"] = ticker, spot, ts
    full.to_parquet(out)
    logger.info(f"{ticker}: {len(full):,} contracts across {len(expiries)} expiries "
                f"(spot={spot}) → {out.relative_to(ROOT.parent)}")


def append_vix_term_structure() -> None:
    """Write-only: today's file holds the last 5 daily closes; readers combine and dedupe
    (load_vix_term_structure). Never reads earlier files: Desktop is iCloud-synced, and an
    offloaded file can't be read from a background job (EDEADLK)."""
    d = ROOT / "vix_term_structure"
    d.mkdir(parents=True, exist_ok=True)
    raw = yf.download(VIX_INDICES, period="5d", auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna(how="all")
    if close.empty:
        logger.warning("VIX indices: no data")
        return
    out = d / f"{dt.date.today().isoformat()}.csv"
    close.to_csv(out)
    logger.info(f"VIX term structure: {len(close)} days → {out.relative_to(ROOT.parent)}")


def load_vix_term_structure() -> pd.DataFrame:
    """All VIX term-structure data: the legacy single CSV plus the per-day files, deduped
    (the latest capture of each date wins)."""
    parts = [pd.read_csv(p, index_col=0, parse_dates=True)
             for p in [ROOT / "vix_term_structure.csv", *sorted((ROOT / "vix_term_structure").glob("*.csv"))]
             if p.exists()]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    day_dir = ROOT / dt.date.today().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    for tk in args.tickers:
        try:
            snapshot_ticker(tk, day_dir, args.force)
        except Exception as e:
            logger.error(f"{tk}: {e}")
    append_vix_term_structure()


if __name__ == "__main__":
    main()
