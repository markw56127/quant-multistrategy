"""
Fetch individual futures-contract daily bars from Databento (GLBX.MDP3).

This is step 1 of resolving TREND_FINDING's 2026-07 addendum: the free "=F"
series cannot settle the sleeve's OOS sign, so we buy clean per-contract data
and build our own back-adjusted continuous series (step 2:
build_backadjusted.py).

Setup:
  1. Create an account at databento.com (new accounts include free credits;
     this pull is daily bars only — the preflight below prints the exact cost
     and aborts unless it is trivial or you pass --yes).
  2. Put DATABENTO_API_KEY=db-... in the repo's .env (see .env.example).
  3. pip install databento
  4. python fetch_databento.py            # preflight + fetch → cache/contracts_daily.parquet
     python fetch_databento.py --yes      # skip the cost confirmation

Output: long-format parquet with columns [date, root, symbol, close, volume]
covering every outright contract of the 8 roots in the dev-selected universe.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOTS = ["ES", "NQ", "YM", "RTY", "ZT", "ZF", "ZN", "ZB"]
DATASET = "GLBX.MDP3"
START, END = "2015-01-01", "2026-06-30"
OUT = Path("cache/contracts_daily.parquet")
COST_ABORT_USD = 10.0          # refuse without --yes above this
OUTRIGHT_RE = re.compile(r"^([A-Z]{2,3})([FGHJKMNQUVXZ])(\d{1,2})$")  # e.g. ESZ5, ZNH24


def load_api_key() -> str:
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        env = Path(__file__).resolve().parent.parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("DATABENTO_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key or key.startswith("your_"):
        sys.exit("DATABENTO_API_KEY not set — add it to .env (see .env.example). "
                 "Get one at databento.com; daily bars for this pull cost single-digit "
                 "dollars and signup credits typically cover it.")
    return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="skip cost confirmation")
    args = ap.parse_args()

    try:
        import databento as db
    except ImportError:
        sys.exit("pip install databento")

    client = db.Historical(load_api_key())
    symbols = [f"{r}.FUT" for r in ROOTS]

    cost = client.metadata.get_cost(
        dataset=DATASET, symbols=symbols, stype_in="parent",
        schema="ohlcv-1d", start=START, end=END,
    )
    logger.info(f"Databento quoted cost for this pull: ${cost:.2f}")
    if cost > COST_ABORT_USD and not args.yes:
        sys.exit(f"Cost ${cost:.2f} > ${COST_ABORT_USD:.0f} guard — rerun with --yes to proceed.")

    logger.info(f"Fetching {DATASET} ohlcv-1d for {ROOTS} [{START} → {END}]...")
    store = client.timeseries.get_range(
        dataset=DATASET, symbols=symbols, stype_in="parent",
        schema="ohlcv-1d", start=START, end=END,
    )
    df = store.to_df()          # index ts_event; columns incl. symbol, close, volume

    df = df.reset_index()
    df["date"] = pd.to_datetime(df["ts_event"]).dt.tz_localize(None).dt.normalize()
    # keep outrights only (parent symbology also returns calendar spreads)
    m = df["symbol"].str.match(OUTRIGHT_RE)
    df = df[m]
    df["root"] = df["symbol"].str.extract(OUTRIGHT_RE)[0]
    df = df[df["root"].isin(ROOTS)]
    out = df[["date", "root", "symbol", "close", "volume"]].sort_values(["root", "date"])

    OUT.parent.mkdir(exist_ok=True)
    out.to_parquet(OUT)
    logger.info(f"Saved {len(out):,} contract-days "
                f"({out['symbol'].nunique()} contracts) → {OUT}")
    logger.info("Next: python build_backadjusted.py")


if __name__ == "__main__":
    main()
