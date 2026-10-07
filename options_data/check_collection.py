"""
Nightly completeness check: is every feed present for today (weekdays only)? If
anything is missing, post a macOS notification, so a gap is noticed the same night
rather than a week later.

    python check_collection.py            # check today
    python check_collection.py 2026-10-06 # check a given date
"""

import datetime as dt
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
INDEX = ["SPY", "QQQ", "IWM", "_SPX", "_VIX"]


def check(day: str) -> list:
    missing = []
    snap = ROOT / "snapshots" / day
    lost = [t for t in INDEX if not (snap / f"{t}.parquet").exists()]
    if lost:
        missing.append(f"index chains ({', '.join(lost)})")
    n_stocks = len(list((snap / "stocks").glob("*.parquet"))) if (snap / "stocks").exists() else 0
    if n_stocks < 45:
        missing.append(f"stock chains ({n_stocks}/50)")
    if not (ROOT / "intraday" / f"{day}.parquet").exists():
        missing.append("1-minute bars")
    if not (ROOT / "estimates" / f"{day}.parquet").exists():
        missing.append("analyst estimates")
    aum = ROOT / "etf_aum.csv"
    if not aum.exists() or not (pd.read_csv(aum, usecols=["date"]).date == day).any():
        missing.append("ETF assets")
    return missing


def notify(title: str, msg: str):
    subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "{title}" sound name "Basso"'],
                   check=False)


if __name__ == "__main__":
    day = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
    if dt.date.fromisoformat(day).weekday() >= 5:
        print(f"{day}: weekend, nothing expected")
        sys.exit(0)
    m = check(day)
    if m:
        print(f"{day}: MISSING {m}")
        notify("Market data collection incomplete",
               f"{day} missing: {'; '.join(m)}. Run: python options_data/catch_up.py")
    else:
        print(f"{day}: all feeds present")
