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


ROOT = Path(__file__).resolve().parent
INDEX = ["SPY", "QQQ", "IWM", "_SPX", "_VIX"]


def _after_hours(path: Path, day: str) -> bool:
    import pandas as pd
    ts = pd.Timestamp(pd.read_parquet(path, columns=["fetched_utc"]).fetched_utc.iloc[0]).tz_convert("America/New_York")
    return not (ts.date().isoformat() == day and (9, 30) <= (ts.hour, ts.minute) <= (16, 0))


def check(day: str) -> list:
    missing = []
    snap = ROOT / "snapshots" / day
    lost = [t for t in INDEX if not (snap / f"{t}.parquet").exists()]
    if lost:
        missing.append(f"index chains ({', '.join(lost)})")
    n_stocks = len(list((snap / "stocks").glob("*.parquet"))) if (snap / "stocks").exists() else 0
    if n_stocks < 45:
        missing.append(f"stock chains ({n_stocks}/50)")
    # chains must be captured during the session: after-hours quotes are stale (excluded from signals)
    for label, p in [("index chains", snap / "_SPX.parquet"), ("stock chains", snap / "stocks" / "AAPL.parquet")]:
        if p.exists() and _after_hours(p, day):
            missing.append(f"{label} captured AFTER HOURS")
    if not (ROOT / "intraday" / f"{day}.parquet").exists():
        missing.append("1-minute bars")
    if not (ROOT / "estimates" / f"{day}.parquet").exists():
        missing.append("analyst estimates")
    if not (ROOT / "etf_aum" / f"{day}.csv").exists():      # existence only: works on iCloud-offloaded files
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
