"""
Daily job: build the day's volatility surfaces, and on signal days compute the
pre-registered forward-test signals (README.md).

Signal day = Wednesday; if Wednesday's data is missing, Thursday. Write-only (new files per
day), so it is safe on the iCloud-synced Desktop. Scheduled by launchd
(com.markwang.forward-tests, weekdays 15:30 PT with a 19:15 retry).
"""

import datetime as dt
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "vol_surface"))
sys.path.insert(0, str(ROOT))

import run_daily  # noqa: E402
import signals  # noqa: E402

FIRST_SIGNAL_DAY = dt.date(2026, 10, 14)


def ready(day: str) -> bool:
    return (signals.EST / f"{day}.parquet").exists() and (signals.PARAMS / f"{day}.csv").exists()


def signal_day_for(today: dt.date):
    """Today's date if today is the week's signal day, else None."""
    if today < FIRST_SIGNAL_DAY or today.weekday() not in (2, 3):
        return None
    wed = today - dt.timedelta(days=today.weekday() - 2)
    if today.weekday() == 2:
        return today
    # Thursday: only if Wednesday produced no signal file and couldn't
    done = (ROOT / "signals" / f"{wed.isoformat()}.csv").exists()
    return None if done or ready(wed.isoformat()) else today


if __name__ == "__main__":
    today = dt.date.today()
    for d in run_daily.days():                       # surfaces for any day not yet processed
        run_daily.process(d)
    day = signal_day_for(today)
    if day is None:
        logger.info(f"{today}: not a signal day")
    elif (ROOT / "signals" / f"{day.isoformat()}.csv").exists():
        logger.info(f"{day}: signals already written")
    elif not ready(day.isoformat()):
        logger.warning(f"{day}: data not ready (estimates or surfaces missing); will retry")
    else:
        s = signals.compute(day.isoformat())
        logger.info(f"{day}: signals written for {s.notna().sum().to_dict()}")
