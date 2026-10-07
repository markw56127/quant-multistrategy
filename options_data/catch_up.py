"""
One-command catch-up: run every collector now and fill whatever today is missing.
Each collector skips what it already has, so this is always safe to run.

    python options_data/catch_up.py

Captures after the 12:30 PT pre-close window are still useful (end-of-day values),
and every row keeps its true fetched_utc timestamp.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
STEPS = [
    [PY, "collect_snapshots.py"],
    [PY, "collect_extras.py", "--part", "stock_options"],
    [PY, "collect_intraday.py"],
    [PY, "collect_extras.py", "--part", "estimates", "etf_aum"],
    [PY, "check_collection.py"],
]

if __name__ == "__main__":
    for cmd in STEPS:
        print(f"→ {' '.join(Path(c).name for c in cmd[1:])}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=False)
