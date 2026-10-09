"""
Build every collected surface and append its daily summary parameters.

    python run_daily.py            # all days not yet processed
    python run_daily.py --rebuild  # reprocess everything

Outputs (write-only per day, so it is safe on the iCloud-synced Desktop):
    results/params/{date}.csv   one row per underlying: ATM 30/90d vol, skew, curvature, term slope,
                                fit quality and no-arbitrage counts
    results/svi/{date}/{underlying}.csv   per-expiry forward, discount, SVI parameters
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from surface import build_surface, SNAP

ROOT = Path(__file__).resolve().parent
OUT_P, OUT_S = ROOT / "results" / "params", ROOT / "results" / "svi"
INDEX = {"SPX": "_SPX.parquet", "SPY": "SPY.parquet", "QQQ": "QQQ.parquet", "IWM": "IWM.parquet"}
FALLBACK_RATE = 0.04


def days():
    return sorted(p.name for p in SNAP.iterdir() if p.is_dir() and p.name[:2] == "20")


def process(day, rebuild=False):
    out = OUT_P / f"{day}.csv"
    if out.exists() and not rebuild:
        return False
    jobs = [(u, SNAP / day / f) for u, f in INDEX.items()]
    jobs += [(p.stem, p) for p in sorted((SNAP / day / "stocks").glob("*.parquet"))] if (SNAP / day / "stocks").exists() else []
    rows = []
    (OUT_S / day).mkdir(parents=True, exist_ok=True)
    for und, path in jobs:
        if not path.exists():
            continue
        try:
            s = build_surface(path, und, day, FALLBACK_RATE, max_days=400 if und in INDEX else 190)
            if s.expiries.empty or s.expiries.a.notna().sum() < 2:
                logger.warning(f"{day} {und}: too few fitted expiries")
                continue
            rows.append({**s.summary(), "kind": "index" if und in INDEX else "stock"})
            s.expiries.to_csv(OUT_S / day / f"{und}.csv", index=False)
        except Exception as e:
            logger.warning(f"{day} {und}: {e}")
    if rows:
        OUT_P.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info(f"{day}: {len(rows)} surfaces")
    return True


def resummarize(day):
    """Recompute daily summaries from the saved per-expiry SVI fits (no refitting)."""
    from surface import Surface
    old = pd.read_csv(OUT_P / f"{day}.csv").set_index("underlying")
    rows = []
    for f in sorted((OUT_S / day).glob("*.csv")):
        und = f.stem
        if und not in old.index:
            continue
        raw = pd.read_parquet(SNAP / day / (INDEX[und] if und in INDEX else f"stocks/{und}.parquet"),
                              columns=["fetched_utc"])
        asof = pd.Timestamp(raw.fetched_utc.iloc[0]).tz_convert("America/New_York")
        regular = asof.date().isoformat() == day and (9, 30) <= (asof.hour, asof.minute) <= (16, 15) and asof.weekday() < 5
        s = Surface(und, day, float(old.loc[und, "spot"]), "regular" if regular else "after_hours",
                    pd.read_csv(f), pd.DataFrame(index=range(int(old.loc[und, "n_points"]))))
        rows.append({**s.summary(), "kind": old.loc[und, "kind"]})
    pd.DataFrame(rows).to_csv(OUT_P / f"{day}.csv", index=False)


def load_params():
    return pd.concat([pd.read_csv(p) for p in sorted(OUT_P.glob("*.csv"))], ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--resummarize", action="store_true", help="recompute summaries from saved SVI fits")
    a = ap.parse_args()
    for d in days():
        if a.resummarize and (OUT_P / f"{d}.csv").exists():
            resummarize(d)
        else:
            process(d, a.rebuild)
