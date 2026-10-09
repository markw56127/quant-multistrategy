"""
Pre-registered forward-test signals (README.md), computed for one date from that day's
banked files only.

    python signals.py 2026-10-14        # writes signals/2026-10-14.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
EST = REPO / "options_data" / "estimates"
PARAMS = REPO / "vol_surface" / "results" / "params"


def _winz(x, lo=0.02, hi=0.98):
    return x.clip(x.quantile(lo), x.quantile(hi))


def _z(x):
    return (x - x.mean()) / x.std()


def f1_revisions(day: str) -> pd.Series:
    e = pd.read_parquet(EST / f"{day}.parquet")
    t = e[(e.table == "eps_trend") & e.period.isin(["0y", "+1y"]) & e.field.isin(["current", "30daysAgo"])]
    t = t.assign(v=pd.to_numeric(t.value, errors="coerce")).pivot_table(
        index=["ticker", "period"], columns="field", values="v")
    t = t[t["30daysAgo"].abs() > 1e-6]
    rev = ((t["current"] - t["30daysAgo"]) / t["30daysAgo"].abs()).unstack("period")
    zs = [_z(_winz(rev[p].dropna())) for p in ["0y", "+1y"] if p in rev]
    return pd.concat(zs, axis=1).mean(axis=1).rename("F1_revision")


def f2_f3_options(day: str) -> pd.DataFrame:
    p = pd.read_csv(PARAMS / f"{day}.csv")
    s = p[(p.kind == "stock") & (p.capture == "regular")].set_index("underlying")   # never after-hours quotes
    px = yf.download(list(s.index), end=pd.Timestamp(day) + pd.Timedelta(days=1), period=None,
                     start=pd.Timestamp(day) - pd.Timedelta(days=45), auto_adjust=True, progress=False)["Close"]
    r = np.log(px).diff().iloc[-21:]
    rv = r.std() * np.sqrt(252)
    return pd.DataFrame({"F2_skew": s.skew30, "F3_iv_minus_rv": s.atm30 - rv.reindex(s.index)})


def compute(day: str) -> pd.DataFrame:
    out = pd.concat([f1_revisions(day), f2_f3_options(day)], axis=1)
    out.index.name = "ticker"
    (ROOT / "signals").mkdir(exist_ok=True)
    out.to_csv(ROOT / "signals" / f"{day}.csv")
    return out


if __name__ == "__main__":
    print(compute(sys.argv[1]).describe().round(3))
