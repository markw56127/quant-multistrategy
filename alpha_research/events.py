"""
C6: index-deletion rebound. Event study on S&P 500 removals (pre-registered in README.md).

Day 0 is the last day a name is an index member: index funds sell at that close. The
cumulative abnormal return is the name's return from close(day 0) to close(day +h),
minus the equal-weight return of that day's members over the same window. Removals
caused by delisting or acquisition (price series ends within 5 trading days of day 0)
are excluded: they cannot be traded, and their removal is a corporate event, not
index mechanics.

Inference clusters by calendar month: event CARs are averaged within each month of
day 0, and the t-stat is taken over those monthly means, because removals come in
batches at quarterly rebalances.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "shared"))
from universe_pit import membership_matrix  # noqa: E402

CACHE = REPO / "factor_model" / "cache"
PRICES = CACHE / "prices_sf_2015-01-01_2026-06-30.parquet"
DEV_END = pd.Timestamp("2025-01-01")
DELIST_WINDOW = 5


def deletion_events(prices, members):
    m = members.reindex(prices.index).fillna(False).astype(bool)
    removed = m.shift(1, fill_value=False) & ~m               # member yesterday, not today
    ev = []
    for d, row in removed.iterrows():
        for t in row.index[row.values]:
            i0 = prices.index.get_loc(d) - 1                  # day 0: last day as member
            if i0 < 1 or t not in prices.columns:
                continue
            ev.append({"ticker": t, "i0": i0, "day0": prices.index[i0]})
    return pd.DataFrame(ev)


def car(prices, members, ev, h):
    m = members.reindex(prices.index).fillna(False).astype(bool)
    out = []
    for e in ev.itertuples():
        p = prices[e.ticker]
        last = p.last_valid_index()
        if last is None or prices.index.get_loc(last) <= e.i0 + DELIST_WINDOW:
            continue                                           # delisted / acquired: not index mechanics
        i1 = e.i0 + h
        if i1 >= len(prices) or np.isnan(p.iloc[e.i0]) or np.isnan(p.iloc[i1]):
            continue
        r = p.iloc[i1] / p.iloc[e.i0] - 1
        mem = m.iloc[e.i0]
        cols = mem.index[mem.values].intersection(prices.columns)
        bench = (prices[cols].iloc[i1] / prices[cols].iloc[e.i0] - 1).mean()
        out.append({"ticker": e.ticker, "day0": e.day0, "end": prices.index[i1], "car": r - bench})
    return pd.DataFrame(out)


def month_clustered(df):
    mm = df.groupby(df.day0.dt.to_period("M"))["car"].mean()
    t = mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))
    return {"mean_car": float(df.car.mean()), "n_events": len(df), "n_months": len(mm),
            "t": float(t), "p_one_sided": float(stats.t.sf(t, len(mm) - 1))}   # predicted sign: +


def run(split=DEV_END):
    prices = pd.read_parquet(PRICES)
    members = membership_matrix(prices.index, cache_dir=str(CACHE))
    ev = deletion_events(prices, members)
    res = {}
    for h in [21, 63]:
        c = car(prices, members, ev, h)
        dev = c[c.end < split]
        res[h] = month_clustered(dev)
        res[h]["n_raw_removals"] = int((ev.day0 < split).sum())
    return res


if __name__ == "__main__":
    for h, r in run().items():
        print(f"CAR[+1,+{h}] dev: mean {r['mean_car']:+.2%}  n={r['n_events']} events / {r['n_months']} months "
              f"(of {r['n_raw_removals']} raw removals)  t={r['t']:+.2f}  p(one-sided)={r['p_one_sided']:.4f}")
