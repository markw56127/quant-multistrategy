"""
Study 2: Treasury month-end index extension. Pre-registered in README.md; DEV ONLY.

Spread = TLT - SHY total return over the last W trading days of each month (W = 2
primary). H1: mean > 0. H2: larger in refunding months (Feb/May/Aug/Nov). Holm over
{H1, H2}.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "treasury_px.parquet"
DEV_START, DEV_END = pd.Timestamp("2003-01-01"), pd.Timestamp("2025-01-01")
HALVES = {"2003-13": ("2003-01-01", "2014-01-01"), "2014-24": ("2014-01-01", "2025-01-01")}
W_PRIMARY, REFUNDING = 2, {2, 5, 8, 11}
COST = 4 * 0.0002
FWER = 0.05


def prices():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    px = yf.download(["TLT", "IEF", "SHY"], start="2002-08-01", auto_adjust=True, progress=False)["Close"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(CACHE)
    return px


def windows(px, long="TLT", short="SHY", w=W_PRIMARY, end_offset=0):
    """Spread return over the w trading days ending `end_offset` days before each month's
    last trading day. One row per month, labelled by that last trading day."""
    lp = np.log(px[[long, short]].dropna())
    days = lp.index
    s = days.to_series()
    month_end = days[s.dt.to_period("M").ne(s.shift(-1).dt.to_period("M"))]
    month_end = month_end[month_end < days[-1]]              # the data's last day is not a completed month-end
    rows = []
    for N in month_end[1:]:
        e = days.get_loc(N) - end_offset
        b = e - w
        if b < 0:
            continue
        r = (lp[long].iloc[e] - lp[long].iloc[b]) - (lp[short].iloc[e] - lp[short].iloc[b])
        rows.append({"N": N, "spread": r, "end": days[e]})
    return pd.DataFrame(rows).set_index("N")


def one_sided_mean(x):
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return float(t), float(stats.t.sf(t, len(x) - 1))


def dev(df):
    return df[(df.index >= DEV_START) & (df.end < DEV_END)]


def main():
    px = prices()
    d = dev(windows(px))
    t1, p1 = one_sided_mean(d.spread)
    ref = d[d.index.month.isin(REFUNDING)].spread
    oth = d[~d.index.month.isin(REFUNDING)].spread
    w = stats.ttest_ind(ref, oth, equal_var=False)
    t2, p2 = float(w.statistic), float(w.pvalue / 2 if w.statistic > 0 else 1 - w.pvalue / 2)
    ps = sorted([("H1", p1), ("H2", p2)], key=lambda x: x[1])
    holm, still = {}, True
    for i, (h, p) in enumerate(ps):
        still = still and p <= FWER / (len(ps) - i)
        holm[h] = still
    halves = {k: d[(d.index >= a) & (d.index < b)].spread.mean() for k, (a, b) in HALVES.items()}
    net = d.spread - COST
    sr = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731

    print(f"PRIMARY (TLT−SHY, last {W_PRIMARY} days), dev 2003–2024, n={len(d)} months")
    print(f"  H1 mean {d.spread.mean()*1e4:+.1f} bps  t={t1:+.2f}  p(1s)={p1:.4f}  Holm {'PASS' if holm['H1'] else 'fail'}")
    print(f"  H2 refunding {ref.mean()*1e4:+.1f} bps (n={len(ref)}) vs other {oth.mean()*1e4:+.1f} bps (n={len(oth)})"
          f"  Welch t={t2:+.2f}  p(1s)={p2:.4f}  Holm {'PASS' if holm['H2'] else 'fail'}")
    print(f"  halves: 2003-13 {halves['2003-13']*1e4:+.1f} bps   2014-24 {halves['2014-24']*1e4:+.1f} bps")
    print(f"  strategy (long spread every month-end): gross SR {sr(d.spread):+.2f}  net SR {sr(net):+.2f}  "
          f"hit {(d.spread > 0).mean():.0%}")
    ok = holm["H1"] and all(v > 0 for v in halves.values())
    print(f"\nBAR: {'STUDY 2 PASSES (dev): one OOS look permitted' if ok else 'STUDY 2 FAILS: OOS not touched'}")

    print("\nsensitivity (no bar):")
    for long in ["TLT", "IEF"]:
        for w_ in range(1, 6):
            x = dev(windows(px, long, "SHY", w_)).spread
            print(f"  {long} last {w_}d: mean {x.mean()*1e4:+6.1f} bps  t={one_sided_mean(x)[0]:+.2f}")
    pl = dev(windows(px, end_offset=10)).spread
    print(f"  PLACEBO (2 days ending N−10): mean {pl.mean()*1e4:+.1f} bps  t={one_sided_mean(pl)[0]:+.2f}")
    (ROOT / "results").mkdir(exist_ok=True)
    d.to_csv(ROOT / "results" / "treasury_extension_dev.csv")


if __name__ == "__main__":
    main()
