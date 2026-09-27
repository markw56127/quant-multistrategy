"""
Study 1: month-end balanced-fund rebalancing. Pre-registered in README.md; DEV ONLY.

S_m     = SPY - IEF return, prior month-end close -> close of N-k    (k = 3 primary)
H1  R_m = SPY - IEF return, close N-k -> close N                     predicted slope < 0
H2  U_m = SPY - IEF return over the first k trading days of m+1       predicted slope > 0
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from scipy import stats

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "etf_px.parquet"
DEV_START, DEV_END = pd.Timestamp("2003-01-01"), pd.Timestamp("2025-01-01")
HALVES = {"2003-13": ("2003-01-01", "2014-01-01"), "2014-24": ("2014-01-01", "2025-01-01")}
K_PRIMARY = 3
COST_PER_LEG = 0.0002                                   # 2 bps per leg per transaction
FWER = 0.05


def prices():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    px = yf.download(["SPY", "IEF", "TLT"], start="2002-07-01", auto_adjust=True, progress=False)["Close"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(CACHE)
    return px


def monthly_windows(px, bond, k):
    """One row per month: signal, H1 window spread, H2 unwind spread. Month m is labelled by
    its last trading day N; H2 uses the first k trading days of the next month."""
    lp = np.log(px[["SPY", bond]].dropna())
    days = lp.index
    month_end = days[days.to_series().dt.to_period("M").ne(days.to_series().shift(-1).dt.to_period("M"))]
    rows = []
    for j in range(1, len(month_end) - 1):
        prev, N, nxt_end = month_end[j - 1], month_end[j], month_end[j + 1]
        iN, ip = days.get_loc(N), days.get_loc(prev)
        if iN - k <= ip:
            continue
        spread = lambda a, b: (lp.SPY.iloc[b] - lp.SPY.iloc[a]) - (lp[bond].iloc[b] - lp[bond].iloc[a])  # noqa: E731
        u_end = iN + k
        if days[u_end] > nxt_end:
            continue
        rows.append({"N": N, "S": spread(ip, iN - k), "R": spread(iN - k, iN), "U": spread(iN, u_end),
                     "U_end": days[u_end]})
    return pd.DataFrame(rows).set_index("N")


def nw_slope(y, x, lags=3):
    r = sm.OLS(y.values, sm.add_constant(x.values)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[1]), float(r.tvalues[1]), len(y)


def strategy(df):
    """Short the leg that outperformed: P&L = -sign(S) * R, net of 8 bps per month."""
    g = -np.sign(df.S) * df.R
    n = g - 4 * COST_PER_LEG
    sr = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
    return {"gross_sr": sr(g), "net_sr": sr(n), "hit": float((g > 0).mean()),
            "mean_bps": float(g.mean() * 1e4), "net_mean_bps": float(n.mean() * 1e4)}


def analyse(px, bond, k, verbose=True):
    df = monthly_windows(px, bond, k)
    dev = df[(df.index >= DEV_START) & (df.U_end < DEV_END)]      # both windows closed before OOS
    b1, t1, n = nw_slope(dev.R, dev.S)
    b2, t2, _ = nw_slope(dev.U, dev.S)
    p1, p2 = stats.t.sf(-t1, n - 2), stats.t.sf(t2, n - 2)          # H1 predicts <0, H2 predicts >0
    out = {"bond": bond, "k": k, "n": n, "b1": b1, "t1": t1, "p1": p1, "b2": b2, "t2": t2, "p2": p2,
           **strategy(dev)}
    for name, (a, b) in HALVES.items():
        h = dev[(dev.index >= a) & (dev.index < b)]
        s = strategy(h)
        out[f"net_sr_{name}"] = s["net_sr"]
        out[f"t1_{name}"] = nw_slope(h.R, h.S)[1]
    terc = pd.qcut(dev.S.abs(), 3, labels=False)
    out["t1_top_tercile"] = nw_slope(dev.R[terc == 2], dev.S[terc == 2])[1]
    out["t1_bottom_tercile"] = nw_slope(dev.R[terc == 0], dev.S[terc == 0])[1]
    return out


def main():
    px = prices()
    prim = analyse(px, "IEF", K_PRIMARY)
    # Holm over {H1, H2}
    ps = sorted([("H1", prim["p1"]), ("H2", prim["p2"])], key=lambda x: x[1])
    holm, still = {}, True
    for i, (h, p) in enumerate(ps):
        still = still and p <= FWER / (len(ps) - i)
        holm[h] = still
    both_halves = prim["net_sr_2003-13"] > 0 and prim["net_sr_2014-24"] > 0
    print(f"PRIMARY (SPY−IEF, k={K_PRIMARY}), dev {DEV_START.year}–2024, n={prim['n']} months")
    print(f"  H1 window slope  {prim['b1']:+.4f}  t={prim['t1']:+.2f}  p(1s)={prim['p1']:.4f}  Holm {'PASS' if holm['H1'] else 'fail'}")
    print(f"  H2 unwind slope  {prim['b2']:+.4f}  t={prim['t2']:+.2f}  p(1s)={prim['p2']:.4f}  Holm {'PASS' if holm['H2'] else 'fail'}")
    print(f"  strategy: gross SR {prim['gross_sr']:+.2f}  net SR {prim['net_sr']:+.2f}  hit {prim['hit']:.0%}  "
          f"mean {prim['mean_bps']:+.1f} bps/mo gross, {prim['net_mean_bps']:+.1f} net")
    print(f"  halves: net SR 2003-13 {prim['net_sr_2003-13']:+.2f} (t1 {prim['t1_2003-13']:+.2f})   "
          f"2014-24 {prim['net_sr_2014-24']:+.2f} (t1 {prim['t1_2014-24']:+.2f})")
    print(f"  |S| terciles: t1 top {prim['t1_top_tercile']:+.2f}   bottom {prim['t1_bottom_tercile']:+.2f}")
    ok = holm["H1"] and both_halves
    print(f"\nBAR: H1 Holm {'PASS' if holm['H1'] else 'FAIL'}; net SR > 0 in both halves "
          f"{'PASS' if both_halves else 'FAIL'}  →  {'STUDY 1 PASSES (dev): one OOS look permitted' if ok else 'STUDY 1 FAILS: OOS not touched'}")

    print("\nsensitivity (no bar): k and bond proxy")
    print(f"{'bond':<5}{'k':>3}{'t1':>7}{'t2':>7}{'net SR':>8}{'SR 03-13':>9}{'SR 14-24':>9}")
    rows = []
    for bond in ["IEF", "TLT"]:
        for k in range(1, 6):
            r = analyse(px, bond, k)
            rows.append(r)
            print(f"{bond:<5}{k:>3}{r['t1']:>+7.2f}{r['t2']:>+7.2f}{r['net_sr']:>+8.2f}"
                  f"{r['net_sr_2003-13']:>+9.2f}{r['net_sr_2014-24']:>+9.2f}")
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame([prim] + rows).to_csv(ROOT / "results" / "month_end_dev.csv", index=False)


if __name__ == "__main__":
    main()
