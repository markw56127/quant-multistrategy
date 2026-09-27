"""
M2 — solve the identified LQ model exactly and score it on history (dev only).

The Garleanu-Pedersen controller for the model identified in M1, against the
canonical value+quality quintile book, under identical real costs. Everything
here follows the pre-registration in README.md (fixed before this file ran).

EXACT SOLUTION.  Per rebalance, with diagonal Lambda, substitute y = Lambda^1/2 x.
The problem becomes one with unit trading cost and risk S = Lambda^-1/2 Sigma
Lambda^-1/2 = V diag(s) V'. Everything is then diagonal in V, and the Riccati
equation for the value function's quadratic term is a scalar quadratic per
eigen-direction:

    V(x_prev, alpha) = -1/2 x' Axx x + x' sum_k A_k alpha_k + const
    FOC:  x = J x_prev + J sum_k (1 + delta a_k A_k) alpha_k,   J = 1/(gamma s + 1 + delta Axx)
    Axx = 1 - J  ->  delta Axx^2 + (c - delta) Axx + (1 - c) = 0,   c = gamma s + 1
    A_k = J / (1 - delta a_k J)

J is the per-direction trade rate: a direction with high risk relative to its
trading cost (large s) is traded quickly, and a cheap-to-hold, expensive-to-trade
one slowly. (1 + delta a_k A_k) weights each alpha stream up by its persistence,
which is GP's "trade toward where the aim portfolio is going".

Parameters are estimated causally at each rebalance from data realised by then:
decay from the expanding-window signal ACF, beta from the expanding-window
Fama-MacBeth h=0 slope over return windows already closed.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identify import (load_dev, wide, xs_slope, pca_cov, ls_book,  # noqa: E402
                      SIGNALS, HOLD, RISK_WIN, TC_LINEAR, C_IMPACT, MAXLAG, SEED, ROOT)

AUM = 1e8                     # scored (pre-registered)
AUM_DIAG = 1e7                # reported, no bar
Q_REF_FRAC = 0.00614          # M1 (d): quadratic-to-sqrt matching trade size / AUM
DELTA = 0.99                  # monthly discount
BURN = 36                     # months of history before the first scored rebalance
K_PCA = 20                    # M1 (c) rule
BORROW = 0.01                 # annual, on shorts
ANN = 12


# ─────────────────────────────────────────────────────────── causal parameters

def precompute_slopes(panel):
    """acf[s][t, h-1]: xs slope of z_{t+h} on z_t.   beta[t]: joint FM h=0 slopes at t."""
    Z = {s: wide(panel, s) for s in SIGNALS}
    R = wide(panel, "fwd_ret")
    dates = R.index
    T = len(dates)
    acf = {s: np.full((T, MAXLAG), np.nan) for s in SIGNALS}
    for s in SIGNALS:
        Zs = Z[s].reindex(dates)
        for t in range(T):
            for h in range(1, min(MAXLAG, T - 1 - t) + 1):
                acf[s][t, h - 1] = xs_slope(Zs.iloc[t + h], Zs.iloc[t])
    beta = np.full((T, len(SIGNALS)), np.nan)
    for t in range(T):
        X = pd.concat({s: Z[s].reindex(dates).iloc[t] for s in SIGNALS}, axis=1).fillna(0.0)
        y = R.iloc[t].reindex(X.index)
        m = y.notna() & (X.abs().sum(axis=1) > 0)
        if m.sum() >= 30:
            beta[t] = sm.OLS(y[m], sm.add_constant(X[m])).fit().params[SIGNALS].values
    return dates, acf, beta


def params_at(T0, acf, beta):
    """Parameters usable at rebalance index T0: signal pairs (t, t+h) with t+h <= T0; return
    windows of dates t <= T0-1 (window of t ends at dates[t+1] <= dates[T0])."""
    def mean_acf(s):
        out = []
        for h in range(1, MAXLAG + 1):
            col = acf[s][: max(T0 - h + 1, 0), h - 1]
            col = col[~np.isnan(col)]
            if len(col) >= 3:
                out.append((h, col.mean(), col.std(ddof=1) / np.sqrt(len(col))))
        return np.array(out)

    v = mean_acf("value")
    (a_v,), _ = curve_fit(lambda h, a: a ** h, v[:, 0], v[:, 1], p0=[0.98], sigma=v[:, 2], bounds=(0, 1))
    q = mean_acf("quality")
    try:
        (w, a_f, a_s), _ = curve_fit(lambda h, w, af, as_: w * af ** h + (1 - w) * as_ ** h,
                                     q[:, 0], q[:, 1], p0=[0.3, 0.8, 0.98], sigma=q[:, 2],
                                     bounds=([0, 0, 0], [1, 1, 1]))
    except RuntimeError:
        (a_s,), _ = curve_fit(lambda h, a: a ** h, q[:, 0], q[:, 1], p0=[0.95], sigma=q[:, 2], bounds=(0, 1))
        w, a_f = 0.0, 0.0
    if a_f > a_s:                                   # label components by speed
        w, a_f, a_s = 1 - w, a_s, a_f
    b = np.nanmean(beta[:T0], axis=0)
    return {"a_v": a_v, "w": w, "a_f": a_f, "a_s": a_s, "beta_v": b[0], "beta_q": b[1]}


# ─────────────────────────────────────────────────────────── per-date inputs

def build_dates(panel, rets, adv, acf, beta, dates, aum):
    """Everything that does not depend on gamma: universe, Sigma eigensystem, Lambda, alphas."""
    days = rets.index
    vol = rets.rolling(63, min_periods=40).std()
    advd = adv.reindex(days, method="ffill")
    by_date = dict(tuple(panel.groupby("date")))
    out = []
    for T0 in range(BURN, len(dates)):
        d = dates[T0]
        g = by_date[d].set_index("ticker")
        di = days.get_loc(d)
        win = rets.iloc[di - RISK_WIN + 1: di + 1]
        names = [n for n in g.index if n in win.columns and win[n].notna().mean() > 0.9]
        g = g.loc[names]
        Sigma = pca_cov(win[names].fillna(0.0).values, K_PCA) * HOLD
        sig = vol.loc[d].reindex(names)
        ad = advd.loc[d].reindex(names)
        sig = sig.fillna(sig.median())
        ad = ad.where(ad > 0).fillna(ad[ad > 0].median())
        lam = 2 * C_IMPACT * sig.values * np.sqrt(aum) / np.sqrt(ad.values * Q_REF_FRAC)
        L = np.sqrt(lam)
        s, V = np.linalg.eigh(Sigma / np.outer(L, L))
        p = params_at(T0, acf, beta)
        v = g["value"].fillna(0.0).values
        q = g["quality"].fillna(0.0).values
        streams = [(p["a_v"], p["beta_v"] * v),
                   (p["a_f"], p["w"] * p["beta_q"] * q),
                   (p["a_s"], (1 - p["w"]) * p["beta_q"] * q)]
        w_ls = ls_book(g[SIGNALS].mean(axis=1)).reindex(names).fillna(0.0).values
        out.append({"date": d, "names": names, "Sigma": Sigma, "s": s, "V": V, "L": L, "lam": lam,
                    "mu": sum(al for _, al in streams), "v": v, "q": q,
                    "streams": [(a, V.T @ (al / L)) for a, al in streams],
                    "w_ls": w_ls, "fwd": g["fwd_ret"].values, "sig": sig.values, "adv": ad.values,
                    "params": p})
    return out


# ─────────────────────────────────────────────────────────── controller + scoring

def run_gp(D, gamma, kappa=1.0):
    """kappa scales Lambda (cost aversion); kappa=1 is the identified cost. Lambda -> kappa Lambda
    maps to s -> s/kappa, L -> sqrt(kappa) L with V unchanged, so no re-decomposition."""
    x_prev = pd.Series(dtype=float)
    books = []
    rk = np.sqrt(kappa)
    for e in D:
        xp = x_prev.reindex(e["names"]).fillna(0.0).values
        c = gamma * e["s"] / kappa + 1.0
        Axx = (-(c - DELTA) + np.sqrt((c - DELTA) ** 2 + 4 * DELTA * (c - 1))) / (2 * DELTA)
        J = 1.0 / (c + DELTA * Axx)
        yt = J * (e["V"].T @ (rk * e["L"] * xp))
        for a, at in e["streams"]:
            Ak = J / (1 - DELTA * a * J)
            yt += J * (1 + DELTA * a * Ak) * at / rk
        x = (e["V"] @ yt) / (rk * e["L"])
        x_prev = pd.Series(x, index=e["names"])
        books.append(x_prev)
    return books


def ex_ante_vol(D, books):
    return np.array([np.sqrt(b.values @ e["Sigma"] @ b.values) for e, b in zip(D, books)])


def calibrate_gamma(D, target):
    lo, hi = np.log(1e-3), np.log(1e7)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        v = ex_ante_vol(D, run_gp(D, np.exp(mid))).mean()
        lo, hi = (mid, hi) if v > target else (lo, mid)
    return float(np.exp(0.5 * (lo + hi)))


def score(D, books, aum):
    """Monthly gross / net returns with 10bps linear + sqrt impact + borrow; liquidates leavers."""
    last = {}                                     # name -> (sigma, adv) at last sighting
    prev = pd.Series(dtype=float)
    rows = []
    for e, x in zip(D, books):
        for n, sg, ad in zip(e["names"], e["sig"], e["adv"]):
            last[n] = (sg, ad)
        allnames = prev.index.union(x.index)
        dx = (x.reindex(allnames).fillna(0.0) - prev.reindex(allnames).fillna(0.0)).abs()
        sg = np.array([last[n][0] for n in allnames])
        ad = np.array([last[n][1] for n in allnames])
        lin = TC_LINEAR * dx.sum()
        imp = float((C_IMPACT * sg * dx.values ** 1.5 * np.sqrt(aum / ad)).sum())
        borrow = BORROW / ANN * x.clip(upper=0).abs().sum()
        gross = float(x.values @ e["fwd"])
        rows.append({"date": e["date"], "gross": gross, "lin": lin, "imp": imp, "borrow": borrow,
                     "net": gross - lin - imp - borrow, "turn": dx.sum(), "gmv": x.abs().sum(),
                     "nmv": x.sum()})
        prev = x
    return pd.DataFrame(rows).set_index("date")


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(ANN)


def boot_diff(a, b, n=2000, block=6):
    """Moving-block bootstrap of Sharpe(a) - Sharpe(b) on paired months."""
    rng = np.random.default_rng(SEED)
    T = len(a)
    starts = np.arange(T - block + 1)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, T // block + 1)])[:T]
        diffs.append(sharpe(a.iloc[idx]) - sharpe(b.iloc[idx]))
    return np.quantile(diffs, [0.05, 0.95])


def evaluate(panel, rets, adv, acf, beta, dates, aum):
    D = build_dates(panel, rets, adv, acf, beta, dates, aum)
    ls_books = [pd.Series(e["w_ls"], index=e["names"]) for e in D]
    target = ex_ante_vol(D, ls_books).mean()
    gamma = calibrate_gamma(D, target)
    gp_books = run_gp(D, gamma)
    return D, gamma, target, score(D, gp_books, aum), score(D, ls_books, aum), gp_books


def report(label, D, gamma, target, gp, ls, gp_books):
    print(f"\n{'='*78}\n{label}   {gp.index[0].date()} → {gp.index[-1].date()}  ({len(gp)} months)\n{'='*78}")
    print(f"gamma = {gamma:.4g}   ex-ante monthly vol target {target:.4f}  "
          f"(GP {ex_ante_vol(D, gp_books).mean():.4f})")
    print(f"{'':<12}{'gross SR':>9}{'net SR':>9}{'net ann':>9}{'real vol':>9}{'turn/mo':>9}"
          f"{'GMV':>7}{'NMV':>7}{'lin bps':>9}{'imp bps':>9}{'borrow':>8}")
    out = {}
    for name, df in [("GP", gp), ("quintile", ls)]:
        out[name] = {"gross_sr": sharpe(df.gross), "net_sr": sharpe(df.net)}
        print(f"{name:<12}{sharpe(df.gross):>+9.2f}{sharpe(df.net):>+9.2f}{df.net.mean()*ANN:>+9.2%}"
              f"{df.net.std()*np.sqrt(ANN):>9.2%}{df.turn.mean():>9.3f}{df.gmv.mean():>7.2f}"
              f"{df.nmv.mean():>+7.2f}{df.lin.mean()*ANN*1e4:>9.1f}{df.imp.mean()*ANN*1e4:>9.1f}"
              f"{df.borrow.mean()*ANN*1e4:>8.1f}")
    lo, hi = boot_diff(gp.net, ls.net)
    print(f"net SR diff GP−quintile {out['GP']['net_sr']-out['quintile']['net_sr']:+.2f}  "
          f"[90% block-boot {lo:+.2f}, {hi:+.2f}]   corr(net) {gp.net.corr(ls.net):+.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    panel, px, rets, adv = load_dev()
    dates, acf, beta = precompute_slopes(panel)

    p0, p1 = params_at(BURN, acf, beta), params_at(len(dates) - 1, acf, beta)
    print("causal parameters, first → last scored rebalance:")
    for k in p0:
        print(f"  {k:<7} {p0[k]:+.4f} → {p1[k]:+.4f}")

    res = {}
    for aum, tag in [(AUM, "SCORED  $100M"), (AUM_DIAG, "DIAGNOSTIC  $10M (no bar)")]:
        D, gamma, target, gp, ls, books = evaluate(panel, rets, adv, acf, beta, dates, aum)
        res[aum] = report(tag, D, gamma, target, gp, ls, books)
        pd.concat({"gp": gp, "quintile": ls}, axis=1).to_csv(ROOT / "results" / f"m2_dev_{int(aum/1e6)}M.csv")

    r = res[AUM]
    c1 = r["GP"]["net_sr"] >= r["quintile"]["net_sr"] + 0.10
    c2 = r["GP"]["gross_sr"] >= r["quintile"]["gross_sr"] - 0.05
    print(f"\nBAR @ $100M: (i) net SR ≥ quintile+0.10: {'PASS' if c1 else 'FAIL'}   "
          f"(ii) gross SR ≥ quintile−0.05: {'PASS' if c2 else 'FAIL'}   →  "
          f"{'DEV PASS — OOS look permitted' if c1 and c2 else 'DEV FAIL — OOS not touched'}")


if __name__ == "__main__":
    main()
