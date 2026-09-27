"""
Track A: collecting risk premia efficiently. Pre-registered in README.md.

A1  covariance estimators (course topics 4-6), scored by the realized volatility of
    each estimator's long-only GMV portfolio
A2  weighting schemes on a fixed covariance (daily_1y)
A3  sizing: Ito's growth formula g(L) = r + L mu - L^2 sigma^2 / 2, and causal
    Kelly / half-Kelly / vol targeting

DEV ONLY by construction: prices are truncated at DEV_END before anything is computed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "px.parquet"
ASSETS = ["SPY", "EFA", "EEM", "IEF", "TLT", "LQD", "GLD", "DBC", "VNQ"]
DEV_END = pd.Timestamp("2025-01-01")
WIN_M, WIN_D = 36, 252
TC, BORROW_SPREAD = 0.0005, 0.005
SEED, NBOOT, BLOCK = 0, 2000, 6


# ─────────────────────────────────────────────────────────── data

def load(end):
    if CACHE.exists():
        px = pd.read_parquet(CACHE)
    else:
        px = yf.download(ASSETS + ["^IRX"], start="2005-12-01", auto_adjust=True, progress=False)["Close"]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        px.to_parquet(CACHE)
    px = px[px.index < end]
    irx = px.pop("^IRX").ffill() / 100                               # annualized 13-week T-bill yield
    px = px[ASSETS].dropna()
    daily = px.pct_change().dropna()
    mpx = px.resample("ME").last()
    monthly = mpx.pct_change().dropna()
    rf_m = (irx.resample("ME").last().shift(1) / 12).reindex(monthly.index).fillna(0)   # known at month start
    return daily, monthly, rf_m


# ─────────────────────────────────────────────────────────── covariance estimators

def cov_sample(R, *_):
    return np.cov(R, rowvar=False)


def cov_single_index(R, *_):
    m = R[:, ASSETS.index("SPY")]
    vm = m.var(ddof=1)
    beta = np.array([np.cov(R[:, i], m)[0, 1] / vm for i in range(R.shape[1])])
    resid = R - np.outer(m - m.mean(), beta) - R.mean(0)
    s2e = resid.var(0, ddof=1)
    s2e[ASSETS.index("SPY")] = 0.0
    return np.outer(beta, beta) * vm + np.diag(s2e)


def cov_const_corr(R, *_):
    C = np.corrcoef(R, rowvar=False)
    n = C.shape[0]
    rho = (C.sum() - n) / (n * (n - 1))
    sd = R.std(0, ddof=1)
    S = rho * np.outer(sd, sd)
    np.fill_diagonal(S, sd ** 2)
    return S


def cov_ledoit_wolf(R, *_):
    return LedoitWolf().fit(R).covariance_


def cov_daily_1y(_, D):
    return np.cov(D, rowvar=False) * 21


ESTIMATORS = {"sample": cov_sample, "single_index": cov_single_index, "const_corr": cov_const_corr,
              "ledoit_wolf": cov_ledoit_wolf, "daily_1y": cov_daily_1y}


# ─────────────────────────────────────────────────────────── weighting schemes (long-only)

def _solve(obj, n, x0=None):
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    r = minimize(obj, np.full(n, 1 / n) if x0 is None else x0, bounds=[(0, 1)] * n,
                 constraints=cons, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    return np.clip(r.x, 0, None) / np.clip(r.x, 0, None).sum()


def w_gmv(S, mu=None):
    return _solve(lambda w: w @ S @ w * 1e4, len(S))


def w_erc(S, mu=None):
    """Equal risk contribution via Spinu's convex form: min ½w'Sw − (1/n)Σlog w, then normalize."""
    n = len(S)
    r = minimize(lambda y: 0.5 * y @ S @ y - np.log(y).sum() / n, np.full(n, 1 / n),
                 jac=lambda y: S @ y - 1 / (n * y), bounds=[(1e-10, None)] * n, method="L-BFGS-B")
    return r.x / r.x.sum()


def w_max_sharpe(S, mu):
    return _solve(lambda w: -(w @ mu) / np.sqrt(w @ S @ w), len(S))


def w_inv_vol(S, mu=None):
    v = 1 / np.sqrt(np.diag(S))
    return v / v.sum()


def w_equal(S, mu=None):
    return np.full(len(S), 1 / len(S))


def w_sixty_forty(S, mu=None):
    w = np.zeros(len(ASSETS))
    w[ASSETS.index("SPY")], w[ASSETS.index("IEF")] = 0.6, 0.4
    return w


SCHEMES = {"sixty_forty": w_sixty_forty, "equal": w_equal, "inv_vol": w_inv_vol,
           "risk_parity": w_erc, "gmv": w_gmv, "max_sharpe": w_max_sharpe}


# ─────────────────────────────────────────────────────────── backtest

def formation_inputs(daily, monthly, rf_m):
    """For each formation month-end t (with 36 months of history), the inputs known at t and
    the next month's realized returns."""
    out = []
    for i in range(WIN_M, len(monthly)):
        t = monthly.index[i - 1]                                      # formation date (month-end)
        R = monthly.iloc[i - WIN_M:i].values
        D = daily[daily.index <= t].iloc[-WIN_D:].values
        mu = (monthly.iloc[i - WIN_M:i].sub(rf_m.iloc[i - WIN_M:i], axis=0)).mean().values
        out.append({"t": t, "hold": monthly.index[i], "R": R, "D": D, "mu": mu,
                    "r_next": monthly.iloc[i].values, "rf_next": rf_m.iloc[i]})
    return out


def run_book(inputs, weight_fn, cov_fn):
    rows, w_prev = [], None
    for x in inputs:
        S = cov_fn(x["R"], x["D"])
        w = weight_fn(S, x["mu"])
        drift = w_prev if w_prev is None else w_prev * (1 + x_prev_r) / (1 + w_prev @ x_prev_r)
        turn = w.sum() if drift is None else np.abs(w - drift).sum()
        r = w @ x["r_next"] - TC * turn
        rows.append({"hold": x["hold"], "ret": r, "excess": r - x["rf_next"], "rf": x["rf_next"],
                     "sigma_hat": np.sqrt(w @ S @ w), "turn": turn, "w": w})
        w_prev, x_prev_r = w, x["r_next"]
    return pd.DataFrame(rows).set_index("hold")


def stats_(df):
    ex = df.excess
    wealth = (1 + df.ret).cumprod()
    return {"ann_excess": ex.mean() * 12, "vol": ex.std() * np.sqrt(12), "sharpe": ex.mean() / ex.std() * np.sqrt(12),
            "max_dd": float((wealth / wealth.cummax() - 1).min()), "turn_yr": df.turn.mean() * 12,
            "log_growth": float(np.log1p(df.ret).mean() * 12)}


def block_boot(fn, n, seed=SEED):
    rng = np.random.default_rng(seed)
    starts = np.arange(n - BLOCK + 1)
    vals = []
    for _ in range(NBOOT):
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, n // BLOCK + 1)])[:n]
        vals.append(fn(idx))
    return np.quantile(vals, [0.05, 0.95])


# ─────────────────────────────────────────────────────────── A1–A3

def a1(inputs):
    print(f"\n{'='*78}\nA1  covariance estimators → long-only GMV, realized OOS risk\n{'='*78}")
    books = {k: run_book(inputs, w_gmv, f) for k, f in ESTIMATORS.items()}
    base = books["sample"].ret
    print(f"{'estimator':<14}{'real vol':>9}{'bias':>7}{'var/sample':>11}{'90% CI':>18}{'< sample?':>11}")
    res = {}
    for k, b in books.items():
        vr = b.ret.var() / base.var()
        lo, hi = block_boot(lambda i: b.ret.iloc[i].var() / base.iloc[i].var(), len(b))
        bias = (b.ret / b.sigma_hat).std()
        res[k] = {"vol": b.ret.std() * np.sqrt(12), "bias": bias, "var_ratio": vr, "ci": (lo, hi), "better": hi < 1}
        print(f"{k:<14}{b.ret.std()*np.sqrt(12):>9.2%}{bias:>7.2f}{vr:>11.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{('yes' if hi < 1 else 'no') if k != 'sample' else '—':>11}")
    n_ok = sum(res[k]["better"] for k in ["single_index", "const_corr", "ledoit_wolf"])
    print(f"A1: {n_ok}/3 structured estimators beat sample (CI < 1) → {'PASS' if n_ok >= 2 else 'FAIL'} (bar: ≥ 2)")
    return res


def a2(inputs):
    print(f"\n{'='*78}\nA2  weighting schemes (covariance = daily_1y)\n{'='*78}")
    books = {k: run_book(inputs, f, cov_daily_1y) for k, f in SCHEMES.items()}
    print(f"{'scheme':<13}{'ex ret':>8}{'vol':>8}{'Sharpe':>8}{'maxDD':>8}{'turn/yr':>9}{'log g':>8}")
    st = {k: stats_(b) for k, b in books.items()}
    for k, s in st.items():
        print(f"{k:<13}{s['ann_excess']:>+8.2%}{s['vol']:>8.2%}{s['sharpe']:>+8.2f}{s['max_dd']:>8.1%}"
              f"{s['turn_yr']:>9.2f}{s['log_growth']:>+8.2%}")
    sr = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
    rp, sf = books["risk_parity"].excess, books["sixty_forty"].excess
    lo, hi = block_boot(lambda i: sr(rp.iloc[i]) - sr(sf.iloc[i]), len(rp))
    print(f"(a) risk_parity − sixty_forty Sharpe: {st['risk_parity']['sharpe'] - st['sixty_forty']['sharpe']:+.2f}"
          f"  90% CI [{lo:+.2f}, {hi:+.2f}]  → claim {'holds' if st['risk_parity']['sharpe'] > st['sixty_forty']['sharpe'] else 'fails'}")
    worst = min(st, key=lambda k: st[k]["sharpe"])
    print(f"(b) lowest Sharpe: {worst}  → claim ('max_sharpe' is worst) {'holds' if worst == 'max_sharpe' else 'fails'}")
    return books, st


def a3(inputs, books):
    print(f"\n{'='*78}\nA3  sizing: Ito growth curve and causal Kelly (risk_parity book)\n{'='*78}")
    b = books["risk_parity"]
    x, rf = b.excess, b.rf
    mu, s2, rbar = x.mean() * 12, x.var() * 12, rf.mean() * 12
    L_star = mu / s2
    print(f"risk_parity dev: μ(excess) {mu:.2%}/yr  σ {np.sqrt(s2):.2%}  → Kelly L* = μ/σ² = {L_star:.2f}")
    print(f"{'L':>5}{'realized g':>12}{'Ito g(L)':>11}{'maxDD':>8}")
    rows = []
    for L in np.arange(0.5, 6.01, 0.5):
        r = rf + L * x - max(L - 1, 0) * BORROW_SPREAD / 12
        g_real = np.log1p(r).mean() * 12
        g_ito = rbar + L * mu - 0.5 * L ** 2 * s2 - max(L - 1, 0) * BORROW_SPREAD
        w = (1 + r).cumprod()
        rows.append((L, g_real, g_ito))
        print(f"{L:>5.1f}{g_real:>+12.2%}{g_ito:>+11.2%}{(w / w.cummax() - 1).min():>8.1%}")
    rr = pd.DataFrame(rows, columns=["L", "real", "ito"])
    L_real = rr.loc[rr.real.idxmax(), "L"]
    print(f"A3a: realized optimum L = {L_real:.1f} vs Ito L* = {L_star:.2f}; max |real − Ito| = "
          f"{(rr.real - rr.ito).abs().max():.2%} → claim {'holds' if abs(L_real - L_star) <= 1 else 'fails'}")

    # A3b: causal sizing from trailing estimates of the CURRENT risk_parity weights
    print(f"\n{'sizing':<12}{'mean L':>8}{'log g':>9}{'vol':>8}{'Sharpe':>8}{'maxDD':>8}{'% capped':>9}")
    out = {}
    for name in ["fixed_1", "full_kelly", "half_kelly", "vol_target"]:
        rets, Ls, cap = [], [], 0
        for xin, (h, row) in zip(inputs, b.iterrows()):
            w = row.w
            hist = (xin["R"] @ w)                                  # trailing 36m, current weights
            m, v = hist.mean() - xin["rf_next"], hist.var(ddof=1)
            if name == "fixed_1":
                L = 1.0
            elif name == "vol_target":
                L = (0.10 / np.sqrt(12)) / row.sigma_hat
            else:
                L = max(m / v, 0) * (0.5 if name == "half_kelly" else 1.0)
            cap += L >= 8
            L = min(L, 8)
            Ls.append(L)
            rets.append(row.rf + L * row.excess - max(L - 1, 0) * BORROW_SPREAD / 12)
        r = pd.Series(rets, index=b.index)
        ex = r - b.rf
        w = (1 + r).cumprod()
        out[name] = {"g": np.log1p(r).mean() * 12, "sharpe": ex.mean() / ex.std() * np.sqrt(12),
                     "dd": (w / w.cummax() - 1).min()}
        print(f"{name:<12}{np.mean(Ls):>8.2f}{out[name]['g']:>+9.2%}{ex.std()*np.sqrt(12):>8.2%}"
              f"{out[name]['sharpe']:>+8.2f}{out[name]['dd']:>8.1%}{cap/len(Ls):>9.0%}")
    hk, fk, vt = out["half_kelly"], out["full_kelly"], out["vol_target"]
    print(f"A3b: half > full Kelly on growth {'✓' if hk['g'] > fk['g'] else '✗'} and drawdown "
          f"{'✓' if hk['dd'] > fk['dd'] else '✗'}; vol_target Sharpe best of the three "
          f"{'✓' if vt['sharpe'] > max(hk['sharpe'], fk['sharpe']) else '✗'}")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    daily, monthly, rf_m = load(DEV_END)
    inputs = formation_inputs(daily, monthly, rf_m)
    print(f"DEV: {len(inputs)} holding months [{inputs[0]['hold'].date()} → {inputs[-1]['hold'].date()}], "
          f"{len(ASSETS)} ETFs, trailing {WIN_M}m (T/N = {WIN_M/len(ASSETS):.0f})")
    a1(inputs)
    books, _ = a2(inputs)
    a3(inputs, books)
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame({k: v.ret for k, v in books.items()}).to_csv(ROOT / "results" / "a2_dev_returns.csv")


if __name__ == "__main__":
    main()
