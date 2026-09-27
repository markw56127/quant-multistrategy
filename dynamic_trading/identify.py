"""
M1 — identify the linear-quadratic trading model, on the DEVELOPMENT window only.

The surrogate is the Garleanu-Pedersen (2013) system, the trading analogue of
MEQ's identified linear response model. The controlled state is our own holdings,
not prices; the market enters only through signals with estimable dynamics:

    signals   f_{t+1} = A f_t + eps          (per stock; A = I - Phi, OU in discrete time)
    returns   r_{t+1} = B f_t + u,  Cov(u) = Sigma
    holdings  x_t     = x_{t-1} + dx_t
    cost      TC_t    = 1/2 dx_t' Lambda dx_t

Four things are identified here, each with a validity check, so that M2 (the
closed-form controller) is built on measured parameters with measured limits:

  (a) SIGNAL DYNAMICS  cross-sectional AR slope of each signal at lags 1..12.
      Is the ACF geometric (one OU component), or c*a^h with c<1 (a fast
      component on top of a slow one)? VAR(1) cross-terms. Half-lives.
  (b) RETURN DECAY     Fama-MacBeth slope of the month-(t+h) return on f_t,
      h = 0..11. GP's model *implies* beta_h = beta_0 * a^h with the SAME a as
      (a). Agreement is the model's internal consistency test.
  (c) RISK MODEL       PCA factor model from trailing daily returns, K chosen by
      a rule fixed here (min mean QLIKE on the two portfolios the strategy would
      actually hold). Bias statistics on those plus random portfolios - the
      mean-variance "aim" portfolio is where estimated risk is most flattering.
  (d) COSTS            GP needs quadratic costs. The repo's costs are 10 bps
      linear + square-root impact (factor_research/capacity.py). Measured: how
      much of total cost is the linear part LQ cannot represent, how wrong the
      quadratic is across the actual trade-size distribution, and whether GP's
      canonical Lambda proportional to Sigma matches per-name impact.

Leakage rule: a row is DEV only if its forward-return window ENDS before
2025-01-01 (the 2024-12-06 rebalance's window ends 2025-01-08 and is excluded).
Nothing in this file reads a price or return dated 2025 or later.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PANEL = REPO / "signal_combiner" / "cache" / "panel.parquet"
PRICES = REPO / "factor_model" / "cache" / "prices_sf_2015-01-01_2026-06-30.parquet"
ADV = REPO / "factor_research" / "cache" / "adv_dollar.parquet"

SIGNALS = ["value", "quality"]
DEV_END = pd.Timestamp("2025-01-01")
HOLD = 21                                  # trading days per rebalance (panel convention)
MAXLAG = 12
SUBPERIODS = {"2016-19": ("2016-01-01", "2020-01-01"), "2020-24": ("2020-01-01", "2025-01-01")}
K_GRID = [1, 3, 5, 10, 20]
RISK_WIN = 252
TC_LINEAR = 0.001                          # 10 bps per unit turnover (factor_model config)
C_IMPACT = 1.0                             # sqrt-impact coefficient (capacity.py)
AUM_GRID = [1e7, 1e8, 1e9]
SEED = 0
BOOT = 1000


# ─────────────────────────────────────────────────────────── data (dev only)

def load_dev():
    """Panel + daily prices restricted so no value dated >= DEV_END is ever read."""
    px = pd.read_parquet(PRICES)
    px = px[px.index < DEV_END]
    panel = pd.read_parquet(PANEL)
    days = px.index
    # window end = HOLD trading days after the rebalance; must exist inside the dev prices
    pos = days.get_indexer(panel["date"])
    ok = (pos >= 0) & (pos + HOLD < len(days))
    panel = panel[ok].copy()
    panel["win_end"] = days[pos[ok] + HOLD]
    assert panel["win_end"].max() < DEV_END
    rets = px.pct_change(fill_method=None)
    adv = pd.read_parquet(ADV)
    adv = adv[adv.index < DEV_END]
    return panel, px, rets, adv


def wide(panel, col):
    return panel.pivot(index="date", columns="ticker", values=col).sort_index()


def nw_mean(y, lags=3):
    y = np.asarray(pd.Series(y).dropna())
    r = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[0]), float(r.bse[0]), len(y)


def xs_slope(y, x):
    """Cross-sectional OLS slope of y on x (with intercept); NaN if too few names."""
    m = y.notna() & x.notna()
    if m.sum() < 30:
        return np.nan
    xv, yv = x[m] - x[m].mean(), y[m] - y[m].mean()
    return float((xv * yv).sum() / (xv ** 2).sum())


# ─────────────────────────────────────────────────────────── (a) signal dynamics

def signal_dynamics(panel):
    out = {}
    for s in SIGNALS:
        Z = wide(panel, s)
        acf = []
        for h in range(1, MAXLAG + 1):
            sl = [xs_slope(Z.iloc[t + h], Z.iloc[t]) for t in range(len(Z) - h)]
            m, se, n = nw_mean(sl)
            acf.append({"h": h, "slope": m, "se": se, "n": n})
        acf = pd.DataFrame(acf)
        a1 = acf.loc[0, "slope"]
        # two-component fit rho_h = c * a^h  (c<1 => a fast-decaying part on top of the slow one)
        (c, a), cov = curve_fit(lambda h, c, a: c * a ** h, acf["h"], acf["slope"],
                                p0=[1.0, a1], sigma=acf["se"])
        # two-OU mixture rho_h = w*a_f^h + (1-w)*a_s^h: a fast and a slow component
        try:
            (w2, af, as_), _ = curve_fit(lambda h, w, af, as_: w * af ** h + (1 - w) * as_ ** h,
                                         acf["h"], acf["slope"], p0=[0.3, 0.6, 0.97],
                                         sigma=acf["se"], bounds=([0, 0, 0], [1, 1, 1]))
        except RuntimeError:
            w2 = af = as_ = np.nan
        out[s] = {
            "mix_w_fast": w2, "mix_a_fast": af, "mix_a_slow": as_,
            "acf": acf,
            "ar1": a1,
            "ar1_halflife_mo": np.log(0.5) / np.log(a1),
            "geo_c": c, "geo_a": a, "geo_a_se": float(np.sqrt(cov[1, 1])),
            "geo_halflife_mo": np.log(0.5) / np.log(a),
            # max deviation of the pure-AR(1) prediction a1^h from the measured ACF
            "ar1_misfit_max": float(np.max(np.abs(acf["slope"] - a1 ** acf["h"]))),
        }

    # VAR(1) cross terms: [v,q]_{t+1} on [v,q]_t, per-date cross-sectional OLS, FM-averaged
    Zs = {s: wide(panel, s) for s in SIGNALS}
    dates = Zs[SIGNALS[0]].index
    coefs = []
    for t in range(len(dates) - 1):
        X = pd.concat({s: Zs[s].iloc[t] for s in SIGNALS}, axis=1)
        for tgt in SIGNALS:
            y = Zs[tgt].iloc[t + 1]
            d = pd.concat([X, y.rename("y")], axis=1).dropna()
            if len(d) < 30:
                continue
            b = sm.OLS(d["y"], sm.add_constant(d[SIGNALS])).fit().params
            coefs.append({"t": t, "tgt": tgt, **{s: b[s] for s in SIGNALS}})
    coefs = pd.DataFrame(coefs)
    var = {tgt: {s: nw_mean(coefs.loc[coefs.tgt == tgt, s])[:2] for s in SIGNALS} for tgt in SIGNALS}
    return out, var


# ─────────────────────────────────────────────────────────── (b) return decay

def _fit_exp(h, beta, se):
    (b0, b), _ = curve_fit(lambda h, b0, b: b0 * b ** h, h, beta, p0=[beta[0], 0.9],
                           sigma=se, bounds=([-1, 0.0], [1, 1.5]))
    return b0, b


def return_decay(panel, a_signal=None, n_boot=BOOT, block=12):
    """FM slopes S[t, h] of month-(t+h) return on f_t. Uncertainty on the fitted decay comes
    from a moving-block bootstrap over signal dates t: the 12 horizons reuse the same months,
    so curve_fit's per-point standard errors would treat correlated points as independent."""
    R = wide(panel, "fwd_ret")
    Zs = {s: wide(panel, s).reindex(R.index) for s in SIGNALS}
    T = len(R)
    S = {s: np.full((T, MAXLAG), np.nan) for s in SIGNALS}
    for t in range(T):
        X = pd.concat({s: Zs[s].iloc[t] for s in SIGNALS}, axis=1).fillna(0.0)
        X = X[X.abs().sum(axis=1) > 0]
        for h in range(min(MAXLAG, T - t)):
            y = R.iloc[t + h].reindex(X.index)
            m = y.notna()
            if m.sum() < 30:
                continue
            p = sm.OLS(y[m], sm.add_constant(X[m])).fit().params
            for s in SIGNALS:
                S[s][t, h] = p[s]
    rows = []
    for s in SIGNALS:
        for h in range(MAXLAG):
            m, se, n = nw_mean(S[s][:, h])
            rows.append({"signal": s, "h": h, "beta": m, "se": se, "t": m / se, "n": n})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    starts = np.arange(T - block + 1)
    boots = [np.concatenate([np.arange(b0, b0 + block) for b0 in rng.choice(starts, T // block + 1)])[:T]
             for _ in range(n_boot)]
    fits = {}
    for s in SIGNALS:
        d = df[df.signal == s]
        hh, beta, se = d["h"].values, d["beta"].values, d["se"].values
        try:
            b0, b = _fit_exp(hh, beta, se)
        except RuntimeError:
            b0 = b = np.nan
        bb, sums = [], []
        for idx in boots:
            mb = np.nanmean(S[s][idx], axis=0)
            sums.append(mb.sum())
            try:
                bb.append(_fit_exp(hh, mb, se)[1])
            except RuntimeError:
                pass
        bb = np.array(bb)
        fits[s] = {"beta0": b0, "b": b, "b_ci90": [float(np.quantile(bb, 0.05)), float(np.quantile(bb, 0.95))],
                   "sum12": float(beta.sum()), "sum12_se": float(np.std(sums))}
        # reconcile with (a): beta_h = bf*bfr^h + bs*a^h, slow rate a FIXED at signal persistence
        if a_signal is not None:
            a = a_signal[s]
            fits[s]["b_boot_share_ge_a"] = float((bb >= a).mean())
            try:
                (bf, bfr, bs), _ = curve_fit(lambda h, bf, bfr, bs: bf * bfr ** h + bs * a ** h,
                                             hh, beta, p0=[0.0005, 0.3, 0.0015],
                                             sigma=se, bounds=([-1, 0, -1], [1, 1, 1]))
            except RuntimeError:
                bf = bfr = bs = np.nan
            fits[s].update({"two_fast": bf, "two_fast_rate": bfr, "two_slow": bs, "two_slow_rate": a})
    return df, fits


# ─────────────────────────────────────────────────────────── (c) risk model

def pca_cov(X, K):
    """Daily covariance: K-factor PCA + diagonal residual. X: T x N, NaN-filled with 0."""
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    L = Vt[:K].T * (S[:K] / np.sqrt(len(X)))
    resid = Xc - U[:, :K] * S[:K] @ Vt[:K]
    return L @ L.T + np.diag(resid.var(0) + 1e-10)


def ls_book(score, q=0.20):
    s = score.dropna().sort_values()
    k = max(int(len(s) * q), 1)
    w = pd.Series(0.0, index=s.index)
    w.iloc[-k:] = 0.5 / k
    w.iloc[:k] = -0.5 / k
    return w


def risk_model(panel, rets):
    rng = np.random.default_rng(SEED)
    days = rets.index
    recs = []
    for d, g in panel.groupby("date"):
        di = days.get_loc(d)
        if di < RISK_WIN:
            continue
        win = rets.iloc[di - RISK_WIN + 1: di + 1]           # returns through close of d
        g = g.set_index("ticker")
        names = [n for n in g.index if n in win.columns and win[n].notna().mean() > 0.9]
        X = win[names].fillna(0.0).values
        r = g.loc[names, "fwd_ret"].values
        score = g.loc[names, SIGNALS].mean(axis=1)
        mu = (score - score.mean()).fillna(0.0).values
        w_ls = ls_book(score).reindex(names).fillna(0.0).values
        w_rand = rng.standard_normal((10, len(names)))
        w_rand -= w_rand.mean(1, keepdims=True)
        for K in K_GRID:
            C = pca_cov(X, K) * HOLD
            w_mv = np.linalg.solve(C, mu)
            w_mv /= np.abs(w_mv).sum()
            for name, w in [("ls_book", w_ls), ("mv_aim", w_mv)] + \
                           [(f"rand{j}", w_rand[j]) for j in range(len(w_rand))]:
                var = float(w @ C @ w)
                recs.append({"date": d, "K": K, "port": name,
                             "z": float(w @ r) / np.sqrt(var), "var": var, "r": float(w @ r)})
    df = pd.DataFrame(recs)
    df["qlike"] = np.log(df["var"]) + df["r"] ** 2 / df["var"]
    df["kind"] = np.where(df.port.str.startswith("rand"), "random", df.port)
    summ = df.groupby(["K", "kind"]).agg(bias=("z", "std"), qlike=("qlike", "mean"),
                                         n=("z", "size")).reset_index()
    # rule fixed in advance: K minimising mean QLIKE over the two held portfolios
    held = summ[summ.kind.isin(["ls_book", "mv_aim"])].groupby("K")["qlike"].mean()
    return summ, int(held.idxmin())


# ─────────────────────────────────────────────────────────── (d) costs

def c_sqrt_total(s_, q, a_):
    return float((C_IMPACT * s_ * q ** 1.5 / np.sqrt(a_)).sum()) / C_IMPACT

def costs(panel, rets, adv):
    days = rets.index
    vol = rets.rolling(63, min_periods=40).std()
    advd = adv.reindex(days, method="ffill")
    W = {d: ls_book(g.set_index("ticker")[SIGNALS].mean(axis=1)) for d, g in panel.groupby("date")}
    W = pd.DataFrame(W).T.sort_index().fillna(0.0)
    dW = W.diff().abs().iloc[1:]
    sig = vol.reindex(dW.index)[dW.columns]
    adv_ = advd.reindex(dW.index)[dW.columns]
    trades = dW.stack()
    trades = trades[trades > 0]
    s_ = sig.stack().reindex(trades.index)
    a_ = adv_.stack().reindex(trades.index)
    ok = s_.notna() & a_.notna() & (a_ > 0)
    trades, s_, a_ = trades[ok], s_[ok], a_[ok]

    out = {"n_trades": int(len(trades)), "by_aum": {}}
    # GP's canonical Lambda = lambda*Sigma implies per-name cost ∝ sigma_i^2. Linear impact
    # (the LQ-consistent form of sqrt impact) implies kappa_i ∝ sigma_i / ADV_i.
    names = pd.DataFrame({"s": s_, "a": a_}).groupby(level=1).median()
    kyle = names["s"] / names["a"]
    out["spearman_kyle_vs_sigma2"] = float(kyle.corr(names["s"] ** 2, method="spearman"))
    out["kyle_iqr_ratio"] = float(kyle.quantile(0.75) / kyle.quantile(0.25))
    out["kyle_p90_p10_ratio"] = float(kyle.quantile(0.9) / kyle.quantile(0.1))
    for A in AUM_GRID:
        q = trades * A                                        # traded $ per name
        c_lin = TC_LINEAR * q
        c_sqrt = C_IMPACT * s_ * q ** 1.5 / np.sqrt(a_)
        # per-name linear-impact quadratic c*s*q^2/sqrt(a*q_ref) (=1/2 kappa q^2), q_ref chosen so
        # total quadratic cost equals total sqrt cost. Per-trade quad/sqrt = sqrt(q/q_ref).
        q_ref = float((s_ * q ** 2 / np.sqrt(a_)).sum() / c_sqrt_total(s_, q, a_)) ** 2
        c_quad = C_IMPACT * s_ * q ** 2 / np.sqrt(a_ * q_ref)
        ratio = np.sqrt(q / q_ref)
        inside = (ratio >= 0.5) & (ratio <= 2.0)
        out["by_aum"][A] = {
            "q_ref_$": q_ref,
            "linear_share_of_total": float(c_lin.sum() / (c_lin.sum() + c_sqrt.sum())),
            "quad_over_sqrt_total": float(c_quad.sum() / c_sqrt.sum()),
            "sqrt_cost_share_quad_off_2x": float(c_sqrt[~inside].sum() / c_sqrt.sum()),
            "trade_size_median_over_qref": float(q.median() / q_ref),
            "linear_cost_bps_of_aum_yr": float(c_lin.sum() / A / (len(dW) / 12) * 1e4),
            "sqrt_cost_bps_of_aum_yr": float(c_sqrt.sum() / A / (len(dW) / 12) * 1e4),
        }
    return out


# ─────────────────────────────────────────────────────────── report

def sub(panel, lo, hi):
    return panel[(panel.date >= lo) & (panel.win_end < pd.Timestamp(hi))]


def main():
    panel, px, rets, adv = load_dev()
    print(f"DEV panel: {panel.date.nunique()} rebalances "
          f"[{panel.date.min().date()} → {panel.date.max().date()}], "
          f"last window ends {panel.win_end.max().date()}")
    res = {"dev_months": int(panel.date.nunique())}

    print(f"\n{'='*78}\n(a) SIGNAL DYNAMICS  cross-sectional AR slope z_(t+h) on z_t\n{'='*78}")
    sd, var = signal_dynamics(panel)
    print(f"{'h':<10}" + "".join(f"{h:>7}" for h in range(1, MAXLAG + 1)))
    for s in SIGNALS:
        print(f"{s:<10}" + "".join(f"{v:>7.3f}" for v in sd[s]["acf"]["slope"]))
        print(f"{'  a1^h':<10}" + "".join(f"{sd[s]['ar1']**h:>7.3f}" for h in range(1, MAXLAG + 1)))
    for s in SIGNALS:
        d = sd[s]
        print(f"{s:<8} AR1={d['ar1']:.3f} (t½={d['ar1_halflife_mo']:.1f}mo)  "
              f"fit c·a^h: c={d['geo_c']:.3f} a={d['geo_a']:.3f}±{d['geo_a_se']:.3f} "
              f"(t½={d['geo_halflife_mo']:.1f}mo)  max|ACF−a1^h|={d['ar1_misfit_max']:.3f}")
        print(f"{'':<8} two-OU mix: w_fast={d['mix_w_fast']:.2f} a_fast={d['mix_a_fast']:.3f} "
              f"a_slow={d['mix_a_slow']:.3f} (slow t½={np.log(.5)/np.log(d['mix_a_slow']):.1f}mo)")
    print("VAR(1) [row=target, col=regressor]  coef (se):")
    for tgt in SIGNALS:
        print(f"  {tgt:<8}" + "  ".join(f"{s}={m:+.3f}({se:.3f})" for s, (m, se) in var[tgt].items()))
    res["signal"] = {s: {k: v for k, v in d.items() if k != "acf"} | {"acf": d["acf"]["slope"].tolist()}
                     for s, d in sd.items()}
    res["var1"] = var

    print(f"\n{'='*78}\n(b) RETURN DECAY  FM slope of month-(t+h) return on f_t (joint v+q), %/mo per z\n{'='*78}")
    rd, fits = return_decay(panel, {s: sd[s]["geo_a"] for s in SIGNALS})
    for s in SIGNALS:
        d = rd[rd.signal == s]
        print(f"{s:<8} beta " + "".join(f"{b*100:>+7.3f}" for b in d["beta"]))
        print(f"{'':<8} t    " + "".join(f"{t:>+7.2f}" for t in d["t"]))
    for s in SIGNALS:
        f = fits[s]
        print(f"{s:<8} fit beta_h=b0·b^h: b0={f['beta0']*100:+.3f}%  b={f['b']:.3f} "
              f"[90% block-boot {f['b_ci90'][0]:.3f}, {f['b_ci90'][1]:.3f}]  P(b≥a)={f['b_boot_share_ge_a']:.2f}  "
              f"vs signal a={sd[s]['geo_a']:.3f}   Σ12 beta={f['sum12']*100:+.2f}%±{f['sum12_se']*100:.2f}")
        print(f"{'':<8} two-comp (slow pinned a={f['two_slow_rate']:.3f}): fast={f['two_fast']*100:+.3f}% "
              f"@{f['two_fast_rate']:.2f}/mo + slow={f['two_slow']*100:+.3f}%")
    res["return_decay"] = rd.to_dict("records")
    res["return_decay_fit"] = fits

    print(f"\n{'='*78}\nSTABILITY  (a)+(b) by subperiod\n{'='*78}")
    res["stability"] = {}
    for name, (lo, hi) in SUBPERIODS.items():
        p = sub(panel, lo, hi)
        sd_s, _ = signal_dynamics(p)
        _, f_s = return_decay(p)
        res["stability"][name] = {s: {"ar1": sd_s[s]["ar1"], "geo_a": sd_s[s]["geo_a"],
                                      "geo_c": sd_s[s]["geo_c"], **f_s[s]} for s in SIGNALS}
        for s in SIGNALS:
            r = res["stability"][name][s]
            print(f"{name}  {s:<8} AR1={r['ar1']:.3f} c={r['geo_c']:.3f} a={r['geo_a']:.3f}  "
                  f"beta0={r['beta0']*100:+.3f}%  b={r['b']:.3f}  Σ12={r['sum12']*100:+.2f}%")

    print(f"\n{'='*78}\n(c) RISK MODEL  PCA-K + diag, trailing {RISK_WIN}d; bias=std(r/σ̂) (ideal 1)\n{'='*78}")
    summ, K_star = risk_model(panel, rets)
    piv = summ.pivot(index="K", columns="kind", values=["bias", "qlike"])
    print(piv.round(3).to_string())
    n = summ["n"].iloc[0]
    print(f"95% band for bias on n={n}: ±{1.96*np.sqrt(1/(2*n)):.2f}   →  K* = {K_star} (rule: min held-portfolio QLIKE)")
    res["risk"] = {"summary": summ.to_dict("records"), "K_star": K_star}

    print(f"\n{'='*78}\n(d) COSTS  10bps linear + sqrt impact on the actual v+q book trades\n{'='*78}")
    c = costs(panel, rets, adv)
    print(f"trades: {c['n_trades']:,}   spearman(σ/ADV, σ²) = {c['spearman_kyle_vs_sigma2']:+.2f}   "
          f"cross-name spread of σ/ADV: IQR ×{c['kyle_iqr_ratio']:.1f}, p90/p10 ×{c['kyle_p90_p10_ratio']:.1f}")
    print(f"{'AUM':>8}{'q_ref $':>12}{'lin bps/yr':>12}{'sqrt bps/yr':>13}{'linear share':>14}"
          f"{'quad/sqrt':>11}{'cost quad>2x off':>18}{'median q/q_ref':>16}")
    for A, r in c["by_aum"].items():
        print(f"{A/1e6:>7.0f}M{r['q_ref_$']:>12,.0f}{r['linear_cost_bps_of_aum_yr']:>12.1f}"
              f"{r['sqrt_cost_bps_of_aum_yr']:>13.1f}{r['linear_share_of_total']:>14.0%}"
              f"{r['quad_over_sqrt_total']:>11.2f}{r['sqrt_cost_share_quad_off_2x']:>18.0%}"
              f"{r['trade_size_median_over_qref']:>16.3f}")
    res["costs"] = c

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "m1_identification.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\n→ results/m1_identification.json")


if __name__ == "__main__":
    main()
