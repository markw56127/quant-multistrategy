"""
A4: the multigroup covariance model (course topic 6), pre-registered in README.md.

Sigma_ij = rho_{g(i) g(j)} * sigma_i * sigma_j, where rho_kk is the mean pairwise correlation
within group k and rho_kl the mean correlation between groups k and l. The test is the
realized variance of the long-only GMV portfolio vs the sample covariance, on a BACKCAST
(1996-2008, Vanguard funds) and on OOS (2025-01 -> 2026-08, ETFs). Dev (2009-24) is
reported for reference only: it is not blind.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import run
from run import (formation_inputs, run_book, w_gmv, cov_sample, cov_const_corr, block_boot, CACHE as ETF_CACHE)

ROOT = Path(__file__).resolve().parent
BACK_CACHE = ROOT / "cache" / "backcast_px.parquet"
GROUPS_ETF = {"SPY": "eq", "EFA": "eq", "EEM": "eq", "VNQ": "eq", "IEF": "tsy", "TLT": "tsy",
              "LQD": "credit", "GLD": "real", "DBC": "real"}
BACK = {"VFINX": ("eq", "SPY"), "VGTSX": ("eq", "EFA"), "VEIEX": ("eq", "EEM"), "VGSIX": ("eq", "VNQ"),
        "VFITX": ("tsy", "IEF"), "VUSTX": ("tsy", "TLT"), "VFICX": ("credit", "LQD"), "VGPMX": ("real", "GLD")}


def make_multigroup(groups):
    g = np.array(groups)

    def cov(R, *_):
        C = np.corrcoef(R, rowvar=False)
        sd = R.std(0, ddof=1)
        keys = sorted(set(g))
        rho = {}
        for a in keys:
            for b in keys:
                ia, ib = np.where(g == a)[0], np.where(g == b)[0]
                vals = [C[i, j] for i in ia for j in ib if i != j]
                rho[a, b] = np.mean(vals) if vals else 0.0
        P = np.array([[rho[g[i], g[j]] for j in range(len(g))] for i in range(len(g))])
        np.fill_diagonal(P, 1.0)
        return P * np.outer(sd, sd)
    return cov


def make_single_index(idx):
    def cov(R, *_):
        m = R[:, idx]
        vm = m.var(ddof=1)
        beta = np.array([np.cov(R[:, i], m)[0, 1] / vm for i in range(R.shape[1])])
        s2e = (R - R.mean(0) - np.outer(m - m.mean(), beta)).var(0, ddof=1)
        s2e[idx] = 0.0
        return np.outer(beta, beta) * vm + np.diag(s2e)
    return cov


def load_generic(px, assets, start_hold, end_hold):
    px = px[assets].dropna()
    daily = px.pct_change().dropna()
    monthly = px.resample("ME").last().pct_change().dropna()
    rf = pd.Series(0.0, index=monthly.index)          # the GMV comparison does not use rf
    inputs = formation_inputs(daily, monthly, rf)
    return [x for x in inputs if start_hold <= x["hold"] < end_hold]


def compare(label, inputs, ests, clean):
    books = {k: run_book(inputs, w_gmv, f) for k, f in ests.items()}
    base = books["sample"].ret
    print(f"\n{label}: {len(inputs)} holding months [{inputs[0]['hold'].date()} → {inputs[-1]['hold'].date()}]"
          f"{'' if clean else '   (NOT blind — reference only)'}")
    print(f"  {'estimator':<13}{'real vol':>9}{'bias':>7}{'var/sample':>11}{'90% CI':>18}")
    out = {}
    for k, b in books.items():
        vr = b.ret.var() / base.var()
        lo, hi = block_boot(lambda i: b.ret.iloc[i].var() / base.iloc[i].var(), len(b)) if k != "sample" else (1, 1)
        out[k] = (vr, lo, hi)
        print(f"  {k:<13}{b.ret.std()*np.sqrt(12):>9.2%}{(b.ret / b.sigma_hat).std():>7.2f}{vr:>11.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}")
    return out


def main():
    etf = pd.read_parquet(ETF_CACHE)
    assets = run.ASSETS
    ests_etf = {"sample": cov_sample, "multigroup": make_multigroup([GROUPS_ETF[a] for a in assets]),
                "single_index": make_single_index(assets.index("SPY")), "const_corr": cov_const_corr}

    if BACK_CACHE.exists():
        bpx = pd.read_parquet(BACK_CACHE)
    else:
        bpx = yf.download(list(BACK), start="1996-01-01", end="2009-01-01", auto_adjust=True, progress=False)["Close"]
        bpx.to_parquet(BACK_CACHE)
    bassets = list(BACK)
    ests_back = {"sample": cov_sample, "multigroup": make_multigroup([BACK[a][0] for a in bassets]),
                 "single_index": make_single_index(bassets.index("VFINX")), "const_corr": cov_const_corr}

    back = compare("BACKCAST (primary)", load_generic(bpx, bassets, pd.Timestamp("1999-01-01"),
                                                      pd.Timestamp("2009-01-01")), ests_back, True)
    oos = compare("OOS (ETFs)", load_generic(etf, assets, pd.Timestamp("2025-01-01"),
                                             pd.Timestamp("2026-09-01")), ests_etf, True)
    compare("DEV (ETFs)", load_generic(etf, assets, pd.Timestamp("2009-03-01"), pd.Timestamp("2025-01-01")),
            ests_etf, False)

    vb, lo, hi = back["multigroup"]
    vo = oos["multigroup"][0]
    passed = vb < 1 and vo < 1
    print(f"\nA4: multigroup/sample variance ratio: backcast {vb:.3f} [CI upper {hi:.3f}], OOS {vo:.3f}  →  "
          f"{'STRONG PASS' if passed and hi < 1 else 'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
