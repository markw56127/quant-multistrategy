"""
M5 — walk-forward pessimistic selection among surrogate-derived controllers.
The VP multi-fidelity rule (vlasov-poisson FINDINGS §16), ported. Pre-registered in
README.md.

Each January from 2020, fit a ridge correction from surrogate-side features of
each candidate (model-predicted net SR, log GMV, log turnover, and their squares)
to its realised net SR over the months already closed. Then select
argmax(y_hat - k * sd_hat), where sd_hat comes from a block bootstrap over those
past months. The selected controller trades until the next January. The stitched
book, switches included, is scored like any other.

VP's lesson this tests: a correction that ranks well still selects badly under
argmax (the optimizer's curse), and pessimism is what makes selection safe.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identify import load_dev, ROOT  # noqa: E402
from m2_controller import (precompute_slopes, build_dates, run_gp, ex_ante_vol, calibrate_gamma,  # noqa: E402
                           score, sharpe, boot_diff, AUM, ANN)
from m3_trust_gap import predicted, G_GRID, K_GRID  # noqa: E402
from m4_apg import replay  # noqa: E402

K_PESS = [0, 1, 2, 4]
K_SCORED = 2
N_BOOT, BLOCK = 200, 6
SEL_YEARS = [2020, 2021, 2022, 2023, 2024]
DEFAULT = "gp_g1_k1"


def features(pr, va, sc, idx):
    p = pr.iloc[idx]
    pred_sr = (p.pred_gross - p.pred_cost).mean() / va[idx].mean() * np.sqrt(ANN)
    lg, lt = np.log(sc.gmv.iloc[idx].mean()), np.log(sc.turn.iloc[idx].mean())
    return [pred_sr, lg, lt, lg ** 2, lt ** 2]


def fit_predict(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    m = Ridge(alpha=1.0).fit((X - mu) / sd, y)
    return m.predict((X - mu) / sd)


def main():
    panel, px, rets, adv = load_dev()
    dates, acf, beta = precompute_slopes(panel)
    D = build_dates(panel, rets, adv, acf, beta, dates, AUM)
    ls_books = [pd.Series(e["w_ls"], index=e["names"]) for e in D]
    g_star = calibrate_gamma(D, ex_ante_vol(D, ls_books).mean())

    cands = {f"gp_g{g}_k{k}": run_gp(D, g * g_star, k) for g in G_GRID for k in K_GRID}
    th = json.load(open(ROOT / "results" / "m4_apg.json"))["theta"]
    theta = torch.tensor([np.log(th["g"]), np.log(th["kappa"]), np.log(th["b"]),
                          np.log(th["eta"] / (1 - th["eta"]))])
    cands["apg_m4"] = replay(theta, D, g_star)
    names = list(cands)
    sc = {n: score(D, b, AUM) for n, b in cands.items()}
    pr = {n: predicted(D, b) for n, b in cands.items()}
    va = {n: ex_ante_vol(D, b) for n, b in cands.items()}
    dts = pd.DatetimeIndex([e["date"] for e in D])
    sel_idx = [int(np.searchsorted(dts, pd.Timestamp(f"{y}-01-01"))) for y in SEL_YEARS]

    rng = np.random.default_rng(0)
    choice = {k: {} for k in K_PESS}
    choice["surrogate_value"] = {}                     # VP "objective alone": argmax model-predicted SR
    diag = []
    for s in sel_idx:
        past = np.arange(s)
        X = np.array([features(pr[n], va[n], sc[n], past) for n in names])
        y = np.array([sharpe(sc[n].net.iloc[past]) for n in names])
        yhat = fit_predict(X, y)
        starts = np.arange(len(past) - BLOCK + 1)
        boots = []
        for _ in range(N_BOOT):
            idx = np.concatenate([np.arange(b, b + BLOCK) for b in rng.choice(starts, len(past) // BLOCK + 1)])
            idx = idx[: len(past)]
            yb = np.array([sharpe(sc[n].net.iloc[idx]) for n in names])
            boots.append(fit_predict(X, yb))
        sd = np.std(boots, axis=0)
        for k in K_PESS:
            choice[k][s] = names[int(np.argmax(yhat - k * sd))]
        choice["surrogate_value"][s] = names[int(np.argmax(X[:, 0]))]
        rho = pd.Series(yhat).corr(pd.Series(y), method="spearman")
        diag.append({"date": dts[s].date(), "n_past": len(past), "ridge_in_sample_rho": rho,
                     **{f"k{k}": choice[k][s] for k in K_PESS}, "surr": choice["surrogate_value"][s]})

    print(pd.DataFrame(diag).to_string(index=False))

    q = score(D, ls_books, AUM).iloc[sel_idx[0]:]
    print(f"\n{'='*78}\nM5 walk-forward @ $100M  {q.index[0].date()} → {q.index[-1].date()} ({len(q)} mo)\n{'='*78}")
    print(f"{'book':<22}{'gross SR':>9}{'net SR':>9}{'turn/mo':>9}{'GMV':>7}{'switches':>9}")
    res = {}

    def stitched(sel):
        books, cur = [], DEFAULT
        for t in range(len(D)):
            cur = sel.get(t, cur)
            books.append(cands[cur][t])
        return score(D, books, AUM).iloc[sel_idx[0]:]

    for key in K_PESS + ["surrogate_value"]:
        df = stitched(choice[key])
        sw = len(set(choice[key].values()))
        label = f"pessimistic k={key}" if key != "surrogate_value" else "surrogate value only"
        res[str(key)] = {"gross_sr": sharpe(df.gross), "net_sr": sharpe(df.net), "df": df}
        print(f"{label:<22}{sharpe(df.gross):>+9.2f}{sharpe(df.net):>+9.2f}{df.turn.mean():>9.3f}"
              f"{df.gmv.mean():>7.2f}{sw:>9}")
    oracle = max(names, key=lambda n: sharpe(sc[n].net.iloc[sel_idx[0]:]))
    for label, df in [("M2 GP (fixed)", sc[DEFAULT].iloc[sel_idx[0]:]),
                      (f"hindsight best ({oracle})", sc[oracle].iloc[sel_idx[0]:]), ("quintile", q)]:
        print(f"{label:<22}{sharpe(df.gross):>+9.2f}{sharpe(df.net):>+9.2f}{df.turn.mean():>9.3f}{df.gmv.mean():>7.2f}")

    r = res[str(K_SCORED)]
    lo, hi = boot_diff(r["df"].net, q.net)
    print(f"\nnet SR diff (k=2) − quintile {r['net_sr']-sharpe(q.net):+.2f} [90% block-boot {lo:+.2f}, {hi:+.2f}]")
    c1 = r["net_sr"] >= sharpe(q.net) + 0.10
    c2 = r["gross_sr"] >= sharpe(q.gross) - 0.05
    print(f"BAR @ $100M (k=2): (i) net ≥ quintile+0.10: {'PASS' if c1 else 'FAIL'}   "
          f"(ii) gross ≥ quintile−0.05: {'PASS' if c2 else 'FAIL'}   →  "
          f"{'DEV PASS — one OOS look permitted' if c1 and c2 else 'DEV FAIL — OOS not touched'}")
    pd.DataFrame(diag).to_csv(ROOT / "results" / "m5_selections.csv", index=False)
    pd.concat({k: v["df"] for k, v in res.items()} | {"quintile": q}, axis=1).to_csv(
        ROOT / "results" / "m5_walkforward.csv")


if __name__ == "__main__":
    main()
