"""
M3 — the trust gap: what the LQ model predicts vs what history delivers, across
controller aggressiveness. Dev only. DESCRIPTIVE: no configuration here is adopted.

The MEQ M3 / VP trust study, ported. Sweep risk aversion (gamma multiplier g on the
vol-matched gamma*) and cost aversion (kappa on the identified Lambda). For each
controller, record three model-vs-truth ratios:

  alpha      realised gross return / model-predicted  sum_t x_t' mu_t
  risk       realised vol           / ex-ante vol       sqrt(x' Sigma x)
  cost       true cost (10bps + sqrt impact) / modelled quadratic 1/2 dx' Lambda dx
             (+ borrow, which the model omits entirely, reported separately)

Then ask the VP question: which cheap, model-side feature of a controller predicts
its shortfall (model-predicted net minus realised net)? In VP it was control
magnitude (rank corr +0.67 for max|H|); the candidates here are GMV, turnover,
and the two knobs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identify import load_dev, ROOT  # noqa: E402
from m2_controller import (precompute_slopes, build_dates, run_gp, ex_ante_vol, calibrate_gamma,  # noqa: E402
                           score, sharpe, ls_book, AUM, ANN)

G_GRID = [0.25, 0.5, 1, 2, 4, 8]
K_GRID = [0.25, 1, 4, 16, 64]


def predicted(D, books):
    """Model-side monthly expected gross return and quadratic trading cost."""
    prev = pd.Series(dtype=float)
    lam_last = {}
    rows = []
    for e, x in zip(D, books):
        lam_last.update(zip(e["names"], e["lam"]))
        allnames = prev.index.union(x.index)
        dx = x.reindex(allnames).fillna(0.0) - prev.reindex(allnames).fillna(0.0)
        lam = np.array([lam_last[n] for n in allnames])
        rows.append({"pred_gross": float(x.values @ e["mu"]),
                     "pred_cost": float(0.5 * (lam * dx.values ** 2).sum())})
        prev = x
    return pd.DataFrame(rows)


def main():
    panel, px, rets, adv = load_dev()
    dates, acf, beta = precompute_slopes(panel)
    D = build_dates(panel, rets, adv, acf, beta, dates, AUM)
    ls_books = [pd.Series(e["w_ls"], index=e["names"]) for e in D]
    target = ex_ante_vol(D, ls_books).mean()
    g_star = calibrate_gamma(D, target)

    rows = []
    for name, books in [("quintile", ls_books)] + [
            (f"g{g}_k{k}", run_gp(D, g * g_star, k)) for g in G_GRID for k in K_GRID]:
        sc = score(D, books, AUM)
        pr = predicted(D, books)
        va = ex_ante_vol(D, books)
        g, k = (np.nan, np.nan) if name == "quintile" else map(float, name[1:].split("_k"))
        rows.append({
            "ctrl": name, "g": g, "k": k,
            "gmv": sc.gmv.mean(), "turn": sc.turn.mean(),
            "gross_sr": sharpe(sc.gross), "net_sr": sharpe(sc.net),
            "alpha_ratio": sc.gross.mean() / pr.pred_gross.mean(),
            "risk_ratio": sc.gross.std() / va.mean(),
            "cost_ratio": (sc.lin + sc.imp).mean() / pr.pred_cost.mean(),
            "borrow_bps": sc.borrow.mean() * ANN * 1e4,
            "pred_net_ann": (pr.pred_gross - pr.pred_cost).mean() * ANN,
            "real_net_ann": sc.net.mean() * ANN,
        })
    df = pd.DataFrame(rows)
    df["shortfall_ann"] = df.pred_net_ann - df.real_net_ann
    # normalise shortfall by predicted gross so it is comparable across scales
    df["shortfall_rel"] = df.shortfall_ann / (df.pred_net_ann.abs() + 1e-12)

    pd.set_option("display.width", 200)
    print(f"gamma* = {g_star:.4g}  (vol-matched, M2)   AUM $100M   {len(D)} months\n")
    cols = ["ctrl", "gmv", "turn", "gross_sr", "net_sr", "alpha_ratio", "risk_ratio",
            "cost_ratio", "borrow_bps", "pred_net_ann", "real_net_ann"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    gp = df[df.ctrl != "quintile"]
    print("\nSpearman rank corr across the 30 GP controllers:")
    feats = ["gmv", "turn", "g", "k"]
    for tgt in ["shortfall_ann", "risk_ratio", "alpha_ratio", "cost_ratio", "net_sr"]:
        print(f"  {tgt:<14}" + "".join(f"{f}={gp[f].corr(gp[tgt], method='spearman'):+.2f}  " for f in feats))

    q = df[df.ctrl == "quintile"].iloc[0]
    best = gp.loc[gp.net_sr.idxmax()]
    print(f"\nquintile net SR {q.net_sr:+.2f};  best GP in grid {best.ctrl} net SR {best.net_sr:+.2f} "
          f"(max over 30 on dev = selection-biased, NOT adoptable)")
    print(f"GP controllers with net SR > quintile: {(gp.net_sr > q.net_sr).sum()}/30")
    df.to_csv(ROOT / "results" / "m3_trust_gap.csv", index=False)


if __name__ == "__main__":
    main()
