"""
M6 — model-based RL: a neural policy trained by analytic policy gradients in the
M1 surrogate, under a VP-style effort penalty. Pre-registered in README.md.

    x = x_GP + scale * MLP(features_i)      one MLP shared across stocks (permutation-equivariant)

x_GP is the M2 closed form, and the MLP's last layer starts at zero, so training
starts exactly at M2. The objective is M4's (mean net return, vol-target penalty)
minus rho_e * turnover, with rho_e set once at initialisation so the effort term
is 10% of |mean sim net return| (vlasov-poisson FINDINGS §15: an effort penalty is
a trust constraint, not an actuator limit).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identify import load_dev, ROOT  # noqa: E402
from m2_controller import (precompute_slopes, build_dates, ex_ante_vol, calibrate_gamma, score,  # noqa: E402
                           sharpe, boot_diff, AUM, ANN)
from m4_apg import Policy, rollout, VOL_PEN, PATHS, HORIZON  # noqa: E402

STEPS, LR, HIDDEN, EFFORT_SHARE = 300, 1e-3, 32, 0.10
THETA_M2 = torch.tensor([0.0, 0.0, -50.0, -50.0])      # g = kappa = 1, no band, no shrink
N_FEAT = 8


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, HIDDEN), nn.Tanh(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, F):                                # F: (P, N, N_FEAT)
        return self.net(F).squeeze(-1)


def _xs_std(f):
    """Cross-sectional standardisation per path (last axis = stocks)."""
    return (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) + 1e-12)


def make_step(model, e, gamma_star):
    p = e["params"]
    Sigma = torch.tensor(e["Sigma"])
    gp = Policy(THETA_M2, Sigma, torch.tensor(e["lam"]),
                [(p["a_v"], None), (p["a_f"], None), (p["a_s"], None)], gamma_star)
    static = [torch.log(torch.tensor(e["sig"])), torch.log(torch.tensor(e["adv"])),
              torch.log(torch.tensor(e["lam"]))]

    def step(x_prev, alphas, v, q):
        x_gp = gp.step(x_prev, alphas)
        P = x_prev.shape[0]
        feats = [x_prev, x_gp, v, q, x_prev @ Sigma] + [s.expand(P, -1) for s in static]
        F = torch.stack([_xs_std(f) for f in feats], -1)
        scale = x_gp.abs().mean(-1, keepdim=True)
        return x_gp + scale * model(F)
    return step


def objective(model, e, gamma_star, target, rho_e, paths, seed):
    nets, vols, turns = rollout(make_step(model, e, gamma_star), e, paths, HORIZON, seed)
    obj = nets.mean() - VOL_PEN * (vols.mean() / target - 1) ** 2 - rho_e * turns.mean()
    return obj, nets, vols, turns


def replay(model, D, gamma_star):
    books, x_prev = [], pd.Series(dtype=float)
    with torch.no_grad():
        for e in D:
            p, w = e["params"], e["params"]["w"]
            v, q = torch.tensor(e["v"])[None], torch.tensor(e["q"])[None]
            alphas = [p["beta_v"] * v, w * p["beta_q"] * q, (1 - w) * p["beta_q"] * q]
            xp = torch.tensor(x_prev.reindex(e["names"]).fillna(0.0).values)[None]
            x = make_step(model, e, gamma_star)(xp, alphas, v, q)[0].numpy()
            x_prev = pd.Series(x, index=e["names"])
            books.append(x_prev)
    return books


def main():
    torch.manual_seed(0)
    panel, px, rets, adv = load_dev()
    dates, acf, beta = precompute_slopes(panel)
    D = build_dates(panel, rets, adv, acf, beta, dates, AUM)
    ls_books = [pd.Series(e["w_ls"], index=e["names"]) for e in D]
    gamma_star = calibrate_gamma(D, ex_ante_vol(D, ls_books).mean())
    e0 = D[0]
    target = float(np.sqrt(e0["w_ls"] @ e0["Sigma"] @ e0["w_ls"]))

    model = Residual()
    with torch.no_grad():
        _, nets0, _, turns0 = objective(model, e0, gamma_star, target, 0.0, PATHS, seed=0)
    rho_e = EFFORT_SHARE * float(nets0.mean().abs()) / float(turns0.mean())
    print(f"surrogate @ {e0['date'].date()}  N={len(e0['names'])}  rho_e={rho_e:.4g} "
          f"(init: sim net/mo {float(nets0.mean()):+.5f}, turnover {float(turns0.mean()):.3f})")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    log = []
    for it in range(STEPS):
        opt.zero_grad()
        obj, nets, vols, turns = objective(model, e0, gamma_star, target, rho_e, PATHS, seed=1 + it)
        (-obj).backward()
        opt.step()
        log.append({"it": it, "obj": obj.item(), "net_ann": nets.mean().item() * ANN,
                    "vol": vols.mean().item(), "turn": turns.mean().item()})
        if it % 25 == 0 or it == STEPS - 1:
            print(f"it {it:>3}  obj {obj.item():+.5f}  sim net/yr {nets.mean().item()*ANN:+.2%}  "
                  f"vol {vols.mean().item():.4f}  turn {turns.mean().item():.3f}")
    pd.DataFrame(log).to_csv(ROOT / "results" / "m6_train_log.csv", index=False)
    torch.save(model.state_dict(), ROOT / "results" / "m6_policy.pt")

    def sim_sr(m):
        with torch.no_grad():
            _, nets, _, _ = objective(m, e0, gamma_star, target, 0.0, 256, seed=10_000)
        return float((nets.mean(0) / nets.std(0)).mean() * np.sqrt(ANN))

    mbrl = score(D, replay(model, D, gamma_star), AUM)
    m2 = score(D, replay(Residual(), D, gamma_star), AUM)
    ls = score(D, ls_books, AUM)
    print(f"\n{'='*78}\nM6 dev replay @ $100M  {mbrl.index[0].date()} → {mbrl.index[-1].date()} ({len(mbrl)} mo)\n{'='*78}")
    print(f"{'':<14}{'sim SR':>8}{'gross SR':>9}{'net SR':>9}{'real vol':>9}{'turn/mo':>9}{'GMV':>7}"
          f"{'lin':>7}{'imp':>7}{'borrow':>8}")
    rows = {}
    for name, df, s in [("MBRL (M6)", mbrl, sim_sr(model)), ("GP (M2)", m2, sim_sr(Residual())),
                        ("quintile", ls, np.nan)]:
        rows[name] = {"gross": sharpe(df.gross), "net": sharpe(df.net), "sim": s}
        print(f"{name:<14}{s:>+8.2f}{sharpe(df.gross):>+9.2f}{sharpe(df.net):>+9.2f}"
              f"{df.net.std()*np.sqrt(ANN):>9.2%}{df.turn.mean():>9.3f}{df.gmv.mean():>7.2f}"
              f"{df.lin.mean()*ANN*1e4:>7.1f}{df.imp.mean()*ANN*1e4:>7.1f}{df.borrow.mean()*ANN*1e4:>8.1f}")
    lo, hi = boot_diff(mbrl.net, ls.net)
    a, q = rows["MBRL (M6)"], rows["quintile"]
    print(f"net SR diff MBRL−quintile {a['net']-q['net']:+.2f} [90% block-boot {lo:+.2f}, {hi:+.2f}]")
    print(f"trust gap (MBRL): sim-predicted SR {a['sim']:+.2f} → realised dev {a['net']:+.2f}")
    c1, c2 = a["net"] >= q["net"] + 0.10, a["gross"] >= q["gross"] - 0.05
    print(f"\nBAR @ $100M: (i) net ≥ quintile+0.10: {'PASS' if c1 else 'FAIL'}   "
          f"(ii) gross ≥ quintile−0.05: {'PASS' if c2 else 'FAIL'}   →  "
          f"{'DEV PASS — one OOS look permitted' if c1 and c2 else 'DEV FAIL — OOS not touched'}")
    pd.concat({"mbrl": mbrl, "m2": m2, "quintile": ls}, axis=1).to_csv(ROOT / "results" / "m6_dev_100M.csv")


if __name__ == "__main__":
    main()
