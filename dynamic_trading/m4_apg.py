"""
M4 — analytic policy gradients through a differentiable simulator of the M1 model,
optimizing against the TRUE cost function. Pre-registered in README.md.

M2 lost to costs the LQ objective can't represent (10 bps linear, borrow,
sqrt-impact). The closed form can't include them, but a simulator can, and
autodiff through the simulated dynamics gives an exact gradient of the simulated
objective with respect to the policy's parameters. This is `meq-mpo/meq_apg.py`'s
method: the LQ solution is the structure, and the gradient tunes what the
quadratic can't see.

Policy (4 parameters, GP-structured):
    Sigma_eta = (1-eta) Sigma + eta diag(Sigma)          hedging shrink -> less leverage/borrow
    x_GP      = GP closed form(x_prev; g*gamma*, kappa*Lambda, Sigma_eta)
    x         = x_prev + softshrink(x_GP - x_prev, b * c / (gamma Sigma_ii))   linear-cost band

Trained ONLY on the surrogate calibrated at the first scored rebalance
(2019-01-04, causal data), then frozen and replayed on 2019-01 → 2024-11.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identify import load_dev, ls_book, SIGNALS, TC_LINEAR, C_IMPACT, ROOT  # noqa: E402
from m2_controller import (precompute_slopes, build_dates, ex_ante_vol, calibrate_gamma, score,  # noqa: E402
                           sharpe, boot_diff, DELTA, AUM, BORROW, ANN)

torch.set_default_dtype(torch.float64)
PATHS, HORIZON, STEPS, LR, VOL_PEN = 64, 60, 300, 0.05, 100.0
EPS = 1e-6
SEED = 0


# ─────────────────────────────────────────────────────────── policy

def unpack(theta):
    log_g, log_k, log_b, eta_raw = theta
    return torch.exp(log_g), torch.exp(log_k), torch.exp(log_b), torch.sigmoid(eta_raw)


def theta_init():
    # M2 point: g=1, kappa=1, b≈0 (log 0.01), eta≈0 (sigmoid(-6)=0.0025)
    return torch.tensor([0.0, 0.0, np.log(0.01), -6.0], requires_grad=True)


class Policy:
    """Precomputes the per-date eigensystem for given theta; step() maps x_prev -> x (batched)."""

    def __init__(self, theta, Sigma, lam, streams_raw, gamma_star):
        g, k, b, eta = unpack(theta)
        gamma = g * gamma_star
        dS = torch.diagonal(Sigma)
        Se = (1 - eta) * Sigma + eta * torch.diag(dS)
        L = torch.sqrt(k * lam)
        s, V = torch.linalg.eigh(Se / torch.outer(L, L))
        c = gamma * s + 1.0
        Axx = (-(c - DELTA) + torch.sqrt((c - DELTA) ** 2 + 4 * DELTA * (c - 1))) / (2 * DELTA)
        self.J = 1.0 / (c + DELTA * Axx)
        self.V, self.L = V, L
        self.gain = [(self.J * (1 + DELTA * a * self.J / (1 - DELTA * a * self.J))) for a, _ in streams_raw]
        self.tau = b * TC_LINEAR / (gamma * dS)

    def step(self, x_prev, alphas):
        """x_prev: (P,N); alphas: list of (P,N) alpha streams (same order as construction)."""
        V, L = self.V, self.L
        y = self.J * ((x_prev * L) @ V)
        for gk, al in zip(self.gain, alphas):
            y = y + gk * ((al / L) @ V)
        x_gp = (y @ V.T) / L
        d = x_gp - x_prev
        return x_prev + d - torch.clamp(d, -self.tau, self.tau)


def sabs(x):
    return torch.sqrt(x * x + EPS ** 2)


# ─────────────────────────────────────────────────────────── surrogate simulator

def rollout(step, e, paths, horizon, seed):
    """Simulate the M1 model under policy `step(x_prev, alphas, v, q) -> x`.
    Returns (net, ex-ante vol, one-way turnover), each months x paths."""
    p = e["params"]
    Sigma = torch.tensor(e["Sigma"])
    sig, adv = torch.tensor(e["sig"]), torch.tensor(e["adv"])
    N = Sigma.shape[0]
    chol = torch.linalg.cholesky(Sigma + 1e-12 * torch.eye(N))
    gen = torch.Generator().manual_seed(seed)
    rn = lambda: torch.randn(paths, N, generator=gen)  # noqa: E731
    w = p["w"]
    v = torch.tensor(e["v"]).expand(paths, N).clone()
    q0 = torch.tensor(e["q"]).expand(paths, N)
    qf = w * q0 + np.sqrt(w * (1 - w)) * rn()           # hidden split, sampled | observed q
    qs = q0 - qf
    x = torch.zeros(paths, N)
    nets, vols, turns = [], [], []
    for _ in range(horizon):
        q = qf + qs
        alphas = [p["beta_v"] * v, w * p["beta_q"] * q, (1 - w) * p["beta_q"] * q]
        x_new = step(x, alphas, v, q)
        dx = sabs(x_new - x)
        cost = TC_LINEAR * dx.sum(1) + (C_IMPACT * sig * dx ** 1.5 * torch.sqrt(AUM / adv)).sum(1)
        borrow = BORROW / ANN * (sabs(x_new) - x_new).sum(1) / 2
        r = p["beta_v"] * v + p["beta_q"] * q + rn() @ chol.T
        nets.append((x_new * r).sum(1) - cost - borrow)
        vols.append(torch.sqrt(((x_new @ Sigma) * x_new).sum(1)))
        turns.append(dx.sum(1))
        x = x_new
        v = p["a_v"] * v + np.sqrt(1 - p["a_v"] ** 2) * rn()
        qf = p["a_f"] * qf + np.sqrt(w * (1 - p["a_f"] ** 2)) * rn()
        qs = p["a_s"] * qs + np.sqrt((1 - w) * (1 - p["a_s"] ** 2)) * rn()
    return torch.stack(nets), torch.stack(vols), torch.stack(turns)


def simulate(theta, e, gamma_star, target_vol, paths, horizon, seed):
    """M4 objective: mean net monthly return with an ex-ante vol-target penalty."""
    p = e["params"]
    pol = Policy(theta, torch.tensor(e["Sigma"]), torch.tensor(e["lam"]),
                 [(p["a_v"], None), (p["a_f"], None), (p["a_s"], None)], gamma_star)
    nets, vols, _ = rollout(lambda x, al, v, q: pol.step(x, al), e, paths, horizon, seed)
    obj = nets.mean() - VOL_PEN * (vols.mean() / target_vol - 1) ** 2
    return obj, nets, vols


# ─────────────────────────────────────────────────────────── history replay

def replay(theta, D, gamma_star):
    books = []
    x_prev = pd.Series(dtype=float)
    with torch.no_grad():
        for e in D:
            p = e["params"]
            pol = Policy(theta, torch.tensor(e["Sigma"]), torch.tensor(e["lam"]),
                         [(p["a_v"], None), (p["a_f"], None), (p["a_s"], None)], gamma_star)
            v, q, w = torch.tensor(e["v"])[None], torch.tensor(e["q"])[None], p["w"]
            alphas = [p["beta_v"] * v, w * p["beta_q"] * q, (1 - w) * p["beta_q"] * q]
            xp = torch.tensor(x_prev.reindex(e["names"]).fillna(0.0).values)[None]
            x = pol.step(xp, alphas)[0].numpy()
            x_prev = pd.Series(x, index=e["names"])
            books.append(x_prev)
    return books


def main():
    panel, px, rets, adv = load_dev()
    dates, acf, beta = precompute_slopes(panel)
    D = build_dates(panel, rets, adv, acf, beta, dates, AUM)
    ls_books = [pd.Series(e["w_ls"], index=e["names"]) for e in D]
    gamma_star = calibrate_gamma(D, ex_ante_vol(D, ls_books).mean())     # M2's gamma*, risk-model only
    e0 = D[0]
    target0 = float(np.sqrt(e0["w_ls"] @ e0["Sigma"] @ e0["w_ls"]))
    print(f"surrogate calibrated at {e0['date'].date()}: N={len(e0['names'])}, "
          f"gamma*={gamma_star:.4g}, target ex-ante vol {target0:.4f}")

    theta = theta_init()
    opt = torch.optim.Adam([theta], lr=LR)
    log = []
    for it in range(STEPS):
        opt.zero_grad()
        obj, nets, vols = simulate(theta, e0, gamma_star, target0, PATHS, HORIZON, seed=SEED + it)
        (-obj).backward()
        opt.step()
        g, k, b, eta = [float(t) for t in unpack(theta.detach())]
        log.append({"it": it, "obj": float(obj), "net_ann": float(nets.mean()) * ANN,
                    "vol": float(vols.mean()), "g": g, "kappa": k, "b": b, "eta": eta})
        if it % 25 == 0 or it == STEPS - 1:
            print(f"it {it:>3}  obj {float(obj):+.5f}  sim net/yr {float(nets.mean())*ANN:+.2%}  "
                  f"vol {float(vols.mean()):.4f}  g={g:.3f} κ={k:.2f} b={b:.3f} η={eta:.3f}")
    pd.DataFrame(log).to_csv(ROOT / "results" / "m4_train_log.csv", index=False)
    theta = theta.detach()

    # sim-predicted performance of the frozen policy, fresh seeds; M2's policy for reference
    def sim_sr(th):
        with torch.no_grad():
            _, nets, vols = simulate(th, e0, gamma_star, target0, 256, HORIZON, seed=10_000)
        # per-path time-series Sharpe (nets: months x paths), averaged over paths
        return float((nets.mean(0) / nets.std(0)).mean() * np.sqrt(ANN)), float(vols.mean())
    sim_apg, sim_m2 = sim_sr(theta), sim_sr(theta_init().detach())

    apg = score(D, replay(theta, D, gamma_star), AUM)
    m2 = score(D, replay(theta_init().detach(), D, gamma_star), AUM)
    ls = score(D, ls_books, AUM)

    print(f"\n{'='*78}\nM4 dev replay @ $100M  {apg.index[0].date()} → {apg.index[-1].date()} ({len(apg)} mo)\n{'='*78}")
    print(f"{'':<14}{'sim SR':>8}{'gross SR':>9}{'net SR':>9}{'real vol':>9}{'turn/mo':>9}{'GMV':>7}"
          f"{'lin':>7}{'imp':>7}{'borrow':>8}")
    rows = {}
    for name, df, sim in [("APG (M4)", apg, sim_apg), ("GP (M2≈θ0)", m2, sim_m2), ("quintile", ls, (np.nan,))]:
        rows[name] = {"gross_sr": sharpe(df.gross), "net_sr": sharpe(df.net), "sim_sr": sim[0]}
        print(f"{name:<14}{sim[0]:>+8.2f}{sharpe(df.gross):>+9.2f}{sharpe(df.net):>+9.2f}"
              f"{df.net.std()*np.sqrt(ANN):>9.2%}{df.turn.mean():>9.3f}{df.gmv.mean():>7.2f}"
              f"{df.lin.mean()*ANN*1e4:>7.1f}{df.imp.mean()*ANN*1e4:>7.1f}{df.borrow.mean()*ANN*1e4:>8.1f}")
    lo, hi = boot_diff(apg.net, ls.net)
    a, q = rows["APG (M4)"], rows["quintile"]
    print(f"net SR diff APG−quintile {a['net_sr']-q['net_sr']:+.2f} [90% block-boot {lo:+.2f}, {hi:+.2f}]")
    print(f"trust gap (APG): sim-predicted SR {a['sim_sr']:+.2f} → realised dev {a['net_sr']:+.2f}")
    c1, c2 = a["net_sr"] >= q["net_sr"] + 0.10, a["gross_sr"] >= q["gross_sr"] - 0.05
    print(f"\nBAR @ $100M: (i) net ≥ quintile+0.10: {'PASS' if c1 else 'FAIL'}   "
          f"(ii) gross ≥ quintile−0.05: {'PASS' if c2 else 'FAIL'}   →  "
          f"{'DEV PASS — one OOS look permitted' if c1 and c2 else 'DEV FAIL — OOS not touched'}")
    g, k, b, eta = [float(t) for t in unpack(theta)]
    json.dump({"theta": {"g": g, "kappa": k, "b": b, "eta": eta}, "rows": rows,
               "boot90": [lo, hi], "gamma_star": gamma_star},
              open(ROOT / "results" / "m4_apg.json", "w"), indent=1)
    pd.concat({"apg": apg, "m2": m2, "quintile": ls}, axis=1).to_csv(ROOT / "results" / "m4_dev_100M.csv")


if __name__ == "__main__":
    main()
