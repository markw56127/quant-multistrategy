"""
The closed-form GP controller in m2_controller.run_gp, checked against a solver that
shares none of its derivation: the deterministic finite-horizon problem

    max  sum_t delta^t [ x_t' alpha_t - gamma/2 x_t' Sigma x_t - 1/2 dx_t' Lambda dx_t ]
    alpha_t = sum_k a_k^t alpha_k0

solved as one stacked quadratic program. By certainty equivalence the stochastic
policy's first move equals this problem's first move, and at a long horizon the
finite-horizon problem converges to the stationary one.
"""

import numpy as np
import pandas as pd

from m2_controller import DELTA, run_gp


def _problem(seed=0, N=5):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((N, N))
    Sigma = G @ G.T / N + 0.1 * np.eye(N)
    lam = rng.uniform(0.5, 5.0, N)
    streams = [(0.97, rng.standard_normal(N) * 0.01), (0.6, rng.standard_normal(N) * 0.01)]
    return Sigma, lam, streams, rng.standard_normal(N) * 0.05


def _entry(Sigma, lam, streams, names):
    L = np.sqrt(lam)
    s, V = np.linalg.eigh(Sigma / np.outer(L, L))
    return {"names": names, "Sigma": Sigma, "s": s, "V": V, "L": L,
            "streams": [(a, V.T @ (al / L)) for a, al in streams]}


def _qp_first_move(Sigma, lam, streams, x_prev, gamma, H=600):
    N = len(x_prev)
    Lam = np.diag(lam)
    n = N * H
    Q = np.zeros((n, n))
    c = np.zeros(n)
    for t in range(H):
        d = DELTA ** t
        i = slice(t * N, (t + 1) * N)
        Q[i, i] += d * (gamma * Sigma + Lam)
        c[i] += d * sum(a ** t * al for a, al in streams)
        if t == 0:
            c[i] += Lam @ x_prev
        else:
            j = slice((t - 1) * N, t * N)
            Q[j, j] += d * Lam
            Q[i, j] -= d * Lam
            Q[j, i] -= d * Lam
    return np.linalg.solve(Q, c)[:N]


def test_closed_form_matches_stacked_qp():
    for seed in range(3):
        for gamma in [0.5, 5.0, 50.0]:
            Sigma, lam, streams, x_prev = _problem(seed)
            names = [f"n{i}" for i in range(len(x_prev))]
            # run_gp starts from zero holdings; seed x_prev through a preceding step whose
            # output we override, so the second step is tested from an arbitrary x_prev
            e = _entry(Sigma, lam, streams, names)
            x0 = run_gp([e], gamma)[0].values                      # from x_prev = 0
            qp0 = _qp_first_move(Sigma, lam, streams, np.zeros_like(x_prev), gamma)
            assert np.allclose(x0, qp0, rtol=1e-6, atol=1e-10), (seed, gamma, x0, qp0)


def test_policy_is_linear_in_x_prev():
    """x = J x_prev + aim-term: check the x_prev response against the QP too."""
    Sigma, lam, streams, x_prev = _problem(7)
    gamma = 5.0
    zero = [(a, 0 * al) for a, al in streams]
    qp = _qp_first_move(Sigma, lam, zero, x_prev, gamma)
    names = [f"n{i}" for i in range(len(x_prev))]
    L = np.sqrt(lam)
    s, V = np.linalg.eigh(Sigma / np.outer(L, L))
    c = gamma * s + 1
    Axx = (-(c - DELTA) + np.sqrt((c - DELTA) ** 2 + 4 * DELTA * (c - 1))) / (2 * DELTA)
    J = 1 / (c + DELTA * Axx)
    x = (V @ (J * (V.T @ (L * x_prev)))) / L
    assert np.allclose(x, qp, rtol=1e-6, atol=1e-12)


def test_no_cost_limit_is_markowitz_on_the_first_move():
    """Lambda -> 0: trade fully to (gamma Sigma)^-1 alpha_0."""
    Sigma, _, streams, _ = _problem(3)
    lam = np.full(len(Sigma), 1e-9)
    e = _entry(Sigma, lam, streams, [f"n{i}" for i in range(len(Sigma))])
    x = run_gp([e], 2.0)[0].values
    mk = np.linalg.solve(2.0 * Sigma, sum(al for _, al in streams))
    assert np.allclose(x, mk, rtol=1e-5)


def test_kappa_shortcut_equals_rescaled_lambda():
    Sigma, lam, streams, _ = _problem(5)
    names = [f"n{i}" for i in range(len(lam))]
    e1, e2 = _entry(Sigma, lam, streams, names), _entry(Sigma, 3.7 * lam, streams, names)
    a = run_gp([e1, e1], 4.0, kappa=3.7)
    b = run_gp([e2, e2], 4.0)
    assert all(np.allclose(x.values, y.values, rtol=1e-8) for x, y in zip(a, b))


# ─────────────────────────────────────────── M4: torch policy and simulator gradients

def _sim_entry(seed=0, N=6):
    Sigma, lam, _, _ = _problem(seed, N)
    rng = np.random.default_rng(seed + 100)
    return {"Sigma": Sigma * 0.01, "lam": lam * 0.1, "sig": rng.uniform(0.01, 0.03, N),
            "adv": rng.uniform(1e7, 1e8, N), "v": rng.standard_normal(N), "q": rng.standard_normal(N),
            "names": [f"n{i}" for i in range(N)],
            "params": {"a_v": 0.98, "w": 0.3, "a_f": 0.8, "a_s": 0.97, "beta_v": 0.002, "beta_q": 0.001}}


def test_torch_policy_reduces_to_closed_form():
    """b -> 0, eta -> 0, g = kappa = 1: the M4 policy is exactly M2's GP."""
    import torch
    from m4_apg import Policy
    e = _sim_entry(1)
    p = e["params"]
    streams = [(p["a_v"], p["beta_v"] * e["v"]), (p["a_f"], p["w"] * p["beta_q"] * e["q"]),
               (p["a_s"], (1 - p["w"]) * p["beta_q"] * e["q"])]
    gamma = 30.0
    ref = run_gp([_entry(e["Sigma"], e["lam"], streams, e["names"])] * 2, gamma)
    theta = torch.tensor([0.0, 0.0, -50.0, -50.0])
    pol = Policy(theta, torch.tensor(e["Sigma"]), torch.tensor(e["lam"]), streams, gamma)
    al = [torch.tensor(a)[None] for _, a in streams]
    x1 = pol.step(torch.zeros(1, len(e["v"])), al)
    x2 = pol.step(x1, al)
    assert np.allclose(x1[0].numpy(), ref[0].values, rtol=1e-8)
    assert np.allclose(x2[0].numpy(), ref[1].values, rtol=1e-8)


def test_simulator_gradient_matches_finite_differences():
    import torch
    from m4_apg import simulate
    e = _sim_entry(2)
    theta = torch.tensor([0.1, 0.3, np.log(0.5), -1.0], requires_grad=True)
    obj, _, _ = simulate(theta, e, 30.0, 0.01, paths=8, horizon=12, seed=3)
    obj.backward()
    g = theta.grad.numpy()
    h = 1e-6
    for i in range(4):
        tp, tm = theta.detach().clone(), theta.detach().clone()
        tp[i] += h
        tm[i] -= h
        fd = (float(simulate(tp, e, 30.0, 0.01, 8, 12, 3)[0]) - float(simulate(tm, e, 30.0, 0.01, 8, 12, 3)[0])) / (2 * h)
        assert np.isclose(g[i], fd, rtol=1e-4, atol=1e-10), (i, g[i], fd)


# ─────────────────────────────────────────── M6: neural residual policy

def _m6_step(model, e, perm=None):
    import torch
    from m6_mbrl import make_step
    if perm is not None:
        e = {**e, "Sigma": e["Sigma"][np.ix_(perm, perm)], "lam": e["lam"][perm], "sig": e["sig"][perm],
             "adv": e["adv"][perm], "v": e["v"][perm], "q": e["q"][perm]}
    p = e["params"]
    v, q = torch.tensor(e["v"])[None], torch.tensor(e["q"])[None]
    al = [p["beta_v"] * v, p["w"] * p["beta_q"] * q, (1 - p["w"]) * p["beta_q"] * q]
    step = make_step(model, e, 30.0)
    x1 = step(torch.zeros_like(v), al, v, q)
    return step(x1, al, v, q)[0].detach().numpy()


def test_m6_zero_init_is_exactly_gp():
    import torch
    from m4_apg import Policy
    from m6_mbrl import Residual, THETA_M2
    e = _sim_entry(4)
    p = e["params"]
    gp = Policy(THETA_M2, torch.tensor(e["Sigma"]), torch.tensor(e["lam"]),
                [(p["a_v"], None), (p["a_f"], None), (p["a_s"], None)], 30.0)
    v, q = torch.tensor(e["v"])[None], torch.tensor(e["q"])[None]
    al = [p["beta_v"] * v, p["w"] * p["beta_q"] * q, (1 - p["w"]) * p["beta_q"] * q]
    ref = gp.step(gp.step(torch.zeros_like(v), al), al)[0].numpy()
    assert np.allclose(_m6_step(Residual(), e), ref, rtol=1e-12, atol=1e-15)


def test_m6_policy_is_permutation_equivariant():
    import torch
    from m6_mbrl import Residual
    torch.manual_seed(1)
    m = Residual()
    torch.nn.init.normal_(m.net[-1].weight, std=0.5)           # non-trivial residual
    e = _sim_entry(4)
    perm = np.random.default_rng(0).permutation(len(e["v"]))
    assert np.allclose(_m6_step(m, e)[perm], _m6_step(m, e, perm), rtol=1e-9, atol=1e-14)
