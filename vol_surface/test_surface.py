"""Synthetic-truth tests for surface.py: each case has a known right answer."""

import numpy as np
import pandas as pd

from surface import b76, implied_vol, implied_forward, expiry_ivs, svi_w, fit_svi, svi_butterfly_ok, arbitrage_checks


def test_iv_round_trip():
    for call in (True, False):
        for K in (80, 100, 125):
            for vol in (0.08, 0.2, 0.6):
                p = b76(100.0, K, 0.25, 0.99, vol, call)
                assert abs(implied_vol(p, 100.0, K, 0.25, 0.99, call) - vol) < 1e-6


def _chain(F=5000.0, D=0.996, T=0.1, smile=lambda k: 0.15 - 0.3 * k + 0.5 * k * k):
    rows = []
    for K in np.arange(0.7 * F, 1.3 * F, 25.0):
        v = smile(np.log(K / F))
        for call in (True, False):
            px = b76(F, K, T, D, v, call)
            rows.append({"strike": K, "type": "call" if call else "put", "mid": px, "T": T})
    return pd.DataFrame(rows)


def test_parity_recovers_forward_and_discount():
    g = _chain(F=5123.0, D=0.9955)                # ≈4.5% rate over 0.1y
    F, D, how = implied_forward(g, spot=5000.0, T=0.1, rate=0.04)
    assert how == "parity" and abs(F - 5123.0) < 1e-6 and abs(D - 0.9955) < 1e-9


def test_otm_ivs_match_generating_smile():
    smile = lambda k: 0.15 - 0.3 * k + 0.5 * k * k  # noqa: E731
    otm, F, D, _ = expiry_ivs(_chain(smile=smile), spot=5000.0, rate=0.0)
    assert np.allclose(otm.iv, smile(otm.k), atol=1e-6)
    assert arbitrage_checks(otm, F, D) == (0, 0)


def test_svi_recovers_parameters():
    p = (0.004, 0.08, -0.6, 0.02, 0.15)
    k = np.linspace(-0.5, 0.3, 40)
    q = fit_svi(k, svi_w(k, *p))
    assert np.allclose(svi_w(k, *q), svi_w(k, *p), atol=1e-7)
    assert svi_butterfly_ok(q)


def test_butterfly_check_flags_arbitrage():
    # very steep wings with tiny curvature at the money create a negative density
    assert not svi_butterfly_ok((-0.05, 1.5, -0.2, 0.0, 0.01))


def test_convexity_check_flags_bad_quote():
    otm, F, D, _ = expiry_ivs(_chain(), spot=5000.0, rate=0.0)
    i = otm.index[len(otm) // 2]
    otm.loc[i, "mid"] *= 1.5                      # one mispriced quote breaks convexity
    assert arbitrage_checks(otm, F, D)[1] > 0
