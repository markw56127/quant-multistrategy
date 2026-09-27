"""
Strategies for the paper trader. A strategy maps an as-of date to target weights (summing
to at most 1; the remainder is cash), using only data through the previous close.

Nothing here has passed an alpha bar. These are **premia baselines** from
premia_portfolio (Track A). Their paper-trading purpose is to validate the pipeline and
measure implementation shortfall, not to make a return claim.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "premia_portfolio"))


def sixty_forty(as_of: pd.Timestamp) -> dict:
    return {"SPY": 0.6, "IEF": 0.4}


def risk_parity_vt(as_of: pd.Timestamp, vol_target: float = 0.10) -> dict:
    """Track A's risk parity on the 9 ETFs, with the daily-1y covariance, scaled to a 10%
    annual vol target and capped at 1.0 (no leverage in paper)."""
    from run import ASSETS, w_erc
    px = yf.download(ASSETS, start=as_of - pd.Timedelta(days=400), end=as_of, auto_adjust=True,
                     progress=False)["Close"][ASSETS].dropna()
    D = px.pct_change().dropna().iloc[-252:].values
    S = np.cov(D, rowvar=False) * 21
    w = w_erc(S)
    scale = min(1.0, (vol_target / np.sqrt(12)) / np.sqrt(w @ S @ w))
    return {a: float(x * scale) for a, x in zip(ASSETS, w)}


STRATEGIES = {"sixty_forty": sixty_forty, "risk_parity_vt": risk_parity_vt}
