"""
Rolling OLS decomposition:  r_i,t = α_i + β_i,t · r_sector,t + ε_i,t

β_i,t is allowed to vary over time — critical for semis because sector
betas shift with the cycle (NVDA's beta to SMH was ~1.2 in 2021 and ~1.8
during the 2023-24 AI surge). A static beta would mis-attribute sector
momentum as idiosyncratic alpha.

ε_i,t (the residual) is the modeling target: the portion of each stock's
return not explained by where the sector went.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger


def rolling_ols_decompose(
    stock_returns: pd.DataFrame,
    sector_returns: pd.Series,
    window: int = 60,
    min_periods: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each stock and each day, run OLS on the trailing `window` days:
        r_i = a + b * r_sector

    Returns:
        betas     : (T, N) time-varying sector beta
        alphas    : (T, N) OLS intercept (persistent daily drift)
        residuals : (T, N) idiosyncratic return  ε_i,t = r_i,t - (a + b*r_sector,t)
    """
    stocks = stock_returns.columns.tolist()
    idx    = stock_returns.index
    T      = len(idx)
    sector = sector_returns.reindex(idx).values

    betas     = pd.DataFrame(np.nan, index=idx, columns=stocks, dtype=np.float64)
    alphas    = pd.DataFrame(np.nan, index=idx, columns=stocks, dtype=np.float64)
    residuals = pd.DataFrame(np.nan, index=idx, columns=stocks, dtype=np.float64)

    for ticker in stocks:
        y = stock_returns[ticker].values
        for t in range(min_periods, T):
            lo = max(0, t - window)
            ys = y[lo:t]
            xs = sector[lo:t]
            if np.std(xs) < 1e-10:
                continue
            b, a = np.polyfit(xs, ys, 1)
            betas.at[idx[t], ticker]     = b
            alphas.at[idx[t], ticker]    = a
            residuals.at[idx[t], ticker] = y[t] - (a + b * sector[t])

    logger.info(
        f"OLS decomposition complete | "
        f"mean β={betas.mean().mean():.2f} | "
        f"idio σ={residuals.std().mean():.4f}"
    )
    return betas, alphas, residuals


def forward_idiosyncratic_return(residuals: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    Sum of idiosyncratic returns over the next `horizon` days.
    This is the supervised learning target: predict this at time t using
    features known at time t.

    The shift(-horizon) aligns each row with the observation date, not the
    target date — so features at row t predict the return from t to t+horizon.
    """
    return residuals.rolling(horizon).sum().shift(-horizon)
