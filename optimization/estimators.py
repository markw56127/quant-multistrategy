"""
Non-linear optimization layer:
  1. MLEEstimator    – MLE for PDE drift/diffusion parameters via L-BFGS
  2. LagrangianOptimizer – Constrained portfolio optimization via Lagrangian relaxation
     Maximize: E[r] - λ·VaR  s.t. Σ|w_i| ≤ 1, |w_i| ≤ max_pos
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize as opt
import torch
import torch.nn as nn
from loguru import logger
from scipy.stats import multivariate_normal


class MLEEstimator:
    """
    Maximum Likelihood Estimation for parametric PDE components.

    Given observed latent factor transitions (z_t → z_{t+1}), estimates
    the drift μ and diffusion σ² parameters of the Fokker-Planck SDE:
        dz = μ(z)dt + σ(z)dW

    Uses the Euler-Maruyama likelihood:
        p(z_{t+1} | z_t) = N(z_t + μ(z_t)·dt, σ²(z_t)·dt·I)
    """

    def __init__(self, dt: float = 0.01, method: str = "L-BFGS-B"):
        self.dt = dt
        self.method = method
        self.result_: Optional[opt.OptimizeResult] = None
        self.mu_params_: Optional[np.ndarray] = None
        self.sigma_params_: Optional[np.ndarray] = None

    def negative_log_likelihood(
        self,
        params: np.ndarray,
        z_t: np.ndarray,
        z_t1: np.ndarray,
        n_mu_params: int,
    ) -> float:
        """
        NLL of the Euler-Maruyama transition density.

        params[:n_mu_params]  = polynomial coefficients for μ
        params[n_mu_params:]  = log-diffusion coefficients
        """
        mu_params    = params[:n_mu_params]
        sigma_params = np.exp(params[n_mu_params:])  # exp to ensure positivity

        # Linear drift: μ(z) = A·z + b  (A: d×d, b: d)
        d = z_t.shape[1]
        A = mu_params[:d * d].reshape(d, d)
        b = mu_params[d * d: d * d + d]
        mu = z_t @ A.T + b  # (N, d)

        # Diagonal diffusion
        sigma2 = np.clip(sigma_params[:d], 1e-6, None)

        # Gaussian NLL
        diff    = z_t1 - (z_t + mu * self.dt)  # (N, d)
        var     = sigma2 * self.dt
        nll     = 0.5 * np.sum(np.log(2 * np.pi * var)) + 0.5 * np.sum(diff ** 2 / var, axis=1)
        return float(nll.mean())

    def fit(
        self,
        latent_trajectories: np.ndarray,
        n_iterations: int = 1000,
    ) -> "MLEEstimator":
        """
        Fit MLE parameters to observed latent factor trajectories.

        Args:
            latent_trajectories: (T, d) array of latent factor time series
        """
        z_t  = latent_trajectories[:-1]
        z_t1 = latent_trajectories[1:]
        d    = z_t.shape[1]

        n_mu_params    = d * d + d  # A matrix + bias
        n_sig_params   = d
        n_params       = n_mu_params + n_sig_params

        x0 = np.zeros(n_params)
        x0[d * d: d * d + d] = 0.0      # zero bias
        x0[n_mu_params:]     = np.log(0.05)  # initial log-sigma

        logger.info(f"MLE fitting: {n_params} parameters, {len(z_t)} transitions")

        result = opt.minimize(
            self.negative_log_likelihood,
            x0,
            args=(z_t, z_t1, n_mu_params),
            method=self.method,
            options={"maxiter": n_iterations, "ftol": 1e-9},
        )

        self.result_       = result
        self.mu_params_    = result.x[:n_mu_params]
        self.sigma_params_ = np.exp(result.x[n_mu_params:])

        logger.info(f"MLE converged: {result.success}, NLL={result.fun:.4f}")
        return self

    def predict_drift(self, z: np.ndarray) -> np.ndarray:
        """μ(z) using fitted linear parameters."""
        if self.mu_params_ is None:
            raise RuntimeError("Estimator not fitted.")
        d = z.shape[-1] if z.ndim > 1 else int((-1 + (1 + 4 * len(self.mu_params_)) ** 0.5) / 2)
        A = self.mu_params_[:d * d].reshape(d, d)
        b = self.mu_params_[d * d: d * d + d]
        return z @ A.T + b

    def predict_diffusion(self) -> np.ndarray:
        """σ²(z) – constant diagonal diffusion matrix."""
        if self.sigma_params_ is None:
            raise RuntimeError("Estimator not fitted.")
        return self.sigma_params_


class LagrangianOptimizer:
    """
    Constrained portfolio optimization using Lagrangian relaxation.

    Objective:
        max  E[r_p] - λ·VaR_α(r_p)
        s.t. Σ|w_i| ≤ 1  (capital constraint)
             |w_i| ≤ max_pos  (position limit)
             Σ w_i = 0  (market-neutral, optional)

    Solves the dual: L(w, λ) = -E[r_p] + λ·VaR + penalty·constraint_violations
    """

    def __init__(
        self,
        max_position_size: float = 0.10,
        risk_aversion: float = 1.0,
        var_confidence: float = 0.95,
        market_neutral: bool = False,
        max_iter: int = 1000,
    ):
        self.max_pos       = max_position_size
        self.risk_aversion = risk_aversion
        self.var_conf      = var_confidence
        self.market_neutral = market_neutral
        self.max_iter      = max_iter

        self._last_weights: Optional[np.ndarray] = None
        self._last_lagrange: Optional[Dict] = None

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        pinn_drift: Optional[np.ndarray] = None,
        n_assets: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns optimal portfolio weights.

        Args:
            expected_returns: (n_assets,) expected return per asset
            cov_matrix:       (n_assets, n_assets) covariance matrix
            pinn_drift:       (n_assets,) PINN-derived drift signal (optional boost)
        """
        n = len(expected_returns)
        # pinn_drift is in latent space (n_latent dims), not asset space (n_assets dims),
        # so it cannot be directly added to expected_returns. The PINN signal enters the
        # portfolio decision through the RL policy's state observation instead.

        def objective(w: np.ndarray) -> float:
            port_ret = w @ expected_returns
            port_var = w @ cov_matrix @ w
            port_std = np.sqrt(np.clip(port_var, 0, None))
            # VaR approximation: normal quantile
            z_alpha  = 1.645  # 95th percentile
            var_est  = -port_ret + z_alpha * port_std
            return -port_ret + self.risk_aversion * var_est

        def grad_objective(w: np.ndarray) -> np.ndarray:
            port_var = w @ cov_matrix @ w
            port_std = np.sqrt(np.clip(port_var, 1e-10, None))
            z_alpha  = 1.645
            d_ret    = -expected_returns
            d_var    = self.risk_aversion * z_alpha * (cov_matrix @ w) / port_std
            return d_ret + d_var

        constraints = [
            {"type": "ineq", "fun": lambda w: 1.0 - np.abs(w).sum()},
        ]
        if self.market_neutral:
            constraints.append({"type": "eq", "fun": lambda w: w.sum()})

        bounds = [(-self.max_pos, self.max_pos)] * n
        w0 = np.zeros(n)

        result = opt.minimize(
            objective,
            w0,
            jac=grad_objective,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": 1e-10},
        )

        w_opt = result.x
        self._last_weights = w_opt
        self._last_lagrange = {
            "success":      result.success,
            "objective":    result.fun,
            "port_return":  w_opt @ expected_returns,
            "port_vol":     np.sqrt(w_opt @ cov_matrix @ w_opt),
        }
        return w_opt

    def rolling_optimize(
        self,
        returns_df: "pd.DataFrame",
        pinn_drifts: Optional["pd.DataFrame"] = None,
        lookback: int = 60,
        rebalance_every: int = 5,
    ) -> "pd.DataFrame":
        """
        Run the optimizer over a rolling window and return a weight matrix.
        """
        import pandas as pd
        tickers = list(returns_df.columns)
        weights_list = []
        dates_list   = []

        for i in range(lookback, len(returns_df), rebalance_every):
            window  = returns_df.iloc[i - lookback: i]
            mu_hat  = window.mean().values * 252
            cov_hat = window.cov().values * 252
            drift   = pinn_drifts.iloc[i].values if pinn_drifts is not None else None

            try:
                w = self.optimize(mu_hat, cov_hat, drift)
            except Exception as e:
                logger.warning(f"Optimization failed at step {i}: {e}")
                w = np.zeros(len(tickers))

            weights_list.append(w)
            dates_list.append(returns_df.index[i])

        return pd.DataFrame(weights_list, index=dates_list, columns=tickers)
