"""
Monte Carlo simulation engine for uncertainty quantification.

Supports:
  1. Standard geometric Brownian motion simulations
  2. PINN-driven simulations (using learned drift + diffusion)
  3. Scenario analysis (bullish / bearish / stress)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class MonteCarloSimulator:
    """
    Runs Monte Carlo simulations to quantify uncertainty in portfolio trajectories.
    """

    def __init__(
        self,
        n_simulations: int = 50_000,
        horizon_days: int = 20,
        seed: Optional[int] = 42,
    ):
        self.n_sims    = n_simulations
        self.horizon   = horizon_days
        self.rng       = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # GBM simulation
    # ------------------------------------------------------------------

    def simulate_gbm(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        lookback: int = 252,
    ) -> np.ndarray:
        """
        Simulate portfolio returns using GBM with historical mu/sigma.

        Returns: (n_sims, horizon) portfolio return paths
        """
        recent = returns.iloc[-lookback:]
        mu_daily  = (recent @ weights).mean()
        cov_daily = recent.cov().values
        port_vol  = np.sqrt(weights @ cov_daily @ weights)

        # Correlated asset simulation → portfolio aggregation
        asset_paths = self._simulate_correlated(
            mu=recent.mean().values,
            cov=cov_daily,
            n_sims=self.n_sims,
            horizon=self.horizon,
        )  # (n_sims, horizon, n_assets)

        port_paths = (asset_paths * weights).sum(axis=-1)  # (n_sims, horizon)
        return port_paths

    def simulate_pinn_driven(
        self,
        pinn_solver,
        vae_trainer,
        weights: np.ndarray,
        z_current: np.ndarray,
        t_current: float,
        sentiment: float = 0.0,
        n_assets: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate paths using the PINN-learned latent factor dynamics.
        Decodes latent factor paths back to return space.

        Returns: (n_sims, horizon) portfolio return paths
        """
        n_assets = n_assets or len(weights)
        port_paths = np.zeros((self.n_sims, self.horizon))

        for sim in range(self.n_sims):
            z = z_current.copy()
            for h in range(self.horizon):
                t = t_current + h / 252.0
                z_next = pinn_solver.step_latent(z, t, sentiment, n_paths=1)[0]
                # Approximate return from latent delta
                delta_z    = z_next - z
                asset_rets = vae_trainer.model.decode(
                    __import__("torch").FloatTensor(z_next).unsqueeze(0)
                ).detach().numpy().flatten()
                port_paths[sim, h] = float((weights[:n_assets] * asset_rets[:n_assets]).sum())
                z = z_next

        return port_paths

    def simulate_scenarios(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        scenarios: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run named stress scenarios by shifting mu and scaling sigma.
        Default scenarios: base, bullish, bearish, crisis.

        Returns: {scenario_name: (n_sims, horizon) paths}
        """
        if scenarios is None:
            scenarios = {
                "base":    {"mu_shift": 0.0,    "vol_scale": 1.0},
                "bullish": {"mu_shift": +0.001, "vol_scale": 0.8},
                "bearish": {"mu_shift": -0.001, "vol_scale": 1.2},
                "crisis":  {"mu_shift": -0.003, "vol_scale": 2.5},
            }

        recent    = returns.iloc[-252:]
        mu_daily  = recent.mean().values
        cov_daily = recent.cov().values
        results   = {}

        for name, params in scenarios.items():
            mu_adj  = mu_daily + params["mu_shift"]
            cov_adj = cov_daily * params["vol_scale"] ** 2
            paths   = self._simulate_correlated(mu_adj, cov_adj, self.n_sims, self.horizon)
            port    = (paths * weights).sum(axis=-1)
            results[name] = port

        return results

    # ------------------------------------------------------------------
    # Path analytics
    # ------------------------------------------------------------------

    def path_statistics(self, paths: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics over (n_sims, horizon) cumulative return paths.
        """
        cum_rets = paths.sum(axis=1)  # total path return
        return {
            "mean":         float(cum_rets.mean()),
            "std":          float(cum_rets.std()),
            "median":       float(np.median(cum_rets)),
            "p5":           float(np.percentile(cum_rets, 5)),
            "p1":           float(np.percentile(cum_rets, 1)),
            "p95":          float(np.percentile(cum_rets, 95)),
            "prob_positive":float((cum_rets > 0).mean()),
            "expected_shortfall_95": float(cum_rets[cum_rets <= np.percentile(cum_rets, 5)].mean()),
        }

    def confidence_bands(
        self,
        paths: np.ndarray,
        levels: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
    ) -> Dict[str, np.ndarray]:
        """
        Returns per-step quantile bands for visualization.
        Each value is a (horizon,) array.
        """
        cum_paths = np.cumsum(paths, axis=1)  # (n_sims, horizon)
        bands = {}
        for q in levels:
            bands[f"q{int(q*100):02d}"] = np.percentile(cum_paths, q * 100, axis=0)
        return bands

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _simulate_correlated(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        n_sims: int,
        horizon: int,
    ) -> np.ndarray:
        """
        Simulate (n_sims, horizon, n_assets) correlated normal returns.
        Uses Cholesky decomposition for correlation structure.
        """
        n_assets = len(mu)
        try:
            L = np.linalg.cholesky(cov + np.eye(n_assets) * 1e-8)
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(cov) + 1e-8))

        Z = self.rng.standard_normal((n_sims, horizon, n_assets))
        correlated = Z @ L.T + mu  # broadcast mu over (n_sims, horizon)
        return correlated
