"""
Physics-Informed Neural Network (PINN) solver for the Fokker-Planck PDE.

The PINN simultaneously:
  1. Fits observed latent factor trajectories (data loss)
  2. Satisfies the Fokker-Planck PDE at collocation points (physics loss)
  3. Satisfies boundary + initial conditions (constraint losses)

The trained PINN provides:
  - p(z, t): probability density over latent factors → used as MDP transition
  - μ(z, t, s): drift → directional market signal
  - σ²(z, t): diffusion → uncertainty / volatility estimate
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .pde_system import DiffusionNetwork, DriftNetwork, FokkerPlanckPDE, PDEConfig


class PINNDensityNet(nn.Module):
    """
    Approximates p(z, t) = probability density.
    Input:  [z (n_latent), t (1)]
    Output: p (1), strictly positive via softplus
    """

    def __init__(self, n_latent: int, hidden_dims: List[int], activation: str = "tanh"):
        super().__init__()
        act_fn = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}.get(activation, nn.Tanh)
        layers = []
        prev = n_latent + 1
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act_fn()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, t], dim=-1)
        return F.softplus(self.net(x))  # strictly positive density


class PINNSolver:
    """
    Full PINN system: density net + drift net + diffusion net, trained jointly.

    After training, the PINN exports:
      - Transition distributions for the Bayesian MDP
      - Uncertainty estimates for the RL state
    """

    def __init__(
        self,
        config: PDEConfig,
        hidden_dims: List[int] = (256, 256, 128, 128, 64),
        activation: str = "tanh",
        lambda_physics: float = 1.0,
        lambda_bc: float = 0.5,
        lambda_ic: float = 0.5,
        learning_rate: float = 1e-3,
        adaptive_weights: bool = True,
        adaptive_update_freq: int = 100,
        adaptive_momentum: float = 0.9,
        device: Optional[str] = None,
    ):
        self.cfg = config
        self.lambda_physics = lambda_physics
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        # Adaptive loss weight annealing (Wang et al. 2021, LRA algorithm).
        # Rescales each λ so that its gradient contribution matches the data loss.
        self.adaptive_weights     = adaptive_weights
        self.adaptive_update_freq = adaptive_update_freq
        self.adaptive_momentum    = adaptive_momentum
        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else
                "mps"  if torch.backends.mps.is_available() else
                "cpu"
            )
        )

        self.pde = FokkerPlanckPDE(config)
        self.density_net = PINNDensityNet(config.n_latent, list(hidden_dims), activation).to(self.device)
        self.drift_net   = DriftNetwork(config.n_latent, hidden_dims=(128, 128, 64)).to(self.device)
        self.diff_net    = DiffusionNetwork(config.n_latent, hidden_dims=(64, 64)).to(self.device)

        all_params = (
            list(self.density_net.parameters())
            + list(self.drift_net.parameters())
            + list(self.diff_net.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5
        )

        self._n_collocation = config.n_grid * 2
        self._n_boundary    = config.n_grid
        self._z0_mean: Optional[torch.Tensor] = None   # initial condition anchor
        self._loss_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        latent_trajectories: np.ndarray,
        sentiment_series: Optional[np.ndarray] = None,
        epochs: int = 5000,
        n_collocation: int = 2048,
        n_boundary: int = 512,
        patience: int = 500,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Train the PINN.

        Args:
            latent_trajectories: (T, n_latent) observed latent factor sequence
            sentiment_series:    (T,) optional sentiment exogenous input
            epochs:              training iterations
            n_collocation:       interior collocation points per iteration
            n_boundary:          boundary collocation points per iteration
        """
        self._n_collocation = n_collocation
        self._n_boundary    = n_boundary

        # Store initial condition
        z0 = torch.FloatTensor(latent_trajectories[:10].mean(axis=0)).to(self.device)
        self._z0_mean = z0.unsqueeze(0)

        # Build observed (z, t) data pairs
        T = len(latent_trajectories)
        t_obs  = torch.FloatTensor(np.linspace(0, 1, T)).unsqueeze(1).to(self.device)
        z_obs  = torch.FloatTensor(latent_trajectories).to(self.device)
        if sentiment_series is not None:
            s_obs  = torch.FloatTensor(sentiment_series.copy()).unsqueeze(1).to(self.device)
        else:
            s_obs = torch.zeros(T, 1, device=self.device)

        best_loss = float("inf")
        patience_count = 0

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()

            # --- Data loss: density should be high at observed latent positions ---
            t_obs_req = t_obs.requires_grad_(True)
            z_obs_req = z_obs.requires_grad_(True)
            p_obs = self.density_net(z_obs_req, t_obs_req)
            # Maximize density at observed points → minimize -log p
            data_loss = -torch.log(p_obs + 1e-8).mean()

            # --- Physics loss at collocation points ---
            z_int, t_int, z_bnd, t_bnd = self.pde.sample_collocation_points(
                n_collocation, n_boundary, self.device
            )
            z_int = z_int.requires_grad_(True)
            t_int = t_int.requires_grad_(True)

            # Interpolate sentiment to collocation time points via linear interp
            if sentiment_series is not None:
                t_idx = (t_int.detach().cpu().numpy().flatten() * (T - 1)).astype(int).clip(0, T - 1)
                s_int = torch.FloatTensor(sentiment_series[t_idx]).unsqueeze(1).to(self.device)
            else:
                s_int = torch.zeros(n_collocation, 1, device=self.device)

            p_int  = self.density_net(z_int, t_int)
            mu_int = self.drift_net(z_int, t_int, s_int)
            s2_int = self.diff_net(z_int, t_int)

            dp_dt = torch.autograd.grad(
                p_int.sum(), t_int, create_graph=True, retain_graph=True
            )[0]

            pde_res = self.pde.residual(p_int, z_int, t_int, mu_int, s2_int, dp_dt)
            physics_loss = (pde_res ** 2).mean()

            # --- Boundary loss ---
            p_bnd = self.density_net(z_bnd.requires_grad_(True), t_bnd.requires_grad_(True))
            bc_res = self.pde.boundary_residual(p_bnd)
            bc_loss = (bc_res ** 2).mean()

            # --- Initial condition loss (Gaussian IC) ---
            n_ic = min(32, T)
            z_ic = z_obs_req[:n_ic]
            t_ic = torch.zeros(n_ic, 1, device=self.device)
            p_ic = self.density_net(z_ic.requires_grad_(True), t_ic.requires_grad_(True))
            mu_ic = self.drift_net(z_ic, t_ic, s_obs[:n_ic])
            s2_ic = self.diff_net(z_ic, t_ic)
            ic_res = self.pde.initial_condition_residual(p_ic, z_ic, mu_ic.detach(), s2_ic.detach())
            ic_loss = (ic_res ** 2).mean()

            # --- Adaptive loss weight annealing (LRA, Wang et al. 2021) ---
            # Every `adaptive_update_freq` epochs, recompute λ so each loss term
            # contributes gradient magnitudes comparable to the data loss.
            if self.adaptive_weights and epoch % self.adaptive_update_freq == 0 and epoch > 1:
                self._update_adaptive_weights(
                    data_loss, physics_loss, bc_loss, ic_loss
                )

            total_loss = (
                data_loss
                + self.lambda_physics * physics_loss
                + self.lambda_bc * bc_loss
                + self.lambda_ic * ic_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.density_net.parameters())
                + list(self.drift_net.parameters())
                + list(self.diff_net.parameters()),
                1.0,
            )
            self.optimizer.step()
            self.scheduler.step()

            loss_dict = {
                "total":          total_loss.item(),
                "data":           data_loss.item(),
                "physics":        physics_loss.item(),
                "bc":             bc_loss.item(),
                "ic":             ic_loss.item(),
                "lambda_physics": self.lambda_physics,
                "lambda_bc":      self.lambda_bc,
                "lambda_ic":      self.lambda_ic,
            }
            self._loss_history.append(loss_dict)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_count = 0
            else:
                patience_count += 1

            if verbose and epoch % 500 == 0:
                logger.info(
                    f"PINN Epoch {epoch}/{epochs} | "
                    f"total={loss_dict['total']:.4f} | "
                    f"data={loss_dict['data']:.4f} | "
                    f"phys={loss_dict['physics']:.4f} | "
                    f"bc={loss_dict['bc']:.4f} | "
                    f"ic={loss_dict['ic']:.4f} | "
                    f"λ_phys={self.lambda_physics:.3f} λ_bc={self.lambda_bc:.3f} λ_ic={self.lambda_ic:.3f}"
                )

            if patience_count >= patience:
                logger.info(f"PINN early stopping at epoch {epoch}")
                break

        return self._loss_history

    def _update_adaptive_weights(
        self,
        data_loss: torch.Tensor,
        physics_loss: torch.Tensor,
        bc_loss: torch.Tensor,
        ic_loss: torch.Tensor,
    ):
        """
        Learning Rate Annealing (LRA) for PINN loss weights.
        (Wang, Teng & Perdikaris, 2021 — "Understanding and mitigating gradient pathologies")

        Target: λ_i = max|∇_θ L_data| / mean|∇_θ (λ_i L_i)|
        i.e., scale each auxiliary loss so its gradient norm matches the data loss.

        Uses exponential moving average of λ to smooth updates.
        """
        all_params = (
            list(self.density_net.parameters())
            + list(self.drift_net.parameters())
            + list(self.diff_net.parameters())
        )

        def _grad_norm(loss: torch.Tensor) -> float:
            grads = torch.autograd.grad(
                loss, all_params, retain_graph=True, allow_unused=True
            )
            norms = [g.abs().mean().item() for g in grads if g is not None]
            return float(np.mean(norms)) if norms else 1e-8

        try:
            norm_data    = _grad_norm(data_loss)
            norm_physics = _grad_norm(physics_loss)
            norm_bc      = _grad_norm(bc_loss)
            norm_ic      = _grad_norm(ic_loss)

            # Target λ = norm_data / norm_auxiliary (clipped for stability)
            lam_phys_target = float(np.clip(norm_data / (norm_physics + 1e-8), 0.01, 100.0))
            lam_bc_target   = float(np.clip(norm_data / (norm_bc   + 1e-8), 0.01, 100.0))
            lam_ic_target   = float(np.clip(norm_data / (norm_ic   + 1e-8), 0.01, 100.0))

            # EMA update
            m = self.adaptive_momentum
            self.lambda_physics = m * self.lambda_physics + (1 - m) * lam_phys_target
            self.lambda_bc      = m * self.lambda_bc      + (1 - m) * lam_bc_target
            self.lambda_ic      = m * self.lambda_ic      + (1 - m) * lam_ic_target

        except Exception as e:
            # Gracefully skip update if graph is unavailable
            pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_density(self, z: np.ndarray, t: float) -> np.ndarray:
        """p(z, t) for an array of latent positions."""
        self.density_net.eval()
        z_t = torch.FloatTensor(z).to(self.device)
        t_t = torch.full((len(z), 1), t, device=self.device)
        return self.density_net(z_t, t_t).cpu().numpy().flatten()

    @torch.no_grad()
    def predict_drift(self, z: np.ndarray, t: float, sentiment: float = 0.0) -> np.ndarray:
        """μ(z, t, s) drift vector for an array of latent positions."""
        self.drift_net.eval()
        z_t = torch.FloatTensor(z).to(self.device)
        t_t = torch.full((len(z), 1), t, device=self.device)
        s_t = torch.full((len(z), 1), sentiment, device=self.device)
        return self.drift_net(z_t, t_t, s_t).cpu().numpy()

    @torch.no_grad()
    def predict_diffusion(self, z: np.ndarray, t: float) -> np.ndarray:
        """σ²(z, t) diffusion tensor for an array of latent positions."""
        self.diff_net.eval()
        z_t = torch.FloatTensor(z).to(self.device)
        t_t = torch.full((len(z), 1), t, device=self.device)
        return self.diff_net(z_t, t_t).cpu().numpy()

    def step_latent(
        self,
        z_current: np.ndarray,
        t: float,
        sentiment: float = 0.0,
        dt: Optional[float] = None,
        n_paths: int = 1,
    ) -> np.ndarray:
        """
        Euler-Maruyama step: z_{t+dt} = z_t + μ·dt + σ·√dt·ε
        Returns (n_paths, n_latent) simulated next states.
        """
        dt = dt or self.cfg.dt
        z_rep = np.tile(z_current.reshape(1, -1), (n_paths, 1))
        mu    = self.predict_drift(z_rep, t, sentiment)
        sigma2 = self.predict_diffusion(z_rep, t)
        noise = np.random.randn(*z_rep.shape) * np.sqrt(sigma2 * dt)
        return z_rep + mu * dt + noise

    def get_transition_distribution(
        self,
        z_current: np.ndarray,
        t: float,
        sentiment: float = 0.0,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean_next, cov_next) of the transition distribution p(z_{t+1}|z_t).
        Used as the Bayesian MDP transition function.
        """
        paths = self.step_latent(z_current, t, sentiment, n_paths=n_samples)
        return paths.mean(axis=0), np.cov(paths.T) if n_samples > 1 else np.diag(self.predict_diffusion(z_current.reshape(1, -1), t).flatten() * self.cfg.dt)

    def save(self, path: str):
        torch.save({
            "density": self.density_net.state_dict(),
            "drift":   self.drift_net.state_dict(),
            "diff":    self.diff_net.state_dict(),
            "config":  self.cfg,
        }, path)
        logger.info(f"PINN saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.density_net.load_state_dict(ckpt["density"])
        self.drift_net.load_state_dict(ckpt["drift"])
        self.diff_net.load_state_dict(ckpt["diff"])
        logger.info(f"PINN loaded from {path}")
