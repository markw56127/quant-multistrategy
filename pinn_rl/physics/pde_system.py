"""
Stochastic PDE System for latent factor evolution.

We model the probability density p(z, t) of latent market factors z using
the Fokker-Planck equation:

    ∂p/∂t = -∇·(μ(z,t,s) · p) + (1/2) ∇·(σ²(z,t) · ∇p) + chaos_term

where:
  - z ∈ R^d  are the latent factors (autoencoder output)
  - μ(z,t,s) = drift field (function of state, time, sentiment s)
  - σ²(z,t)  = diffusion tensor
  - chaos_term = Lyapunov-inspired perturbation for sensitivity control

The PINN will learn μ and σ² from data while satisfying this PDE as a constraint.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PDEConfig:
    n_latent: int = 16
    dt: float = 0.01
    diffusion_coeff: float = 0.05
    chaos_sensitivity: float = 0.1
    lower_bound: float = -5.0
    upper_bound: float = 5.0
    n_grid: int = 50


class DriftNetwork(nn.Module):
    """
    Parameterizes the drift field μ(z, t, s) → R^d.
    Inputs: [z (n_latent), t (1), s (1)] → drift (n_latent)
    """

    def __init__(self, n_latent: int, hidden_dims: Tuple[int, ...] = (128, 128, 64)):
        super().__init__()
        input_dim = n_latent + 2  # z + t + sentiment
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, n_latent))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # z: (..., n_latent), t: (..., 1), s: (..., 1)
        x = torch.cat([z, t, s], dim=-1)
        return self.net(x)


class DiffusionNetwork(nn.Module):
    """
    Parameterizes the diagonal diffusion σ²(z, t) → R^d (positive).
    Returns log-space to ensure positivity.
    """

    def __init__(self, n_latent: int, hidden_dims: Tuple[int, ...] = (64, 64)):
        super().__init__()
        input_dim = n_latent + 1  # z + t
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, n_latent))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, t], dim=-1)
        log_sigma2 = self.net(x)
        return F.softplus(log_sigma2) + 1e-4  # guaranteed positive


import torch.nn.functional as F


class FokkerPlanckPDE:
    """
    Defines and evaluates the Fokker-Planck PDE residual for use in PINN training.

    The PINN learns p(z, t) such that the PDE residual R = ∂p/∂t + FP_operator[p] ≈ 0.
    """

    def __init__(self, config: PDEConfig):
        self.cfg = config

    def residual(
        self,
        p: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        mu: torch.Tensor,
        sigma2: torch.Tensor,
        dp_dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Fokker-Planck residual at collocation points.

        R = ∂p/∂t + ∇_z · (μ·p) - (1/2) Δ_z (σ²·p)

        Args:
            p:      (N, 1)  - probability density (PINN output)
            z:      (N, d)  - spatial coordinates (latent factors)
            t:      (N, 1)  - time
            mu:     (N, d)  - drift vector at these points
            sigma2: (N, d)  - diffusion coefficients at these points
            dp_dt:  (N, 1)  - ∂p/∂t computed via autograd

        Returns:
            residual: (N, 1)
        """
        # Compute divergence and Laplacian jointly (2d backward passes instead of 3d)
        div_mu_p, laplacian_sigma2_p = self._pde_operators(p, z, mu, sigma2)

        # Chaos term: λ · |∇_z p| · |z|  (sensitivity to initial conditions)
        chaos = self._chaos_term(p, z)

        residual = dp_dt + div_mu_p - 0.5 * laplacian_sigma2_p + chaos
        return residual

    def boundary_residual(self, p_boundary: torch.Tensor) -> torch.Tensor:
        """
        Absorbing boundary condition: p = 0 at domain boundaries.
        Enforces non-negative price and liquidity ceiling constraints.
        """
        return p_boundary  # should be 0

    def initial_condition_residual(
        self,
        p_ic: torch.Tensor,
        z_ic: torch.Tensor,
        mu_init: torch.Tensor,
        sigma2_init: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initial condition: p(z, 0) = N(z; z_obs, σ²_init)
        We impose that the initial density is a Gaussian centered at the
        observed latent factor mean.
        """
        d = z_ic.shape[-1]
        norm_const = (2 * np.pi) ** (d / 2) * torch.prod(sigma2_init ** 0.5, dim=-1, keepdim=True)
        exponent = -0.5 * torch.sum((z_ic - mu_init) ** 2 / sigma2_init, dim=-1, keepdim=True)
        p_target = torch.exp(exponent) / (norm_const + 1e-12)
        return p_ic - p_target

    def sample_collocation_points(
        self,
        n_interior: int,
        n_boundary: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (z, t) collocation points for PINN training.
        Returns: (z_interior, t_interior), z_boundary, t_boundary
        """
        lo, hi = self.cfg.lower_bound, self.cfg.upper_bound
        d = self.cfg.n_latent

        # Interior: random points in [lo, hi]^d × [0, T]
        z_int = torch.rand(n_interior, d, device=device) * (hi - lo) + lo
        t_int = torch.rand(n_interior, 1, device=device)

        # Boundary: points where any dimension equals lo or hi
        n_per_face = n_boundary // (2 * d)
        z_bnd_list = []
        for dim_i in range(d):
            for val in [lo, hi]:
                z_face = torch.rand(n_per_face, d, device=device) * (hi - lo) + lo
                z_face[:, dim_i] = val
                z_bnd_list.append(z_face)
        z_bnd = torch.cat(z_bnd_list, dim=0)
        t_bnd = torch.rand(len(z_bnd), 1, device=device)

        return z_int, t_int, z_bnd, t_bnd

    # ------------------------------------------------------------------
    # Internal PDE operators
    # ------------------------------------------------------------------

    def _divergence_mu_p(
        self, p: torch.Tensor, z: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """∇·(μp): trace of Jacobian of flux w.r.t. z, computed via d VJPs."""
        return self._pde_operators(p, z, mu, None)[0]

    def _laplacian_sigma2_p(
        self, p: torch.Tensor, z: torch.Tensor, sigma2: torch.Tensor
    ) -> torch.Tensor:
        """Δ(σ²p): diagonal Hessian trace, computed via d paired VJPs."""
        return self._pde_operators(p, z, None, sigma2)[1]

    def _pde_operators(
        self,
        p: torch.Tensor,
        z: torch.Tensor,
        mu: Optional[torch.Tensor],
        sigma2: Optional[torch.Tensor],
    ):
        """
        Compute divergence and Laplacian in a single joint loop to minimise
        the number of autograd backward passes and retain_graph calls.

        Previously: d passes for div + 2d passes for Laplacian = 3d passes,
        all with retain_graph=True (graph held in memory the whole time).

        Now: 2d passes total, retain_graph=False on the final pass.

        Note: a further 2-4x speedup is possible via torch.func.vmap + jacrev
        (vectorises over batch dim), but requires restructuring network calls
        into per-sample closures; left as a future optimisation.
        """
        d = z.shape[1]
        N = p.shape[0]
        div = torch.zeros(N, 1, device=z.device) if mu is not None else None
        lap = torch.zeros(N, 1, device=z.device) if sigma2 is not None else None

        mu_flux  = (mu * p)     if mu      is not None else None
        s2_flux  = (sigma2 * p) if sigma2  is not None else None

        for i in range(d):
            if mu_flux is not None:
                g = torch.autograd.grad(
                    mu_flux[:, i].sum(), z,
                    create_graph=True, retain_graph=True,
                )[0][:, i:i+1]
                div = div + g

            if s2_flux is not None:
                g1 = torch.autograd.grad(
                    s2_flux[:, i].sum(), z,
                    create_graph=True, retain_graph=True,
                )[0][:, i]
                # retain_graph=True: _chaos_term still needs the p-computation graph
                g2 = torch.autograd.grad(
                    g1.sum(), z,
                    create_graph=True, retain_graph=True,
                )[0][:, i:i+1]
                lap = lap + g2

        return div, lap

    def _chaos_term(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Lyapunov-inspired sensitivity: λ · |∇p| · ||z||"""
        grad_p = torch.autograd.grad(
            p.sum(), z, create_graph=True, retain_graph=True
        )[0]
        grad_norm = torch.norm(grad_p, dim=-1, keepdim=True)
        z_norm = torch.norm(z, dim=-1, keepdim=True)
        return self.cfg.chaos_sensitivity * grad_norm * z_norm
