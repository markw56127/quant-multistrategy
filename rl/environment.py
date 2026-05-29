"""
Bayesian MDP Trading Environment (Gymnasium-compatible).

State space:
  - Latent factors z from the autoencoder (n_latent dims)
  - PINN-derived uncertainty σ² (n_latent dims)
  - Sentiment scalar (1 dim)
  - Current portfolio weights (n_assets dims)
  - Time fraction t/T (1 dim)

Action space:
  - Target portfolio weights ∈ [-1, 1]^n_assets (negative = short)
  - Clipped to satisfy max_position_size constraint

Reward:
  - Differential Sharpe ratio (Moody & Saffell 2001) on the per-period
    cost-adjusted return. Per-step rewards sum to an estimate of the
    Sharpe ratio, so maximizing reward maximizes Sharpe directly.

Episode timing:
  - One env step = one rebalance period (e.g. 5 trading days).
  - Inside a period, weights are held fixed (matches backtest engine).
  - Transaction cost is paid once per rebalance, not per day.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger


class TradingEnvironment(gym.Env):
    """
    Bayesian MDP trading environment.

    The PINN-derived uncertainty enters the state directly so the agent can
    learn risk-aware behavior; the PINN penalty in the reward is opt-in
    (default off) since latent uncertainty in the observation is the more
    principled signal.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        returns: pd.DataFrame,
        latent_factors: pd.DataFrame,
        latent_uncertainty: Optional[np.ndarray] = None,
        sentiment: Optional[pd.Series] = None,
        pinn_solver=None,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.10,
        pinn_penalty: float = 0.0,
        dispersion_bonus: float = 1.0,
        initial_capital: float = 1_000_000.0,
        rebalance_freq: int = 5,
        dsr_eta: float = 0.01,
        reward_scale: float = 1.0,
        episode_length: Optional[int] = None,
        random_start: bool = False,
    ):
        super().__init__()
        assert len(returns) == len(latent_factors), "returns and latent_factors must have same length"

        self.returns = returns.values.astype(np.float32)
        self.latent  = latent_factors.values.astype(np.float32)
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)
        self.n_latent = self.latent.shape[1]
        self.T        = len(self.returns)

        self.latent_uncertainty = (
            latent_uncertainty.astype(np.float32)
            if latent_uncertainty is not None
            else np.ones_like(self.latent) * 0.1
        )
        self.sentiment = (
            sentiment.values.astype(np.float32)
            if sentiment is not None
            else np.zeros(self.T, dtype=np.float32)
        )

        self.pinn          = pinn_solver
        self.trans_cost    = transaction_cost
        self.slippage      = slippage
        self.max_pos       = max_position_size
        self.pinn_penalty    = pinn_penalty
        self.dispersion_bonus = dispersion_bonus
        self.initial_cap     = initial_capital
        self.rebalance_freq = max(1, int(rebalance_freq))
        self.dsr_eta       = dsr_eta
        self.reward_scale  = reward_scale
        self.episode_length = episode_length
        self.random_start   = random_start

        # State dimension: latent + uncertainty + sentiment + weights + t_frac
        self._state_dim = self.n_latent * 2 + 1 + self.n_assets + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self._step = 0
        self._weights = np.zeros(self.n_assets, dtype=np.float32)
        self._portfolio_value = initial_capital
        self._portfolio_history: List[float] = []
        self._weight_history: List[np.ndarray] = []

        # Differential Sharpe running stats (Moody & Saffell)
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        self._dsr_count = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        warmup = max(1, self.n_latent)
        warmup_aligned = ((warmup + self.rebalance_freq - 1) // self.rebalance_freq) * self.rebalance_freq

        ep_len = self.episode_length if self.episode_length is not None else self.T
        ep_len = min(ep_len, self.T - warmup_aligned)

        if self.random_start and ep_len < self.T - warmup_aligned:
            max_start = self.T - ep_len
            raw_start = self.np_random.integers(warmup_aligned, max_start + 1)
            # align to rebalance grid
            self._step = int(raw_start // self.rebalance_freq * self.rebalance_freq)
        else:
            self._step = warmup_aligned

        self._episode_end = self._step + ep_len
        self._weights = np.zeros(self.n_assets, dtype=np.float32)
        self._portfolio_value = self.initial_cap
        self._portfolio_history = [self.initial_cap]
        self._weight_history = []
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        self._dsr_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Rebalance to new target weights, pay cost once
        new_weights = self._normalize_weights(action)
        turnover = float(np.abs(new_weights - self._weights).sum())
        cost     = turnover * (self.trans_cost + self.slippage)
        self._weights = new_weights

        # Apply cost up-front
        self._portfolio_value *= (1.0 - cost)

        # 2. Hold weights for rebalance_freq days, accumulate daily returns
        end_step = min(self._step + self.rebalance_freq, self.T - 1)
        daily_returns: List[float] = []
        for s in range(self._step, end_step):
            day_ret = float((self._weights * self.returns[s]).sum())
            self._portfolio_value *= (1.0 + day_ret)
            self._portfolio_history.append(self._portfolio_value)
            self._weight_history.append(self._weights.copy())
            daily_returns.append(day_ret)

        # Period-level cost-adjusted return (used by reward signal)
        gross_period_return = float(np.sum(daily_returns)) if daily_returns else 0.0
        period_return = gross_period_return - cost

        # 3. Optional PINN prior — now action-dependent: penalize gross
        #    exposure scaled by PINN-predicted uncertainty.
        pinn_pen = self._pinn_action_penalty(new_weights) if self.pinn_penalty > 0 else 0.0

        # 4. Dispersion bonus: std of abs-weight vector.
        #    std(|w|) = 0 when all positions are identical (equal-weight saturation),
        #    higher when the policy expresses differentiated conviction across assets.
        dispersion = float(np.std(np.abs(new_weights)))

        # 5. Reward: differential Sharpe + dispersion shaping
        dsr = self._differential_sharpe(period_return)
        reward = float(self.reward_scale * dsr
                       - self.pinn_penalty * pinn_pen
                       + self.dispersion_bonus * dispersion)

        self._step = end_step
        done      = self._step >= self._episode_end - 1
        truncated = False

        info = {
            "portfolio_value": self._portfolio_value,
            "period_return":   period_return,
            "gross_period_return": gross_period_return,
            "turnover":        turnover,
            "cost":            cost,
            "pinn_penalty":    pinn_pen,
            "dispersion":      dispersion,
            "weights":         new_weights,
            "daily_returns":   daily_returns,
            "dsr":             dsr,
        }
        return self._get_obs(), reward, done, truncated, info

    def render(self, mode: str = "human"):
        logger.info(
            f"Step {self._step}/{self.T} | "
            f"Portfolio: ${self._portfolio_value:,.0f} | "
            f"Weights max: {self._weights.max():.3f}"
        )

    # ------------------------------------------------------------------
    # Reward: differential Sharpe ratio (Moody & Saffell 2001)
    # ------------------------------------------------------------------

    def _differential_sharpe(self, R: float) -> float:
        """
        Per-step contribution to the running Sharpe ratio under an EMA with
        decay (1 - eta). Maximizing the sum approximates maximizing Sharpe.
        """
        eta = self.dsr_eta
        A_prev = self._dsr_A
        B_prev = self._dsr_B
        delta_A = R - A_prev
        delta_B = R * R - B_prev

        # Update EMAs
        A_new = A_prev + eta * delta_A
        B_new = B_prev + eta * delta_B
        self._dsr_A = A_new
        self._dsr_B = B_new
        self._dsr_count += 1

        # Cold start: not enough history, return scaled raw return
        if self._dsr_count < max(int(1.0 / eta), 10):
            return float(R * 100.0)

        # Use updated EMAs for variance (avoids one-step lag in denominator)
        var_proxy = max(B_new - A_new * A_new, 1e-6)
        denom = var_proxy ** 1.5
        dsr = (B_prev * delta_A - 0.5 * A_prev * delta_B) / denom
        # Clip to keep critic targets bounded under early-period instability
        return float(np.clip(dsr, -10.0, 10.0))

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step, self.T - 1)
        z   = self.latent[idx]
        unc = self.latent_uncertainty[idx]
        s   = np.array([self.sentiment[idx]], dtype=np.float32)
        w   = self._weights
        t   = np.array([idx / self.T], dtype=np.float32)
        return np.concatenate([z, unc, s, w, t]).astype(np.float32)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    # ------------------------------------------------------------------
    # PINN-based action prior (opt-in)
    # ------------------------------------------------------------------

    def _pinn_action_penalty(self, new_weights: np.ndarray) -> float:
        """
        Action-dependent prior: penalize gross exposure scaled by PINN-
        predicted latent uncertainty. When the PINN is uncertain about the
        next state, large positions are discouraged.
        """
        if self.pinn is None:
            return 0.0
        try:
            idx = min(self._step, self.T - 1)
            z_current = self.latent[idx]
            t = idx / self.T
            s = float(self.sentiment[idx])
            _, cov_pred = self.pinn.get_transition_distribution(z_current, t, s, n_samples=20)
            mean_unc = float(np.sqrt(np.diag(cov_pred).clip(0.0, None)).mean())
            gross_exposure = float(np.abs(new_weights).sum())
            return mean_unc * gross_exposure
        except Exception:
            return 0.0

    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:
        """Clip to max_pos, then normalize long/short so abs sum ≤ 1."""
        w = np.clip(action, -self.max_pos, self.max_pos)
        long_sum  = w[w > 0].sum()
        short_sum = np.abs(w[w < 0]).sum()
        if long_sum > 1.0:
            w[w > 0] /= long_sum
        if short_sum > 1.0:
            w[w < 0] /= short_sum
        return w.astype(np.float32)

    # ------------------------------------------------------------------
    # Portfolio analytics helpers
    # ------------------------------------------------------------------

    def portfolio_stats(self) -> Dict[str, float]:
        vals = np.array(self._portfolio_history)
        if len(vals) < 3:
            return {}
        rets = np.diff(vals) / vals[:-1]
        if len(rets) < 2 or rets.std() < 1e-12:
            return {"total_return": float(vals[-1] / vals[0] - 1.0)}
        sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)
        pd_vals = pd.Series(vals)
        drawdown = (pd_vals / pd_vals.cummax() - 1).min()
        return {
            "total_return": (vals[-1] / vals[0]) - 1,
            "annualized_return": ((vals[-1] / vals[0]) ** (252 / len(rets))) - 1,
            "sharpe_ratio": sharpe,
            "max_drawdown": drawdown,
            "volatility": rets.std() * np.sqrt(252),
            "final_value": vals[-1],
        }