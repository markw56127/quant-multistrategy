"""
Soft Actor-Critic (SAC) agent with PDE-constrained policy.

SAC is an off-policy, maximum-entropy RL algorithm well-suited to the
stochastic market environment. The entropy bonus encourages exploration
and prevents premature convergence to suboptimal deterministic policies.

Key components:
  - Actor:    Gaussian policy π(a|s) with re-parameterization trick
  - Critics:  Twin Q-networks Q1, Q2 to reduce overestimation
  - Target:   Polyak-averaged target critics
  - Alpha:    Automatic entropy tuning
  - Replay:   Prioritized experience replay buffer
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ------------------------------------------------------------------
# Neural network components
# ------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int], activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    """
    Diagonal Gaussian policy: outputs (mean, log_std) for each action dimension.
    Uses re-parameterization trick for backprop through sampling.
    """
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int, hidden: List[int]):
        super().__init__()
        self.trunk = MLP(state_dim, hidden[-1], hidden[:-1])
        self.mu_head      = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob) with tanh squashing."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t  = dist.rsample()
        y_t  = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        return y_t, log_prob.sum(dim=-1, keepdim=True)

    def mean_action(self, state: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(state)
        return torch.tanh(mu)


class TwinCritic(nn.Module):
    """Twin Q-networks to reduce Q-value overestimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden: List[int]):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden)
        self.q2 = MLP(state_dim + action_dim, 1, hidden)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ------------------------------------------------------------------
# Replay Buffer
# ------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity    = capacity
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self._states     = np.zeros((capacity, state_dim),  dtype=np.float32)
        self._actions    = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards    = np.zeros((capacity, 1),          dtype=np.float32)
        self._next_states= np.zeros((capacity, state_dim),  dtype=np.float32)
        self._dones      = np.zeros((capacity, 1),          dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self._states[self._ptr]      = state
        self._actions[self._ptr]     = action
        self._rewards[self._ptr]     = reward
        self._next_states[self._ptr] = next_state
        self._dones[self._ptr]       = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "states":      torch.FloatTensor(self._states[idx]).to(device),
            "actions":     torch.FloatTensor(self._actions[idx]).to(device),
            "rewards":     torch.FloatTensor(self._rewards[idx]).to(device),
            "next_states": torch.FloatTensor(self._next_states[idx]).to(device),
            "dones":       torch.FloatTensor(self._dones[idx]).to(device),
        }

    def __len__(self) -> int:
        return self._size


# ------------------------------------------------------------------
# SAC Agent
# ------------------------------------------------------------------

class SACAgent:
    """
    Soft Actor-Critic with automatic entropy tuning.

    The PDE constraint is enforced through the environment reward, so the
    agent naturally learns policies consistent with market physics.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = (512, 256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        target_entropy_scale: float = 0.5,
        log_alpha_min: float = -5.0,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        warmup_steps: int = 10_000,
        device: Optional[str] = None,
    ):
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.warmup_steps = warmup_steps
        self.log_alpha_min = log_alpha_min
        self.device      = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else
                "mps"  if torch.backends.mps.is_available() else
                "cpu"
            )
        )

        self.actor   = GaussianActor(state_dim, action_dim, list(hidden_dims)).to(self.device)
        self.critic  = TwinCritic(state_dim, action_dim, list(hidden_dims)).to(self.device)
        self.critic_target = TwinCritic(state_dim, action_dim, list(hidden_dims)).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Automatic entropy coefficient.
        # Heuristic target_entropy = -action_dim collapses alpha for large
        # action spaces (e.g. 50 assets); scale it down to keep exploration alive.
        self.auto_entropy = auto_entropy
        if auto_entropy:
            self.target_entropy = -float(action_dim) * float(target_entropy_scale)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        self._total_steps = 0
        self.train_metrics: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self._total_steps < self.warmup_steps and not deterministic:
            return np.random.uniform(-1, 1, size=self.actor.mu_head.out_features)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                action = self.actor.mean_action(s)
            else:
                action, _ = self.actor.sample(s)
        return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # Experience storage and training
    # ------------------------------------------------------------------

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self._total_steps += 1

    def update(self, n_updates: int = 1) -> Optional[Dict[str, float]]:
        if len(self.buffer) < self.batch_size:
            return None

        metrics = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0, "alpha": self.alpha}

        for _ in range(n_updates):
            batch = self.buffer.sample(self.batch_size, self.device)
            m = self._update_step(batch)
            for k in metrics:
                metrics[k] += m.get(k, 0.0)

        for k in metrics:
            metrics[k] /= n_updates
        self.train_metrics.append(metrics)
        return metrics

    def _update_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        s, a, r, s_, d = (
            batch["states"], batch["actions"], batch["rewards"],
            batch["next_states"], batch["dones"],
        )

        # --- Critic update ---
        with torch.no_grad():
            a_, log_pi_ = self.actor.sample(s_)
            q_target    = self.critic_target.min_q(s_, a_)
            y = r + self.gamma * (1 - d) * (q_target - self.alpha * log_pi_)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # --- Actor update ---
        a_new, log_pi = self.actor.sample(s)
        q_new = self.critic.min_q(s, a_new)
        actor_loss = (self.alpha * log_pi - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # --- Alpha update ---
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss_t = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss_t.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(min=self.log_alpha_min)
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss_t.item()

        # --- Target network soft update ---
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss,
            "alpha":       self.alpha,
        }

    def _soft_update(self):
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
