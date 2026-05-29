"""
Stable-Baselines3 baseline algorithms: PPO and TRPO.

Both run on the same TradingEnvironment as the custom SAC agent, enabling
apples-to-apples comparison of:
  - PPO  (on-policy, clipped surrogate, SB3 built-in)
  - TRPO (on-policy, trust-region, requires sb3-contrib)
  - SAC  (off-policy, max-entropy, custom implementation)

Usage:
    from rl.sb3_baselines import SB3Trainer
    trainer = SB3Trainer(algo="PPO", env=env, cfg=cfg["rl"])
    trainer.learn(total_timesteps=500_000)
    weights_df = trainer.extract_policy_weights(returns_df)
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


def _make_sb3_algo(algo: str, env, cfg: dict):
    """Instantiate an SB3 algorithm on the given env with config hyperparams."""
    policy_kwargs = dict(net_arch=cfg.get("hidden_dims", [256, 256]))
    lr = cfg.get("learning_rate", 3e-4)
    gamma = cfg.get("gamma", 0.99)

    if algo == "PPO":
        from stable_baselines3 import PPO
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=lr,
            gamma=gamma,
            n_steps=cfg.get("ppo_n_steps", 2048),
            batch_size=cfg.get("batch_size", 256),
            n_epochs=cfg.get("ppo_n_epochs", 10),
            clip_range=cfg.get("ppo_clip", 0.2),
            ent_coef=cfg.get("ppo_ent_coef", 0.01),
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

    if algo == "TRPO":
        try:
            from sb3_contrib import TRPO
        except ImportError:
            raise ImportError(
                "TRPO requires sb3-contrib: pip install sb3-contrib"
            )
        return TRPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=lr,
            gamma=gamma,
            n_steps=cfg.get("trpo_n_steps", 2048),
            batch_size=cfg.get("batch_size", 256),
            cg_max_steps=cfg.get("trpo_cg_steps", 15),
            target_kl=cfg.get("trpo_target_kl", 0.01),
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

    raise ValueError(f"Unknown SB3 algo: {algo!r}. Choose 'PPO' or 'TRPO'.")


class SB3Trainer:
    """
    Thin wrapper around SB3 algorithms that provides:
      - .learn()                 — runs the SB3 training loop
      - .extract_policy_weights() — deterministic rollout → DataFrame of weights
      - .save() / .load()        — checkpoint helpers
    """

    def __init__(self, algo: str, env, cfg: dict):
        self.algo = algo.upper()
        self.env  = env
        self.cfg  = cfg
        self.model = _make_sb3_algo(self.algo, env, cfg)
        logger.info(f"SB3Trainer: algo={self.algo}")

    def learn(self, total_timesteps: int) -> "SB3Trainer":
        logger.info(f"{self.algo}: starting training for {total_timesteps:,} timesteps")
        self.model.learn(total_timesteps=total_timesteps, progress_bar=False)
        logger.info(f"{self.algo}: training complete")
        return self

    def extract_policy_weights(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Run a deterministic rollout and collect the portfolio weights the
        policy committed to at each rebalance step.

        Returns a DataFrame indexed on the same dates as `returns`, forward-
        filled between rebalance steps (matching the backtest engine contract).
        """
        obs, _ = self.env.reset()
        done = truncated = False
        weights_list = []
        dates_list   = []

        while not (done or truncated):
            period_start_step = self.env._step
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = self.env.step(action)
            weights_list.append(info["weights"])
            dates_list.append(returns.index[min(period_start_step, len(returns) - 1)])

        return pd.DataFrame(weights_list, index=dates_list, columns=returns.columns)

    def save(self, path: str):
        self.model.save(path)
        logger.info(f"{self.algo} saved to {path}")

    def load(self, path: str):
        if self.algo == "PPO":
            from stable_baselines3 import PPO
            self.model = PPO.load(path, env=self.env)
        elif self.algo == "TRPO":
            from sb3_contrib import TRPO
            self.model = TRPO.load(path, env=self.env)
        logger.info(f"{self.algo} loaded from {path}")
        return self