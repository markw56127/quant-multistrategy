"""
Small RL hyperparameter sweep.

Reuses cached upstream artifacts (data, VAE, PINN) — point this at a
checkpoints/ dir that already has them, and pass --skip-data --skip-features
--skip-pinn so the upstream stages load from cache.

For each grid point we:
  1. Build a fresh env + SAC agent with the swept params.
  2. Train for --epochs episodes.
  3. Run the deterministic policy and backtest it.
  4. Record sharpe, total return, max drawdown, final alpha.

A summary table is printed at the end and saved to a CSV.

Usage (after one full train.py run has populated checkpoints/):
    python tune_rl.py --skip-data --skip-features --skip-pinn --no-sentiment \
        --epochs 100 --out sweep_results.csv
"""

from __future__ import annotations

import argparse
import copy
import itertools
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from loguru import logger

from train import (
    prepare_rl_inputs,
    build_env_and_agent,
    _train_rl,
    _extract_policy_weights,
)
from backtest.engine import BacktestConfig, BacktestEngine


def parse_args():
    p = argparse.ArgumentParser(description="Sweep RL hyperparameters")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--out", default="sweep_results.csv")
    p.add_argument("--epochs", type=int, default=100,
                   help="episodes per grid point")
    p.add_argument("--skip-data",     action="store_true")
    p.add_argument("--skip-features", action="store_true")
    p.add_argument("--skip-pinn",     action="store_true")
    p.add_argument("--no-sentiment",  action="store_true")
    p.add_argument("--epochs-pinn",   type=int, default=5000,
                   help="only used if --skip-pinn is not set")
    # Sweep-grid overrides (comma-separated lists)
    p.add_argument("--target-entropy-scale", default="0.3,0.5,0.7")
    p.add_argument("--dsr-eta",              default="0.01")
    p.add_argument("--learning-rate",        default="3e-4")
    return p.parse_args()


def parse_list(arg: str, cast=float) -> List:
    return [cast(x.strip()) for x in arg.split(",") if x.strip()]


def evaluate(agent, env, cfg: dict, returns_rl: pd.DataFrame, benchmark_returns) -> Dict[str, float]:
    """Deterministic rollout → backtest → metrics."""
    weights = _extract_policy_weights(
        agent, env, returns_rl, env.latent, env.latent_uncertainty, env.sentiment
    )
    bt_cfg_raw = cfg["backtest"]
    _freq = bt_cfg_raw["rebalance_freq"]
    if isinstance(_freq, str):
        _freq = {"daily": 1, "weekly": 5, "monthly": 21}.get(_freq, 5)
    bt_cfg = BacktestConfig(
        name="sweep",
        initial_capital=bt_cfg_raw["initial_capital"],
        transaction_cost=bt_cfg_raw["transaction_cost"],
        slippage=bt_cfg_raw["slippage"],
        max_position_size=bt_cfg_raw["max_position_size"],
        rebalance_freq=int(_freq),
    )
    engine = BacktestEngine(returns=returns_rl, benchmark_returns=benchmark_returns)
    result = engine.run_single(weights, bt_cfg)
    rets = pd.Series(result.returns) if not isinstance(result.returns, pd.Series) else result.returns
    if len(rets) < 2:
        return {"sharpe": float("nan"), "total_return": float("nan"), "max_dd": float("nan")}
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * (252 ** 0.5))
    pv = pd.Series(result.portfolio_values)
    total  = float(pv.iloc[-1] / pv.iloc[0] - 1.0)
    max_dd = float((pv / pv.cummax() - 1.0).min())
    return {"sharpe": sharpe, "total_return": total, "max_dd": max_dd}


def run_sweep(args) -> pd.DataFrame:
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    # Need these flags on the args for prepare_rl_inputs
    prep = prepare_rl_inputs(base_cfg, args)
    returns_rl = prep["returns_rl"]
    benchmark_returns = prep["data"].benchmark_returns

    grid = list(itertools.product(
        parse_list(args.target_entropy_scale),
        parse_list(args.dsr_eta),
        parse_list(args.learning_rate),
    ))
    logger.info(f"Sweep size: {len(grid)} combos × {args.epochs} episodes")

    rows: List[Dict[str, Any]] = []
    for i, (tes, eta, lr) in enumerate(grid, 1):
        logger.info("=" * 60)
        logger.info(f"COMBO {i}/{len(grid)}: target_entropy_scale={tes}  dsr_eta={eta}  lr={lr}")
        logger.info("=" * 60)

        cfg = copy.deepcopy(base_cfg)
        cfg["rl"]["target_entropy_scale"] = float(tes)
        cfg["rl"]["dsr_eta"]              = float(eta)
        cfg["rl"]["learning_rate"]        = float(lr)

        env, agent = build_env_and_agent(cfg, prep)
        t0 = time.time()
        _train_rl(agent, env, n_episodes=args.epochs)
        train_secs = time.time() - t0

        metrics = evaluate(agent, env, cfg, returns_rl, benchmark_returns)
        row = {
            "target_entropy_scale": tes,
            "dsr_eta":              eta,
            "learning_rate":        lr,
            "sharpe":               metrics["sharpe"],
            "total_return":         metrics["total_return"],
            "max_dd":               metrics["max_dd"],
            "final_alpha":          float(agent.alpha),
            "train_seconds":        round(train_secs, 1),
        }
        rows.append(row)
        logger.info(f"→ Sharpe={row['sharpe']:.3f}  total={row['total_return']:.2%}  "
                    f"maxDD={row['max_dd']:.2%}  alpha={row['final_alpha']:.4f}  "
                    f"({row['train_seconds']:.0f}s)")

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df


def main():
    args = parse_args()
    Path(args.checkpoints).mkdir(exist_ok=True)
    df = run_sweep(args)
    logger.info("\n" + "=" * 60 + "\nSWEEP RESULTS (sorted by Sharpe)\n" + "=" * 60)
    logger.info("\n" + df.to_string(index=False))
    df.to_csv(args.out, index=False)
    logger.info(f"\nSaved → {args.out}")
    best = df.iloc[0]
    logger.info(
        f"\nBest: target_entropy_scale={best['target_entropy_scale']}  "
        f"dsr_eta={best['dsr_eta']}  lr={best['learning_rate']}  "
        f"→ Sharpe={best['sharpe']:.3f}"
    )


if __name__ == "__main__":
    main()