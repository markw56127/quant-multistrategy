"""
Training orchestration: runs the full pipeline in sequential stages.

Stage 1: Data Pipeline  → returns, features, sentiment
Stage 2: Factor Discovery → PCA eigen-factors, VAE latent encoding
Stage 3: PINN Training  → Fokker-Planck physics solver
Stage 4: RL Training    → SAC policy optimization
Stage 5: Backtesting    → parallel evaluation + reward calibration
Stage 6: Risk Analysis  → Monte Carlo, VaR, confidence intervals
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from data.pipeline import DataPipeline
from data.fundamentals import FundamentalsEngine
from features.pca_factors import FactorDecomposition
from features.autoencoder import AutoencoderTrainer
from physics.pde_system import PDEConfig
from physics.pinn import PINNSolver
from rl.environment import TradingEnvironment
from rl.sac_agent import SACAgent
from rl.sb3_baselines import SB3Trainer
from optimization.estimators import LagrangianOptimizer
from risk.monte_carlo import MonteCarloSimulator
from risk.metrics import RiskMetrics
from backtest.engine import BacktestEngine, BacktestConfig
from visualization.dashboard import TradingDashboard


def parse_args():
    p = argparse.ArgumentParser(description="Train Trading Model")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--skip-data",        action="store_true", help="Use cached data")
    p.add_argument("--skip-features",    action="store_true", help="Load pretrained PCA + VAE")
    p.add_argument("--skip-pinn",        action="store_true", help="Load pretrained PINN")
    p.add_argument("--skip-rl",          action="store_true", help="Load pretrained RL")
    p.add_argument("--no-sentiment",     action="store_true", help="Skip sentiment (faster)")
    p.add_argument("--no-fundamentals",  action="store_true", help="Skip fundamental data (EPS etc.)")
    p.add_argument("--algo",            default=None,
                   choices=["SAC", "PPO", "TRPO"],
                   help="Override rl.algorithm from config")
    p.add_argument("--epochs-pinn",    type=int, default=5000)
    p.add_argument("--epochs-rl",      type=int, default=200)
    p.add_argument("--checkpoints",    default="checkpoints")
    p.add_argument("--dashboard",      action="store_true", help="Launch Dash dashboard after training")
    return p.parse_args()


def prepare_rl_inputs(cfg: dict, args) -> dict:
    """
    Run stages 1-3 (data + features + PINN) and return everything stage 4
    (RL training) needs. Honors --skip-* flags via cached checkpoints.

    Returns a dict with keys:
        returns_rl, latent_rl, unc_rl, sent_rl, pinn, data
    """
    Path(args.checkpoints).mkdir(exist_ok=True)
    Path("cache/data").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Stage 1: Data Pipeline
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Pipeline")
    logger.info("=" * 60)
    pipeline = DataPipeline(args.config)
    data = pipeline.run(
        force_refresh=not args.skip_data,
        include_sentiment=not args.no_sentiment,
    )
    logger.info(f"Pipeline output: {data}")

    # ---------------------------------------------------------------
    # Stage 2: Factor Discovery
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Dimensionality Reduction")
    logger.info("=" * 60)
    pca_cfg = cfg["features"]
    pca_path = Path(args.checkpoints) / "pca.pkl"
    factor_decomp = FactorDecomposition(
        n_pca_components=pca_cfg["n_pca_components"],
        n_global_factors=pca_cfg["n_global_factors"],
    )
    etf_returns = data.ingestion.build_return_matrix(data.etf_ohlcv) if hasattr(data, 'ingestion') else pd.DataFrame()

    if args.skip_features and pca_path.exists():
        factor_decomp.load(str(pca_path))
        global_factors = factor_decomp.global_factors_
        logger.info("PCA loaded from checkpoint")
    else:
        global_factors = factor_decomp.fit_transform(data.sector_returns, etf_returns if not etf_returns.empty else None)
        factor_decomp.save(str(pca_path))
    logger.info(f"Global factors shape: {global_factors.shape}")

    # Optionally augment feature matrix with fundamental data
    fund_cfg = cfg.get("fundamentals", {})
    include_fundamentals = (
        fund_cfg.get("enabled", True) and not getattr(args, "no_fundamentals", False)
    )
    if include_fundamentals:
        logger.info("Appending fundamental features (EPS, earnings surprise, analyst estimates)...")
        fund_engine = FundamentalsEngine()
        fund_matrix = fund_engine.build_fundamental_matrix(
            tickers=data.all_tickers,
            price_returns=data.all_returns,
            start_date=cfg["data"]["start_date"],
            end_date=cfg["data"]["end_date"],
        )
        if not fund_matrix.empty:
            data.feature_matrix = data.feature_matrix.join(fund_matrix, how="left").fillna(0.0)
            logger.info(f"Feature matrix after fundamentals: {data.feature_matrix.shape}")

    X, y = pipeline.get_aligned_input(data, normalize=False)
    if X.shape[1] == 0:
        raise RuntimeError("Feature matrix has 0 columns. Check technical_indicators.py.")
    vae_cfg = pca_cfg["autoencoder"]
    vae_path = Path(args.checkpoints) / "vae.pt"
    vae = AutoencoderTrainer(
        input_dim=X.shape[1],
        hidden_dims=vae_cfg["hidden_dims"],
        latent_dim=vae_cfg["latent_dim"],
        beta=vae_cfg["beta_vae"],
    )
    if args.skip_features and vae_path.exists():
        vae.load(str(vae_path))
        latent_df = pd.DataFrame(
            vae.transform(X.values),
            index=X.index,
            columns=[f"z_{i+1}" for i in range(vae_cfg["latent_dim"])],
        )
        logger.info("VAE loaded from checkpoint")
    else:
        latent_df = vae.fit_transform(X.values, index=X.index, epochs=500, batch_size=256)
        vae.save(str(vae_path))
    latent_mean, latent_std = vae.transform_with_uncertainty(X.values)

    # ---------------------------------------------------------------
    # Stage 3: PINN Training
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3: Physics-Informed Neural Network")
    logger.info("=" * 60)
    pde_cfg_raw = cfg["pde"]
    pinn_cfg_raw = cfg["pinn"]
    pde_config = PDEConfig(
        n_latent=vae_cfg["latent_dim"],
        dt=pde_cfg_raw["dt"],
        diffusion_coeff=pde_cfg_raw["diffusion_coeff"],
        chaos_sensitivity=pde_cfg_raw["chaos_sensitivity"],
        lower_bound=pde_cfg_raw["boundary"]["lower"],
        upper_bound=pde_cfg_raw["boundary"]["upper"],
    )
    pinn_path = Path(args.checkpoints) / "pinn.pt"
    pinn = PINNSolver(
        config=pde_config,
        hidden_dims=pinn_cfg_raw["hidden_dims"],
        activation=pinn_cfg_raw["activation"],
        lambda_physics=pinn_cfg_raw["lambda_physics"],
        lambda_bc=pinn_cfg_raw["lambda_bc"],
        lambda_ic=pinn_cfg_raw["lambda_ic"],
        learning_rate=pinn_cfg_raw["learning_rate"],
        adaptive_weights=pinn_cfg_raw.get("adaptive_weights", True),
        adaptive_update_freq=pinn_cfg_raw.get("adaptive_update_freq", 100),
        adaptive_momentum=pinn_cfg_raw.get("adaptive_momentum", 0.9),
    )
    if args.skip_pinn and pinn_path.exists():
        pinn.load(str(pinn_path))
        logger.info("PINN loaded from checkpoint")
    else:
        # Compute common_idx early so PINN and RL share the same sentiment slice
        _common_idx = latent_df.index.intersection(data.all_returns.index)
        sentiment_arr = (
            data.sentiment_matrix.mean(axis=1).reindex(_common_idx).ffill().fillna(0.0).values
            if not data.sentiment_matrix.empty else None
        )
        loss_hist = pinn.fit(
            latent_trajectories=latent_mean[:len(_common_idx)],
            sentiment_series=sentiment_arr,
            epochs=args.epochs_pinn,
            n_collocation=pinn_cfg_raw["n_collocation"],
            n_boundary=pinn_cfg_raw["n_boundary"],
            patience=pinn_cfg_raw["patience"],
        )
        pinn.save(str(pinn_path))
        logger.info(f"PINN trained. Final loss: {loss_hist[-1]['total']:.4f}")

    # Align returns and latent for the RL stage
    common_idx = latent_df.index.intersection(data.all_returns.index)
    returns_rl = data.all_returns.loc[common_idx].fillna(0.0)
    latent_rl  = latent_df.loc[common_idx]
    unc_rl     = latent_std[:len(common_idx)]

    # Build sentiment aligned to common_idx — used for both PINN and RL so both
    # see the same domain. PINN training above uses latent_df.index which equals
    # common_idx after the intersection, so this is consistent.
    sent_series = (
        data.sentiment_matrix.mean(axis=1).reindex(common_idx).ffill().fillna(0.0)
        if not data.sentiment_matrix.empty else None
    )
    sent_rl = sent_series

    return {
        "returns_rl": returns_rl,
        "latent_rl":  latent_rl,
        "unc_rl":     unc_rl,
        "sent_rl":    sent_rl,
        "pinn":       pinn,
        "data":       data,
        "vae":        vae,
        "latent_df":  latent_df,
    }


def build_env_and_agent(cfg: dict, prep: dict, random_start: bool = False):
    """Construct env + SAC agent from a config dict and prepared inputs."""
    rl_cfg = cfg["rl"]
    _bt_freq = cfg["backtest"]["rebalance_freq"]
    if isinstance(_bt_freq, str):
        _bt_freq = {"daily": 1, "weekly": 5, "monthly": 21}.get(_bt_freq, 5)
    rebalance_freq = int(rl_cfg.get("rebalance_freq", _bt_freq))

    env = TradingEnvironment(
        returns=prep["returns_rl"],
        latent_factors=prep["latent_rl"],
        latent_uncertainty=prep["unc_rl"],
        sentiment=prep["sent_rl"],
        pinn_solver=prep["pinn"],
        transaction_cost=cfg["backtest"]["transaction_cost"],
        slippage=cfg["backtest"]["slippage"],
        max_position_size=cfg["backtest"]["max_position_size"],
        rebalance_freq=rebalance_freq,
        dsr_eta=rl_cfg.get("dsr_eta", 0.01),
        reward_scale=rl_cfg.get("reward_scale", 1.0),
        pinn_penalty=rl_cfg.get("pinn_penalty", 0.0),
        dispersion_bonus=rl_cfg.get("dispersion_bonus", 0.0),
        episode_length=rl_cfg.get("episode_length", 252) if random_start else None,
        random_start=random_start,
    )
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.n_assets,
        hidden_dims=rl_cfg["hidden_dims"],
        learning_rate=rl_cfg["learning_rate"],
        gamma=rl_cfg["gamma"],
        tau=rl_cfg["tau"],
        batch_size=rl_cfg["batch_size"],
        buffer_size=rl_cfg["buffer_size"],
        warmup_steps=rl_cfg["warmup_steps"],
        auto_entropy=rl_cfg.get("auto_entropy", True),
        target_entropy_scale=rl_cfg.get("target_entropy_scale", 0.5),
        log_alpha_min=rl_cfg.get("log_alpha_min", -5.0),
    )
    return env, agent


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prep = prepare_rl_inputs(cfg, args)
    data       = prep["data"]
    returns_rl = prep["returns_rl"]
    latent_rl  = prep["latent_rl"]
    unc_rl     = prep["unc_rl"]
    sent_rl    = prep["sent_rl"]
    pinn       = prep["pinn"]

    # ---------------------------------------------------------------
    # Stage 4: Reinforcement Learning
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 4: Reinforcement Learning (SAC)")
    logger.info("=" * 60)
    rl_algo = (args.algo or cfg["rl"].get("algorithm", "SAC")).upper()
    rl_path_stem = Path(args.checkpoints) / f"{rl_algo.lower()}_agent"
    env, sac_agent = build_env_and_agent(cfg, prep, random_start=True)

    if rl_algo == "SAC":
        agent = sac_agent
        rl_path = Path(str(rl_path_stem) + ".pt")
        if args.skip_rl and rl_path.exists():
            agent.load(str(rl_path))
            logger.info("SAC agent loaded from checkpoint")
        else:
            _train_rl(agent, env, n_episodes=args.epochs_rl)
            agent.save(str(rl_path))
            logger.info(f"SAC training complete. Episodes: {args.epochs_rl}")

    else:  # PPO or TRPO via SB3
        rl_path = Path(str(rl_path_stem))
        sb3_trainer = SB3Trainer(algo=rl_algo, env=env, cfg=cfg["rl"])
        if args.skip_rl and (rl_path.with_suffix(".zip")).exists():
            sb3_trainer.load(str(rl_path))
        else:
            total_ts = cfg["rl"].get("sb3_total_timesteps", 500_000)
            sb3_trainer.learn(total_timesteps=total_ts)
            sb3_trainer.save(str(rl_path))
            logger.info(f"{rl_algo} training complete. Timesteps: {total_ts}")
        agent = sb3_trainer

    # ---------------------------------------------------------------
    # Stage 5: Backtesting
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 5: Backtesting")
    logger.info("=" * 60)
    bt_cfg = cfg["backtest"]
    backtest_engine = BacktestEngine(
        returns=returns_rl,
        benchmark_returns=data.benchmark_returns,
        n_workers=bt_cfg["n_parallel_workers"],
    )

    # Extract learned policy weights (works for both SAC and SB3 agents)
    if isinstance(agent, SB3Trainer):
        policy_weights = agent.extract_policy_weights(returns_rl)
    else:
        policy_weights = _extract_policy_weights(agent, env, returns_rl, latent_rl, unc_rl, sent_rl)

    # Lagrangian re-optimization of policy weights
    lagrangian = LagrangianOptimizer(
        max_position_size=bt_cfg["max_position_size"],
        risk_aversion=cfg["optimization"]["risk_budget"],
        market_neutral=False,
    )
    _freq = bt_cfg["rebalance_freq"]
    if isinstance(_freq, str):
        _freq = _freq.replace("weekly", "5").replace("monthly", "21").replace("daily", "1")
    rebalance_every = int(_freq)
    opt_weights = lagrangian.rolling_optimize(returns_rl, lookback=60, rebalance_every=rebalance_every)

    bt_common = {k: bt_cfg[k] for k in ["initial_capital", "transaction_cost", "slippage", "max_position_size"]}
    configs = [
        (policy_weights, BacktestConfig(name=f"{rl_algo}_Policy",  **bt_common)),
        (opt_weights,    BacktestConfig(name="Lagrangian_Opt", **bt_common)),
    ]

    bt_results = backtest_engine.run_parallel(configs)
    comparison = backtest_engine.cross_compare(bt_results)
    logger.info(f"\n{comparison.to_string()}")

    # ---------------------------------------------------------------
    # Stage 6: Risk Analysis
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 6: Risk & Uncertainty Analysis")
    logger.info("=" * 60)
    best_result = bt_results[0]
    risk_cfg = cfg["risk"]

    mc = MonteCarloSimulator(n_simulations=risk_cfg["n_monte_carlo"], horizon_days=risk_cfg["horizon_days"])
    best_weights = policy_weights.iloc[-1].values
    mc_paths = mc.simulate_gbm(returns_rl, best_weights, lookback=risk_cfg["lookback_days"])
    mc_bands = mc.confidence_bands(mc_paths)
    mc_stats  = mc.path_statistics(mc_paths)
    logger.info(f"MC Stats: {mc_stats}")

    risk = RiskMetrics(confidence_levels=risk_cfg["confidence_levels"])
    report = risk.full_report(best_result.returns, data.benchmark_returns.values if not data.benchmark_returns.empty else None, mc_paths)
    logger.info(risk.format_report(report))

    # ---------------------------------------------------------------
    # Stage 7: Visualization
    # ---------------------------------------------------------------
    if args.dashboard:
        logger.info("Launching dashboard...")
        dash = TradingDashboard()
        reward_hist = [m["total"] for m in pinn._loss_history]
        dash.launch_app(
            portfolio_values=best_result.portfolio_values,
            returns=best_result.returns,
            latent=latent_rl.values,
            reward_history=reward_hist,
            mc_bands=mc_bands,
            port=8050,
        )


def _train_rl(agent: SACAgent, env: TradingEnvironment, n_episodes: int) -> list:
    """Run SAC training loop."""
    reward_history = []
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        done = truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            metrics = agent.update(n_updates=1)
            ep_reward += reward
            state = next_state
        reward_history.append(ep_reward)
        if ep % 10 == 0:
            stats = env.portfolio_stats()
            logger.info(
                f"Episode {ep}/{n_episodes} | "
                f"Reward={ep_reward:.3f} | "
                f"Sharpe={stats.get('sharpe_ratio', 0):.3f} | "
                f"Alpha={agent.alpha:.4f}"
            )
    return reward_history


def _extract_policy_weights(agent, env, returns, latent, unc, sent) -> pd.DataFrame:
    """Run deterministic policy rollout and collect weight decisions.

    Weights are indexed on the *start* of each holding period (the date the
    agent committed to them), not the end. This matches how the backtest
    engine ffills weights forward.
    """
    state, _ = env.reset()
    weights_list: list = []
    dates_list:   list = []
    done = truncated = False
    while not (done or truncated):
        period_start_step = env._step
        action = agent.select_action(state, deterministic=True)
        state, _, done, truncated, info = env.step(action)
        weights_list.append(info["weights"])
        dates_list.append(returns.index[min(period_start_step, len(returns) - 1)])
    return pd.DataFrame(weights_list, index=dates_list, columns=returns.columns)


if __name__ == "__main__":
    main()
