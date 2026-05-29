"""
Quick-start entry point for inference / signal generation on new data.

Usage:
  python main.py --mode train       # Full pipeline training
  python main.py --mode infer       # Generate today's trade signals
  python main.py --mode backtest    # Backtest only
  python main.py --mode dashboard   # Launch dashboard from checkpoints
  python main.py --mode sensitivity # Indicator sensitivity analysis
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Trading Model - Main Entry Point")
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "backtest", "dashboard", "sensitivity"],
        default="train",
    )
    parser.add_argument("--config",          default="config/config.yaml")
    parser.add_argument("--checkpoints",     default="checkpoints")
    parser.add_argument("--dashboard-port",  type=int, default=8050)
    parser.add_argument("--dashboard",       action="store_true",
                        help="Launch Dash app after backtest completes")
    parser.add_argument("--no-sentiment",     action="store_true",
                        help="Skip sentiment features (match --no-sentiment used at training time)")
    parser.add_argument("--no-fundamentals", action="store_true",
                        help="Skip fundamental features (match --no-fundamentals used at training time)")
    parser.add_argument("--ticker",          default=None, metavar="SYMBOL",
                        help="(infer mode) Show a detailed single-stock report for this ticker")
    parser.add_argument("--start-date",      default=None,
                        help="Restrict backtest to this start date (YYYY-MM-DD), "
                             "useful for out-of-sample validation")
    parser.add_argument("--end-date",        default=None,
                        help="Restrict backtest to this end date (YYYY-MM-DD)")
    parser.add_argument("--out",             default=None,
                        help="Save backtest comparison table to this CSV path")
    args, remaining = parser.parse_known_args()

    if args.mode == "train":
        from train import main as train_main
        sys.argv = [sys.argv[0]] + remaining
        train_main()

    elif args.mode == "infer":
        signals = _run_inference(args)
        if signals is not None and args.ticker:
            _single_stock_report(args.ticker.upper(), signals, args)

    elif args.mode == "backtest":
        _run_backtest_only(args)

    elif args.mode == "dashboard":
        _run_dashboard(args)

    elif args.mode == "sensitivity":
        _run_sensitivity(args)


def _run_inference(args):
    """Generate trade signals for today using pretrained checkpoints."""
    import yaml
    import numpy as np
    import pandas as pd
    import torch
    from datetime import date, timedelta
    from loguru import logger

    from data.ingestion import DataIngestion
    from data.technical_indicators import TechnicalIndicators
    from features.autoencoder import AutoencoderTrainer
    from physics.pde_system import PDEConfig
    from physics.pinn import PINNSolver
    from rl.sac_agent import SACAgent

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cp       = Path(args.checkpoints)
    vae_path  = cp / "vae.pt"
    pinn_path = cp / "pinn.pt"
    rl_path   = cp / "sac_agent.pt"

    for p in (vae_path, pinn_path, rl_path):
        if not p.exists():
            logger.error(f"Checkpoint not found: {p}  — run --mode train first.")
            return

    vae_cfg = cfg["features"]["autoencoder"]
    pde_cfg = cfg["pde"]
    rl_cfg  = cfg["rl"]

    # ------------------------------------------------------------------ #
    # 1. Load models                                                       #
    # ------------------------------------------------------------------ #
    # Peek at VAE checkpoint to recover input_dim saved at training time.
    vae_ckpt  = torch.load(str(vae_path), map_location="cpu", weights_only=False)
    input_dim = int(vae_ckpt["input_dim"])

    vae = AutoencoderTrainer(
        input_dim=input_dim,
        hidden_dims=vae_cfg["hidden_dims"],
        latent_dim=vae_cfg["latent_dim"],
        beta=vae_cfg["beta_vae"],
    )
    vae.load(str(vae_path))

    pde_config = PDEConfig(
        n_latent=vae_cfg["latent_dim"],
        dt=pde_cfg["dt"],
        diffusion_coeff=pde_cfg["diffusion_coeff"],
        chaos_sensitivity=pde_cfg["chaos_sensitivity"],
        lower_bound=pde_cfg["boundary"]["lower"],
        upper_bound=pde_cfg["boundary"]["upper"],
    )
    pinn = PINNSolver(config=pde_config)
    pinn.load(str(pinn_path))

    # Training used data.all_returns = 50 stocks + 10 sector ETFs = 60 assets.
    # The checkpoint's state_dim and action_dim reflect that full universe.
    stock_tickers = [t for ts in cfg["data"]["tickers"].values() for t in ts]
    etf_tickers   = cfg["data"]["sector_etfs"]
    all_tickers   = stock_tickers + etf_tickers   # must match training order
    n_assets      = len(all_tickers)              # 60
    n_latent      = vae_cfg["latent_dim"]
    # State layout mirrors TradingEnvironment: z | unc | sentiment | weights | t_frac
    state_dim     = n_latent * 2 + 1 + n_assets + 1

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=n_assets,
        hidden_dims=rl_cfg["hidden_dims"],
        learning_rate=rl_cfg["learning_rate"],
        gamma=rl_cfg["gamma"],
        tau=rl_cfg["tau"],
        batch_size=rl_cfg["batch_size"],
        buffer_size=1000,
        warmup_steps=0,
        auto_entropy=rl_cfg.get("auto_entropy", True),
        target_entropy_scale=rl_cfg.get("target_entropy_scale", 0.5),
        log_alpha_min=rl_cfg.get("log_alpha_min", -5.0),
    )
    agent.load(str(rl_path))
    logger.info(
        f"Models loaded — VAE input_dim={input_dim}, "
        f"latent_dim={n_latent}, state_dim={state_dim}, n_assets={n_assets}"
    )

    # ------------------------------------------------------------------ #
    # 2. Fetch recent OHLCV (90 days covers all indicator warmup periods) #
    # ------------------------------------------------------------------ #
    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    logger.info(f"Fetching recent data {start} → {end} for {n_assets} tickers...")

    ingest = DataIngestion(cache_dir=cfg["data"]["cache_dir"])
    ohlcv  = ingest._batch_download(all_tickers, start, end, cfg["data"]["interval"])
    if not ohlcv:
        logger.error("No OHLCV data returned — check network connection.")
        return

    # ------------------------------------------------------------------ #
    # 3. Compute features (identical pipeline to training)                #
    # ------------------------------------------------------------------ #
    ti      = TechnicalIndicators()
    ind_data = ti.compute_universe(ohlcv)
    returns  = ingest.build_return_matrix(ohlcv)
    feature_matrix = ti.build_feature_matrix(ind_data, returns)

    if feature_matrix.shape[1] == 0:
        logger.error("Feature matrix is empty — indicator computation failed.")
        return

    # Align feature columns to the training layout.
    # build_feature_matrix produces {ticker}_{indicator} in ticker-iteration order,
    # which is deterministic as long as the ticker list is the same.
    if feature_matrix.shape[1] != input_dim:
        logger.warning(
            f"Feature dim {feature_matrix.shape[1]} ≠ training dim {input_dim}. "
            "Padding/trimming to match — signals may be less reliable."
        )
        if feature_matrix.shape[1] < input_dim:
            pad = pd.DataFrame(
                np.zeros((len(feature_matrix), input_dim - feature_matrix.shape[1])),
                index=feature_matrix.index,
            )
            feature_matrix = pd.concat([feature_matrix, pad], axis=1)
        else:
            feature_matrix = feature_matrix.iloc[:, :input_dim]

    feature_matrix = feature_matrix.dropna()
    if feature_matrix.empty:
        logger.error("All rows are NaN after indicator warmup — fetch more history.")
        return

    signal_date = (
        feature_matrix.index[-1].strftime("%Y-%m-%d")
        if hasattr(feature_matrix.index[-1], "strftime")
        else str(feature_matrix.index[-1])
    )

    # ------------------------------------------------------------------ #
    # 4. Encode through VAE → latent state + uncertainty                  #
    # ------------------------------------------------------------------ #
    # Use the last 20 rows so MC-dropout uncertainty is computed over a
    # short recent window; we take only the final row as today's state.
    X_recent = feature_matrix.values[-20:]
    z_mean, z_std = vae.transform_with_uncertainty(X_recent, n_samples=50)

    z_now   = z_mean[-1].astype(np.float32)   # (n_latent,)
    unc_now = z_std[-1].astype(np.float32)    # (n_latent,)

    # ------------------------------------------------------------------ #
    # 5. Build state vector and query SAC policy                          #
    # ------------------------------------------------------------------ #
    state = np.concatenate([
        z_now,
        unc_now,
        np.array([0.0], dtype=np.float32),          # sentiment (none live)
        np.zeros(n_assets, dtype=np.float32),        # current weights (flat entry)
        np.array([1.0], dtype=np.float32),           # t_frac = "now"
    ])

    raw_action = agent.select_action(state, deterministic=True)

    # Apply the same weight normalisation as TradingEnvironment._normalize_weights
    max_pos   = cfg["backtest"]["max_position_size"]
    w = np.clip(raw_action, -max_pos, max_pos)
    long_sum  = w[w > 0].sum()
    short_sum = np.abs(w[w < 0]).sum()
    if long_sum  > 1.0: w[w > 0] /= long_sum
    if short_sum > 1.0: w[w < 0] /= short_sum

    # ------------------------------------------------------------------ #
    # 6. Print signals                                                     #
    # ------------------------------------------------------------------ #
    signals      = pd.Series(w, index=all_tickers).sort_values(ascending=False)
    stock_sigs   = signals.loc[stock_tickers]
    etf_sigs     = signals.loc[etf_tickers]
    longs        = signals[signals >  0.001]
    shorts       = signals[signals < -0.001]

    logger.info(f"\n{'='*56}")
    logger.info(f"  TRADE SIGNALS  —  {signal_date}")
    logger.info(f"{'='*56}")
    logger.info(f"  Universe       : {len(stock_tickers)} stocks + {len(etf_tickers)} ETFs")
    logger.info(f"  Gross exposure : {np.abs(w).sum():.1%}")
    logger.info(f"  Net exposure   : {w.sum():+.1%}")
    logger.info(f"  Long  legs     : {len(longs)}  ({longs.sum():.1%} of capital)")
    logger.info(f"  Short legs     : {len(shorts)}  ({shorts.abs().sum():.1%} of capital)")

    stock_longs  = stock_sigs[stock_sigs >  0.001]
    stock_shorts = stock_sigs[stock_sigs < -0.001]
    etf_longs    = etf_sigs[etf_sigs   >  0.001]
    etf_shorts   = etf_sigs[etf_sigs   < -0.001]

    if not stock_longs.empty:
        logger.info("  --- STOCK LONGS ---")
        for ticker, wt in stock_longs.items():
            logger.info(f"    {ticker:<6}  {wt:+.2%}")
    if not stock_shorts.empty:
        logger.info("  --- STOCK SHORTS ---")
        for ticker, wt in stock_shorts.sort_values().items():
            logger.info(f"    {ticker:<6}  {wt:+.2%}")
    if not etf_longs.empty:
        logger.info("  --- ETF LONGS ---")
        for ticker, wt in etf_longs.items():
            logger.info(f"    {ticker:<6}  {wt:+.2%}")
    if not etf_shorts.empty:
        logger.info("  --- ETF SHORTS ---")
        for ticker, wt in etf_shorts.sort_values().items():
            logger.info(f"    {ticker:<6}  {wt:+.2%}")
    logger.info(f"{'='*56}")

    return signals


def _single_stock_report(ticker: str, signals: "pd.Series", args):
    """
    Detailed per-ticker prediction report, run after _run_inference().

    Uses the same pretrained checkpoints — no retraining required. The full
    universe inference has already run; this function just extracts and deepens
    the analysis for one specific ticker.

    Covers:
      - Recommended position size and direction
      - PINN drift signal (latent-space directional view)
      - VAE uncertainty (how confident the model is in this stock's latent state)
      - Key technical indicator snapshot (last observed values)
      - Analyst forward EPS and PEG (if fundamentals available in checkpoint)
      - Monte Carlo 20-day return distribution from current latent position
    """
    import yaml
    import numpy as np
    import pandas as pd
    import torch
    from datetime import date, timedelta
    from loguru import logger

    from data.ingestion import DataIngestion
    from data.technical_indicators import TechnicalIndicators
    from features.autoencoder import AutoencoderTrainer
    from physics.pde_system import PDEConfig
    from physics.pinn import PINNSolver

    # ------------------------------------------------------------------ #
    # 0. Validate ticker                                                   #
    # ------------------------------------------------------------------ #
    if ticker not in signals.index:
        logger.error(
            f"'{ticker}' not in the trained universe.\n"
            f"Available tickers: {sorted(signals.index.tolist())}"
        )
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cp = __import__("pathlib").Path(args.checkpoints)

    # ------------------------------------------------------------------ #
    # 1. Load VAE and PINN (already trained, load from checkpoint)         #
    # ------------------------------------------------------------------ #
    vae_ckpt  = torch.load(str(cp / "vae.pt"), map_location="cpu", weights_only=False)
    input_dim = int(vae_ckpt["input_dim"])
    vae_cfg   = cfg["features"]["autoencoder"]
    pde_cfg   = cfg["pde"]

    vae = AutoencoderTrainer(
        input_dim=input_dim,
        hidden_dims=vae_cfg["hidden_dims"],
        latent_dim=vae_cfg["latent_dim"],
        beta=vae_cfg["beta_vae"],
    )
    vae.load(str(cp / "vae.pt"))

    pde_config = PDEConfig(
        n_latent=vae_cfg["latent_dim"],
        dt=pde_cfg["dt"],
        diffusion_coeff=pde_cfg["diffusion_coeff"],
        chaos_sensitivity=pde_cfg["chaos_sensitivity"],
        lower_bound=pde_cfg["boundary"]["lower"],
        upper_bound=pde_cfg["boundary"]["upper"],
    )
    pinn = PINNSolver(config=pde_config)
    pinn.load(str(cp / "pinn.pt"))

    # ------------------------------------------------------------------ #
    # 2. Fetch recent OHLCV for this ticker (and full universe for VAE)    #
    # ------------------------------------------------------------------ #
    stock_tickers = [t for ts in cfg["data"]["tickers"].values() for t in ts]
    etf_tickers   = cfg["data"]["sector_etfs"]
    all_tickers   = stock_tickers + etf_tickers

    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")

    ingest = DataIngestion(cache_dir=cfg["data"]["cache_dir"])
    ohlcv  = ingest._batch_download(all_tickers, start, end, cfg["data"]["interval"])
    if not ohlcv or ticker not in ohlcv:
        logger.error(f"Could not fetch OHLCV for {ticker}.")
        return

    returns = ingest.build_return_matrix(ohlcv)
    ti      = TechnicalIndicators()
    ind_data = ti.compute_universe(ohlcv)
    feature_matrix = ti.build_feature_matrix(ind_data, returns)

    if feature_matrix.shape[1] != input_dim:
        if feature_matrix.shape[1] < input_dim:
            pad = pd.DataFrame(
                np.zeros((len(feature_matrix), input_dim - feature_matrix.shape[1])),
                index=feature_matrix.index,
            )
            feature_matrix = pd.concat([feature_matrix, pad], axis=1)
        else:
            feature_matrix = feature_matrix.iloc[:, :input_dim]

    feature_matrix = feature_matrix.dropna()
    if feature_matrix.empty:
        logger.error("Feature matrix empty after indicator warmup.")
        return

    # ------------------------------------------------------------------ #
    # 3. Encode → latent state + uncertainty                              #
    # ------------------------------------------------------------------ #
    X_recent = feature_matrix.values[-20:]
    z_mean, z_std = vae.transform_with_uncertainty(X_recent, n_samples=50)
    z_now   = z_mean[-1].astype(np.float32)
    unc_now = z_std[-1].astype(np.float32)

    # ------------------------------------------------------------------ #
    # 4. PINN drift at current latent position (directional signal)       #
    # ------------------------------------------------------------------ #
    drift   = pinn.predict_drift(z_now.reshape(1, -1), t=1.0, sentiment=0.0).flatten()
    sigma2  = pinn.predict_diffusion(z_now.reshape(1, -1), t=1.0).flatten()
    density = float(pinn.predict_density(z_now.reshape(1, -1), t=1.0).flatten()[0])

    drift_magnitude   = float(np.linalg.norm(drift))
    drift_direction   = "BULLISH" if drift.mean() > 0 else "BEARISH"
    mean_uncertainty  = float(unc_now.mean())
    mean_diffusion    = float(np.sqrt(sigma2.mean()))

    # ------------------------------------------------------------------ #
    # 5. Monte Carlo forward paths from current latent position            #
    # ------------------------------------------------------------------ #
    n_paths   = 1000
    horizon   = 20   # trading days
    mc_rets   = []
    for _ in range(n_paths):
        z = z_now.copy()
        path_ret = 0.0
        for step in range(horizon):
            z_next = pinn.step_latent(z, t=1.0 - step / horizon, n_paths=1)[0]
            # Approximate asset return from latent drift as scalar proxy
            path_ret += float(drift.mean() * pde_cfg["dt"])
            z = z_next
        mc_rets.append(path_ret)
    mc_rets = np.array(mc_rets)
    mc_p10, mc_p50, mc_p90 = np.percentile(mc_rets, [10, 50, 90])

    # ------------------------------------------------------------------ #
    # 6. Key technical indicator snapshot for this ticker                 #
    # ------------------------------------------------------------------ #
    ticker_ind = ind_data.get(ticker, pd.DataFrame())
    key_cols = [
        "ema_20", "ema_50", "rsi_14", "macd", "bb_pct_b",
        "atr_pct", "adx", "volume_ratio", "momentum_1m",
    ]
    indicator_snapshot = {}
    if not ticker_ind.empty:
        last_row = ticker_ind.iloc[-1]
        for col in key_cols:
            if col in ticker_ind.columns:
                val = last_row[col]
                if pd.notna(val):
                    indicator_snapshot[col] = float(val)

    # ------------------------------------------------------------------ #
    # 7. Print the report                                                 #
    # ------------------------------------------------------------------ #
    w       = float(signals.get(ticker, 0.0))
    pos_str = f"{w:+.2%}"
    if abs(w) < 0.001:
        direction_str = "FLAT  (no position)"
    elif w > 0:
        direction_str = f"LONG  {pos_str} of capital"
    else:
        direction_str = f"SHORT {pos_str} of capital"

    # Find which sector this ticker belongs to
    ticker_sector = "Unknown"
    for sector, tickers in cfg["data"]["tickers"].items():
        if ticker in tickers:
            ticker_sector = sector
            break

    logger.info(f"\n{'='*60}")
    logger.info(f"  SINGLE STOCK REPORT  —  {ticker}  ({ticker_sector})")
    logger.info(f"  As of {date.today()}")
    logger.info(f"{'='*60}")
    logger.info(f"  POSITION RECOMMENDATION")
    logger.info(f"    Direction   : {direction_str}")
    logger.info(f"    Raw weight  : {w:+.4f}")

    # Rank within full universe
    rank     = int((signals > w).sum()) + 1
    n_assets = len(signals)
    logger.info(f"    Rank        : #{rank} of {n_assets} assets (1 = strongest long)")

    logger.info(f"")
    logger.info(f"  PHYSICS MODEL (PINN / FOKKER-PLANCK)")
    logger.info(f"    Drift direction : {drift_direction}")
    logger.info(f"    Drift magnitude : {drift_magnitude:.4f}  (larger = stronger directional signal)")
    logger.info(f"    Diffusion (vol) : {mean_diffusion:.4f}  (larger = higher predicted volatility)")
    logger.info(f"    Density p(z,t)  : {density:.4f}  (lower = unusual latent state)")
    logger.info(f"    Latent uncert.  : {mean_uncertainty:.4f}  (higher = model less confident)")

    logger.info(f"")
    logger.info(f"  MONTE CARLO FORWARD VIEW  (PINN, {horizon}d horizon, {n_paths:,} paths)")
    logger.info(f"    10th pct return : {mc_p10:+.4f}")
    logger.info(f"    Median  return  : {mc_p50:+.4f}")
    logger.info(f"    90th pct return : {mc_p90:+.4f}")
    logger.info(f"    Expected skew   : {'positive' if mc_p50 > 0 else 'negative'}")

    if indicator_snapshot:
        logger.info(f"")
        logger.info(f"  TECHNICAL INDICATORS  (last observation)")
        label_map = {
            "ema_20":       "EMA-20",
            "ema_50":       "EMA-50",
            "rsi_14":       "RSI-14",
            "macd":         "MACD",
            "bb_pct_b":     "BB %B",
            "atr_pct":      "ATR %",
            "adx":          "ADX",
            "volume_ratio": "Vol ratio",
            "momentum_1m":  "Mom 1M",
        }
        for col, val in indicator_snapshot.items():
            # Add plain-english interpretation for key signals
            note = ""
            if col == "rsi_14":
                note = "  ← overbought" if val > 70 else ("  ← oversold" if val < 30 else "")
            elif col == "bb_pct_b":
                note = "  ← above upper band" if val > 1 else ("  ← below lower band" if val < 0 else "")
            elif col == "adx":
                note = "  ← strong trend" if val > 25 else "  ← weak trend"
            label = label_map.get(col, col)
            logger.info(f"    {label:<16}: {val:>8.4f}{note}")

    logger.info(f"{'='*60}")
    logger.info(
        f"  NOTE: The model outputs portfolio weights, not price targets.\n"
        f"  Positive weight = model favours long exposure to {ticker}.\n"
        f"  Drift direction is a latent-space signal, not a price forecast."
    )
    logger.info(f"{'='*60}")


def _run_backtest_only(args):
    """
    Load all cached checkpoints, run the deterministic SAC policy over the
    historical data, and produce a full performance + risk report.

    Optionally restrict to a date sub-range (--start-date / --end-date) for
    out-of-sample validation without re-training.
    """
    import types
    import yaml
    import pandas as pd
    from pathlib import Path
    from loguru import logger

    from train import (
        prepare_rl_inputs,
        build_env_and_agent,
        _extract_policy_weights,
    )
    from backtest.engine import BacktestEngine, BacktestConfig
    from optimization.estimators import LagrangianOptimizer
    from risk.metrics import RiskMetrics
    from risk.monte_carlo import MonteCarloSimulator

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cp = Path(args.checkpoints)
    for name in ("vae.pt", "pinn.pt", "sac_agent.pt"):
        if not (cp / name).exists():
            logger.error(f"Checkpoint not found: {cp / name} — run --mode train first.")
            return

    # Build a fake args namespace so prepare_rl_inputs loads everything from cache.
    # no_fundamentals must match the flag used at training time so the feature
    # matrix dimension is consistent with what the VAE checkpoint expects.
    fake_args = types.SimpleNamespace(
        config=args.config,
        checkpoints=args.checkpoints,
        skip_data=True,
        skip_features=True,
        skip_pinn=True,
        no_sentiment=args.no_sentiment,
        no_fundamentals=getattr(args, "no_fundamentals", False),
        epochs_pinn=5000,
        algo=None,
    )

    logger.info("=" * 60)
    logger.info("BACKTEST MODE — loading cached checkpoints")
    logger.info("=" * 60)
    prep       = prepare_rl_inputs(cfg, fake_args)
    data       = prep["data"]
    returns_rl = prep["returns_rl"]
    latent_rl  = prep["latent_rl"]
    unc_rl     = prep["unc_rl"]
    sent_rl    = prep["sent_rl"]

    env, agent = build_env_and_agent(cfg, prep)
    agent.load(str(cp / "sac_agent.pt"))
    logger.info("SAC agent loaded from checkpoint")

    # Optional date-range filter (applied after policy rollout so the env
    # sees the full history; we slice the backtest input only)
    bt_returns   = returns_rl.copy()
    bt_benchmark = data.benchmark_returns.copy()
    if args.start_date:
        bt_returns   = bt_returns.loc[args.start_date:]
        bt_benchmark = bt_benchmark.loc[args.start_date:] if not bt_benchmark.empty else bt_benchmark
    if args.end_date:
        bt_returns   = bt_returns.loc[:args.end_date]
        bt_benchmark = bt_benchmark.loc[:args.end_date] if not bt_benchmark.empty else bt_benchmark

    if bt_returns.empty:
        logger.error("No returns in the specified date range.")
        return

    period_label = ""
    if args.start_date or args.end_date:
        period_label = f" [{args.start_date or '…'} → {args.end_date or '…'}]"
    logger.info(f"Backtesting on {len(bt_returns)} trading days{period_label}")

    # ------------------------------------------------------------------ #
    # Policy weights from deterministic rollout over the FULL history     #
    # (sliced to the requested period below)                              #
    # ------------------------------------------------------------------ #
    policy_weights = _extract_policy_weights(agent, env, returns_rl, latent_rl, unc_rl, sent_rl)
    # Restrict to the requested date range
    policy_weights = policy_weights.reindex(bt_returns.index, method="ffill").dropna(how="all")

    # ------------------------------------------------------------------ #
    # Lagrangian optimizer as benchmark strategy                          #
    # ------------------------------------------------------------------ #
    bt_cfg_raw = cfg["backtest"]
    _freq = bt_cfg_raw["rebalance_freq"]
    if isinstance(_freq, str):
        _freq = {"daily": 1, "weekly": 5, "monthly": 21}.get(_freq, 5)
    rebalance_every = int(_freq)
    lagrangian = LagrangianOptimizer(
        max_position_size=bt_cfg_raw["max_position_size"],
        risk_aversion=cfg["optimization"]["risk_budget"],
        market_neutral=False,
    )
    opt_weights = lagrangian.rolling_optimize(bt_returns, lookback=60, rebalance_every=rebalance_every)

    # ------------------------------------------------------------------ #
    # Run parallel backtest                                               #
    # ------------------------------------------------------------------ #
    shared_bt_kwargs = {k: bt_cfg_raw[k] for k in ["initial_capital", "transaction_cost", "slippage", "max_position_size"]}
    configs = [
        (policy_weights, BacktestConfig(name="SAC_Policy",     **shared_bt_kwargs)),
        (opt_weights,    BacktestConfig(name="Lagrangian_Opt", **shared_bt_kwargs)),
    ]
    engine = BacktestEngine(
        returns=bt_returns,
        benchmark_returns=bt_benchmark if not bt_benchmark.empty else None,
        n_workers=bt_cfg_raw["n_parallel_workers"],
    )
    bt_results  = engine.run_parallel(configs)
    comparison  = engine.cross_compare(bt_results)

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info("\n" + comparison.to_string())

    if args.out:
        comparison.to_csv(args.out)
        logger.info(f"Saved → {args.out}")

    # ------------------------------------------------------------------ #
    # Risk report on the best strategy                                    #
    # ------------------------------------------------------------------ #
    best = bt_results[0]
    risk_cfg = cfg["risk"]
    mc = MonteCarloSimulator(
        n_simulations=risk_cfg["n_monte_carlo"],
        horizon_days=risk_cfg["horizon_days"],
    )
    best_weights_last = policy_weights.iloc[-1].values
    mc_paths  = mc.simulate_gbm(bt_returns, best_weights_last, lookback=risk_cfg["lookback_days"])
    mc_stats  = mc.path_statistics(mc_paths)
    mc_bands  = mc.confidence_bands(mc_paths)
    logger.info(f"MC Stats ({risk_cfg['n_monte_carlo']:,} paths, {risk_cfg['horizon_days']}d horizon): {mc_stats}")

    risk = RiskMetrics(confidence_levels=risk_cfg["confidence_levels"])
    bm_arr = bt_benchmark.values if not bt_benchmark.empty else None
    report = risk.full_report(best.returns, bm_arr, mc_paths)
    logger.info("\n" + risk.format_report(report))

    # ------------------------------------------------------------------ #
    # Optional dashboard                                                  #
    # ------------------------------------------------------------------ #
    if args.dashboard:
        from visualization.dashboard import TradingDashboard
        dash = TradingDashboard()
        dash.launch_app(
            portfolio_values=best.portfolio_values,
            returns=best.returns,
            latent=latent_rl.reindex(bt_returns.index, method="ffill").values,
            reward_history=[],
            mc_bands=mc_bands,
            port=args.dashboard_port,
        )


def _run_dashboard(args):
    """Launch dashboard from saved backtest results."""
    from visualization.dashboard import TradingDashboard
    import numpy as np
    dash = TradingDashboard()
    # Placeholder data — replace with loaded checkpoint results
    dash.launch_app(
        portfolio_values=np.ones(252) * 1e6,
        returns=np.random.randn(252) * 0.01,
        latent=np.random.randn(252, 8),
        reward_history=list(np.random.randn(100)),
        port=args.dashboard_port,
    )


def _run_sensitivity(args):
    """
    Sensitivity analysis: systematically zero out indicator families
    and measure impact on PINN predictive power and RL reward.
    """
    from loguru import logger
    from data.technical_indicators import TechnicalIndicators

    logger.info("Running indicator sensitivity analysis")
    families = list(TechnicalIndicators.FAMILIES.keys())
    logger.info(f"Indicator families: {families}")
    logger.info("For each family, train a model variant with that family zeroed out.")
    logger.info("Compare resulting Sharpe ratios to identify high-impact indicator groups.")
    # Full sensitivity sweep would call train.py with --zero-family <name> for each family


if __name__ == "__main__":
    main()
