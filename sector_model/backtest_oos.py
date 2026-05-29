"""
Out-of-sample backtest: train on 2015-2023, test on 2024.

This script validates model generalization by:
  1. Training on historical data (2015-2023)
  2. Testing on completely unseen data (2024)
  3. Comparing in-sample vs OOS metrics
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from data.universe import load_sector_data
from data.fundamentals import fetch_fundamental_features
from signals.residuals import rolling_ols_decompose, forward_idiosyncratic_return
from signals.sector import SectorRegimeModel
from signals.cross_section import CrossSectionalModel, build_features
from backtest.engine import SectorBacktest


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_oos_backtest(cfg: dict, train_end: str = "2023-12-31", oos_out: str = "results/backtest_oos.csv") -> dict:
    """
    Run out-of-sample backtest.

    Args:
        cfg: Config dict
        train_end: Last date to include in training (inclusive)
        oos_out: Path to save OOS results

    Returns:
        Dict with keys: 'is_stats', 'oos_stats', 'is_results', 'oos_results'
    """
    cache_dir = cfg["data"].get("cache_dir", "cache/data")

    logger.info("=" * 60)
    logger.info("OUT-OF-SAMPLE BACKTEST: Train on 2015-2023, Test on 2024")
    logger.info("=" * 60)

    # 1. Load all data
    logger.info("── Stage 1: Loading sector data ──")
    stock_ret, sector_ret, prices = load_sector_data(cfg["data"], cache_dir)

    # 2. Decompose on full data (needed for feature engineering)
    logger.info("── Stage 2: Rolling OLS decomposition (full history) ──")
    sig_cfg = cfg["signals"]
    betas, alphas, residuals = rolling_ols_decompose(
        stock_ret, sector_ret,
        window=sig_cfg["ols_window"],
        min_periods=sig_cfg["ols_min_periods"],
    )
    targets = forward_idiosyncratic_return(residuals, horizon=sig_cfg["forward_horizon"])

    # 3. Regime model — fit on IS data only for feature panel.
    #    Trading exposure decisions use a rolling refit inside the engine loop.
    logger.info("── Stage 3: Fitting HMM regime model (in-sample only) ──")
    train_end_dt = pd.Timestamp(train_end)
    regime_model = SectorRegimeModel(
        n_states=cfg["regime"]["n_states"],
        n_iter=cfg["regime"]["n_iter"],
    )
    regime_model.fit(sector_ret.loc[:train_end_dt], vol_window=cfg["regime"]["vol_window"])
    regime_proba = regime_model.predict_proba(sector_ret, vol_window=cfg["regime"]["vol_window"])

    # 4. Fundamentals
    logger.info("── Stage 4a: Fetching fundamental features ──")
    stock_tickers = cfg["data"]["universe"]
    fundamentals = fetch_fundamental_features(
        tickers=stock_tickers,
        prices=prices,
        trading_dates=stock_ret.index,
        cache_dir=cache_dir,
    )

    # 5. Features
    logger.info("── Stage 4b: Building cross-sectional feature panel ──")
    features = build_features(stock_ret, residuals, betas, regime_proba, sector_ret, fundamentals)
    logger.info(f"Feature panel: {features.shape[0]} rows × {features.shape[1]} cols")

    # Split data
    is_dates = stock_ret.index[stock_ret.index <= train_end_dt]
    oos_dates = stock_ret.index[stock_ret.index > train_end_dt]

    logger.info(f"── In-sample: {is_dates[0].date()} to {is_dates[-1].date()}")
    logger.info(f"── Out-of-sample: {oos_dates[0].date()} to {oos_dates[-1].date()}")

    # 6. In-sample backtest (walk-forward on training data)
    logger.info("── Stage 5a: In-sample walk-forward backtest ──")
    alpha_model_is = CrossSectionalModel(cfg["alpha_model"])

    bt_cfg = dict(cfg["backtest"])
    bt_cfg["portfolio"] = cfg["portfolio"]

    bt_is = SectorBacktest(
        stock_returns=stock_ret.loc[:train_end_dt],
        residuals=residuals.loc[:train_end_dt],
        features=features.loc[:train_end_dt],
        targets=targets.loc[:train_end_dt],
        sector_returns=sector_ret.loc[:train_end_dt],
        betas=betas.loc[:train_end_dt],
        regime_model=regime_model,
        alpha_model=alpha_model_is,
        cfg=bt_cfg,
    )
    results_is = bt_is.run()

    rebal_freq = cfg["backtest"]["rebalance_freq"]
    stats_is = SectorBacktest.performance_stats(results_is, periods_per_year=252 / rebal_freq)

    # 7. Out-of-sample backtest
    # Train final alpha model ONLY on in-sample data, then test on OOS
    logger.info("── Stage 5b: Out-of-sample backtest (test only) ──")
    alpha_model_oos = CrossSectionalModel(cfg["alpha_model"])

    # Train on full in-sample period
    alpha_model_oos.fit(features.loc[:train_end_dt], targets.loc[:train_end_dt], is_dates[0], is_dates[-1])

    # Create OOS backtest engine with frozen alpha model
    # Use shorter train_window for OOS since we have less data
    oos_cfg = dict(bt_cfg)
    oos_cfg["train_window"] = 20  # 1 month warmup instead of 2 years
    oos_cfg["retrain_every_n"] = 999  # Don't retrain on OOS data

    bt_oos = SectorBacktest(
        stock_returns=stock_ret.loc[train_end_dt:],
        residuals=residuals.loc[train_end_dt:],
        features=features.loc[train_end_dt:],
        targets=targets.loc[train_end_dt:],
        sector_returns=sector_ret.loc[train_end_dt:],
        betas=betas.loc[train_end_dt:],
        regime_model=regime_model,
        alpha_model=alpha_model_oos,
        cfg=oos_cfg,
    )
    results_oos = bt_oos.run()

    stats_oos = SectorBacktest.performance_stats(results_oos, periods_per_year=252 / rebal_freq)

    # 8. Report
    logger.info("─" * 60)
    logger.info(f"IN-SAMPLE PERFORMANCE (2015-{train_end[:4]}):")
    for k, v in stats_is.items():
        logger.info(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    logger.info("─" * 60)
    logger.info(f"OUT-OF-SAMPLE PERFORMANCE ({int(train_end[:4])+1}-2024):")
    for k, v in stats_oos.items():
        logger.info(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    logger.info("─" * 60)
    logger.info("COMPARISON (OOS - IS):")
    for k in stats_is.keys():
        is_val = stats_is.get(k, 0)
        oos_val = stats_oos.get(k, 0)
        if isinstance(is_val, float) and isinstance(oos_val, float):
            diff = oos_val - is_val
            pct = (diff / is_val * 100) if is_val != 0 else 0
            logger.info(f"  {k:30s}: {diff:+.4f} ({pct:+.1f}%)")
    logger.info("─" * 60)

    # Save results
    Path(oos_out).parent.mkdir(parents=True, exist_ok=True)
    results_oos.to_csv(oos_out)
    logger.info(f"OOS results saved to {oos_out}")

    # Feature importance from OOS model
    if alpha_model_oos.model is not None:
        imp = alpha_model_oos.feature_importance()
        logger.info(f"OOS Feature importance:\n{imp.to_string()}")

    return {
        "is_stats": stats_is,
        "oos_stats": stats_oos,
        "is_results": results_is,
        "oos_results": results_oos,
    }


if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    run_oos_backtest(cfg, train_end="2021-12-31", oos_out="results/backtest_oos.csv")
