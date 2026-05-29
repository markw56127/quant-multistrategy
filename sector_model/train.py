"""
Sector model training pipeline.

Stages:
  1. Fetch sector universe data (cached per sector after first run)
  2. Rolling OLS decomposition → sector beta + idiosyncratic returns
  3. Fit HMM regime model on sector ETF
  4. Build cross-sectional feature panel (MultiIndex date × ticker)
  5. Walk-forward backtest (fits LightGBM quarterly, rebalances monthly)
  6. Print performance summary; save results

Run from the sector_model/ directory:
  python train.py                                        # default: semis
  python train.py --config config/sectors/it_software.yaml
  python train.py --config config/sectors/financials.yaml
  python train.py --config config/sectors/industrials.yaml

Output is written to results/<sector_name>/backtest.csv unless --out is given.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from data.universe import load_sector_data, fetch_volumes
from data.fundamentals import fetch_fundamental_features
from signals.residuals import rolling_ols_decompose, forward_cross_sectional_excess
from signals.sector import SectorRegimeModel
from signals.cross_section import CrossSectionalModel, CompositeFactorModel, build_features
from backtest.engine import SectorBacktest


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict, out_path: str = "results/backtest.csv") -> pd.DataFrame:
    cache_dir   = cfg["data"].get("cache_dir", "cache/data")
    sector_name = cfg["data"].get("sector_etf", "unknown")

    logger.info(f"═══ Sector: {sector_name} | universe: {len(cfg['data']['universe'])} stocks ═══")

    # 1. Data
    logger.info("── Stage 1: Loading sector data ──")
    stock_ret, sector_ret, prices = load_sector_data(cfg["data"], cache_dir)

    # 2. Decompose
    logger.info("── Stage 2: Rolling OLS decomposition ──")
    sig_cfg = cfg["signals"]
    betas, alphas, residuals = rolling_ols_decompose(
        stock_ret, sector_ret,
        window=sig_cfg["ols_window"],
        min_periods=sig_cfg["ols_min_periods"],
    )
    targets = forward_cross_sectional_excess(stock_ret, horizon=sig_cfg["forward_horizon"])

    # 3. Regime model (fit on full history — HMM parameters are stable across cycle)
    logger.info("── Stage 3: Fitting HMM regime model ──")
    regime_model = SectorRegimeModel(
        n_states=cfg["regime"]["n_states"],
        n_iter=cfg["regime"]["n_iter"],
    )
    regime_model.fit(sector_ret, vol_window=cfg["regime"]["vol_window"])
    regime_proba = regime_model.predict_proba(sector_ret, vol_window=cfg["regime"]["vol_window"])

    # 4. Supplementary data
    logger.info("── Stage 4a: Fetching volumes ──")
    stock_tickers = cfg["data"]["universe"]
    volumes = fetch_volumes(
        tickers=stock_tickers,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        cache_dir=cache_dir,
    )

    logger.info("── Stage 4b: Fetching fundamental features ──")
    fundamentals = fetch_fundamental_features(
        tickers=stock_tickers,
        prices=prices,
        trading_dates=stock_ret.index,
        cache_dir=cache_dir,
    )

    # 4. Features
    logger.info("── Stage 4c: Building cross-sectional feature panel ──")
    features = build_features(
        stock_ret, residuals, betas, regime_proba, sector_ret,
        fundamentals=fundamentals, volumes=volumes,
    )
    logger.info(f"Feature panel: {features.shape[0]} rows × {features.shape[1]} cols")

    # 5. Walk-forward backtest
    logger.info("── Stage 5: Walk-forward backtest ──")
    alpha_model = CrossSectionalModel(cfg["alpha_model"])
    composite_model = CompositeFactorModel(cfg.get("alpha_model", {}))  # Use same config for consistency

    bt_cfg = dict(cfg["backtest"])
    bt_cfg["portfolio"] = cfg["portfolio"]

    bt = SectorBacktest(
        stock_returns=stock_ret,
        residuals=residuals,
        features=features,
        targets=targets,
        sector_returns=sector_ret,
        betas=betas,
        regime_model=regime_model,
        alpha_model=alpha_model,
        cfg=bt_cfg,
        second_model=composite_model,
        second_model_name="composite_factors",
    )
    backtest_result = bt.run()
    
    # Handle both single and dual model results
    if isinstance(backtest_result, tuple):
        results, composite_results = backtest_result
        has_composite = True
    else:
        results = backtest_result
        composite_results = None
        has_composite = False

    # 6. Report
    rebal_freq = cfg["backtest"]["rebalance_freq"]
    stats = SectorBacktest.performance_stats(results, periods_per_year=252 / rebal_freq)

    logger.info("─" * 60)
    logger.info("LIGHTGBM MODEL PERFORMANCE:")
    for k, v in stats.items():
        logger.info(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")
    
    if has_composite:
        composite_stats = SectorBacktest.performance_stats(composite_results, periods_per_year=252 / rebal_freq)
        logger.info("─" * 60)
        logger.info("COMPOSITE FACTORS MODEL PERFORMANCE:")
        for k, v in composite_stats.items():
            logger.info(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")
        
        logger.info("─" * 60)
        logger.info("MODEL COMPARISON:")
        lgbm_excess = stats.get('excess_return', float('nan'))
        comp_excess = composite_stats.get('excess_return', float('nan'))
        lgbm_sharpe = stats.get('annualized_sharpe', float('nan'))
        comp_sharpe = composite_stats.get('annualized_sharpe', float('nan'))
        lgbm_ir = stats.get('information_ratio', float('nan'))
        comp_ir = composite_stats.get('information_ratio', float('nan'))
        lgbm_ic = stats.get('mean_ic', float('nan'))
        comp_ic = composite_stats.get('mean_ic', float('nan'))
        
        logger.info(f"  LightGBM excess return:      {lgbm_excess:.4f}")
        logger.info(f"  Composite excess return:     {comp_excess:.4f}")
        logger.info(f"  LightGBM Sharpe:             {lgbm_sharpe:.4f}")
        logger.info(f"  Composite Sharpe:            {comp_sharpe:.4f}")
        logger.info(f"  LightGBM Information Ratio:  {lgbm_ir:.4f}")
        logger.info(f"  Composite Information Ratio: {comp_ir:.4f}")
        logger.info(f"  LightGBM mean IC:            {lgbm_ic:.4f}")
        logger.info(f"  Composite mean IC:           {comp_ic:.4f}")
    
    logger.info("─" * 60)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)
    logger.info(f"LightGBM results saved to {out_path}")

    if has_composite:
        composite_out_path = out_path.replace(".csv", "_composite.csv")
        composite_results.to_csv(composite_out_path)
        logger.info(f"Composite results saved to {composite_out_path}")

    # Save feature importance from final LightGBM fit
    if alpha_model.model is not None:
        imp = alpha_model.feature_importance()
        logger.info(f"LightGBM Feature importance:\n{imp.to_string()}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sector model training pipeline")
    parser.add_argument("--config", default="config/sectors/semis.yaml")
    parser.add_argument("--out",    default=None,
                        help="Output CSV path. Defaults to results/<sector>/backtest.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.out is None:
        sector = cfg["data"].get("sector_etf", Path(args.config).stem).lower()
        out_path = f"results/{sector}/backtest.csv"
    else:
        out_path = args.out

    run(cfg, out_path=out_path)
