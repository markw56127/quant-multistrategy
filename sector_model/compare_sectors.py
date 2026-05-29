"""
Run the backtest across all configured sectors and print a comparison table.

Usage (from sector_model/ directory):
    python compare_sectors.py
    python compare_sectors.py --sectors semis financials vgt_expanded

Each sector runs independently with its own data cache. Results are written to
results/<config_key>/ so configs with the same ETF don't overwrite each other.
Both LightGBM and composite model results are shown side-by-side.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from train import load_config, run
from backtest.engine import SectorBacktest

SECTOR_CONFIGS = {
    "semis":        "config/sectors/semis.yaml",
    "it_software":  "config/sectors/it_software.yaml",
    "financials":   "config/sectors/financials.yaml",
    "industrials":  "config/sectors/industrials.yaml",
    "vgt_expanded": "config/sectors/vgt_expanded.yaml",
}

DISPLAY_METRICS = [
    "total_return",
    "benchmark_total_return",
    "excess_return",
    "annualized_sharpe",
    "information_ratio",
    "max_drawdown",
    "mean_ic",
    "ic_hit_rate",
    "avg_gross_exposure",
]


def run_sector(name: str, config_path: str):
    logger.info(f"\n{'═'*60}")
    logger.info(f"Running: {name.upper()} ({config_path})")
    logger.info(f"{'═'*60}")

    cfg      = load_config(config_path)
    out_path = f"results/{name}/backtest.csv"   # use config key, not sector_etf

    try:
        results = run(cfg, out_path=out_path)
        rebal_freq = cfg["backtest"]["rebalance_freq"]
        ppy        = 252 / rebal_freq
        lgbm_stats = SectorBacktest.performance_stats(results, periods_per_year=ppy)

        # Load composite results if they were written alongside
        comp_path = out_path.replace(".csv", "_composite.csv")
        comp_stats = {}
        if Path(comp_path).exists():
            comp_df    = pd.read_csv(comp_path, index_col=0, parse_dates=True)
            comp_stats = SectorBacktest.performance_stats(comp_df, periods_per_year=ppy)

        return {"sector": name, "lgbm": lgbm_stats, "composite": comp_stats}
    except Exception as e:
        logger.error(f"{name}: FAILED — {e}")
        return {"sector": name, "lgbm": {}, "composite": {}}


def print_comparison(all_results: list[dict]) -> None:
    if not all_results:
        return

    rows = []
    for r in all_results:
        name = r["sector"]
        for model_label, stats in [("lgbm", r["lgbm"]), ("composite", r["composite"])]:
            if not stats:
                continue
            row = {"sector": f"{name} [{model_label}]"}
            for m in DISPLAY_METRICS:
                val = stats.get(m, float("nan"))
                if isinstance(val, float):
                    row[m] = f"{val:+.3f}" if any(k in m for k in ("return", "drawdown", "ratio", "ic")) \
                             else f"{val:.3f}"
                else:
                    row[m] = str(val)
            rows.append(row)

    df = pd.DataFrame(rows).set_index("sector")
    print("\n" + "═" * 100)
    print("SECTOR × MODEL COMPARISON")
    print("═" * 100)
    print(df.to_string())
    print("═" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare backtest across sectors")
    parser.add_argument(
        "--sectors", nargs="+",
        default=list(SECTOR_CONFIGS.keys()),
        choices=list(SECTOR_CONFIGS.keys()),
        help="Which sectors to run (default: all)",
    )
    args = parser.parse_args()

    all_results = []
    for name in args.sectors:
        config_path = SECTOR_CONFIGS[name]
        if not Path(config_path).exists():
            logger.warning(f"Config not found: {config_path} — skipping {name}")
            continue
        all_results.append(run_sector(name, config_path))

    print_comparison(all_results)
