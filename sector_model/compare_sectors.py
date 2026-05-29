"""
Run the backtest across all configured sectors and print a comparison table.

Usage (from sector_model/ directory):
    python compare_sectors.py
    python compare_sectors.py --sectors semis it_software
    python compare_sectors.py --train-end 2021-12-31  # OOS split

Each sector runs independently with its own data cache, so runs can be
parallelised manually by opening separate terminals.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from train import load_config, run
from backtest.engine import SectorBacktest

SECTOR_CONFIGS = {
    "semis":        "config/sectors/semis.yaml",
    "it_software":  "config/sectors/it_software.yaml",
    "financials":   "config/sectors/financials.yaml",
    "industrials":  "config/sectors/industrials.yaml",
}

# Metrics to show in the comparison table, in order
DISPLAY_METRICS = [
    "total_return",
    "benchmark_total_return",
    "excess_return",
    "annualized_return",
    "annualized_sharpe",
    "information_ratio",
    "max_drawdown",
    "mean_ic",
    "ic_hit_rate",
    "avg_gross_exposure",
    "total_cost",
]


def run_sector(name: str, config_path: str) -> dict:
    logger.info(f"\n{'═'*60}")
    logger.info(f"Running sector: {name.upper()} ({config_path})")
    logger.info(f"{'═'*60}")

    cfg      = load_config(config_path)
    sector   = cfg["data"].get("sector_etf", name).lower()
    out_path = f"results/{sector}/backtest.csv"

    try:
        results = run(cfg, out_path=out_path)
        rebal_freq = cfg["backtest"]["rebalance_freq"]
        stats = SectorBacktest.performance_stats(results, periods_per_year=252 / rebal_freq)
        stats["sector"] = name
        return stats
    except Exception as e:
        logger.error(f"{name}: FAILED — {e}")
        return {"sector": name}


def print_comparison(all_stats: list[dict]) -> None:
    if not all_stats:
        return

    rows = []
    for s in all_stats:
        row = {"sector": s.get("sector", "?")}
        for m in DISPLAY_METRICS:
            val = s.get(m, float("nan"))
            if isinstance(val, float):
                row[m] = f"{val:+.3f}" if "return" in m or "drawdown" in m or "ratio" in m or "ic" in m \
                         else f"{val:.3f}"
            else:
                row[m] = str(val)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("sector")

    print("\n" + "═" * 80)
    print("SECTOR COMPARISON")
    print("═" * 80)
    print(df.to_string())
    print("═" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare backtest across sectors")
    parser.add_argument(
        "--sectors", nargs="+",
        default=list(SECTOR_CONFIGS.keys()),
        choices=list(SECTOR_CONFIGS.keys()),
        help="Which sectors to run (default: all)",
    )
    args = parser.parse_args()

    all_stats = []
    for name in args.sectors:
        config_path = SECTOR_CONFIGS[name]
        if not Path(config_path).exists():
            logger.warning(f"Config not found: {config_path} — skipping {name}")
            continue
        stats = run_sector(name, config_path)
        all_stats.append(stats)

    print_comparison(all_stats)
