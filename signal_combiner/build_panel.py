"""
Build the cross-sectional factor panel for the linear-vs-XGBoost bake-off.

Reuses factor_model's exact machinery (same survivorship-free universe, same
EDGAR fundamentals, same sector-neutral z-scores) so the bake-off compares
*combiners* on identical inputs — not a different feature set. At each monthly
rebalance we record, per stock:

    features : value, momentum, quality, low_vol, size   (sector-neutral z-scores)
    target   : fwd_ret  — simple return over the next rebalance period

The panel is cached to cache/panel.parquet; pass rebuild=True to regenerate.

Lookahead note: features at date t use only data through t (compute_factor_scores
is the same point-in-time routine the factor sleeve trades on); the target is the
return EARNED from t to t+1, never seen at t. The bake-off's walk-forward CV then
adds a purge+embargo so a training label cannot overlap the test month.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sector_model"))
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "factor_model"))

from universe_pit import (  # noqa: E402
    historical_universe, membership_matrix, fetch_sectors, fetch_prices_survivorship_free,
)
from data_fundamentals import fetch_factor_fundamentals, fetch_cik_map  # noqa: E402
from factors import compute_factor_scores, FACTORS  # noqa: E402


def build_panel(rebuild: bool = False) -> pd.DataFrame:
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent / "config.yaml"))
    cache = cfg["cache_dir"]            # reuses factor_model/cache
    panel_path = Path(__file__).resolve().parent / "cache" / "panel.parquet"
    if panel_path.exists() and not rebuild:
        logger.info(f"Loading cached panel from {panel_path}")
        return pd.read_parquet(panel_path)

    d, bt = cfg["data"], cfg["backtest"]
    logger.info("═══ Building factor panel (reusing factor_model inputs) ═══")
    tickers = historical_universe(d["start_date"], d["end_date"], cache_dir=cache)
    prices = fetch_prices_survivorship_free(tickers, d["start_date"], d["end_date"], cache_dir=cache)
    valid = list(prices.columns)

    from data.sp500 import fetch_sp500_universe  # noqa: E402
    seed = fetch_sp500_universe(cache_dir=cache).set_index("Symbol")["GICS Sector"].to_dict()
    sectors = fetch_sectors(valid, seed=seed, cache_dir=cache)
    members_mat = membership_matrix(prices.index, cache_dir=cache)

    cik_map = fetch_cik_map(cache_dir=cache)
    fundamentals = fetch_factor_fundamentals(
        valid, prices.index, cache_dir=f"{cache}/factor_fund", cik_map=cik_map)

    dates = prices.index
    rebal_dates = dates[bt["warmup_days"]::bt["rebalance_freq"]]
    logger.info(f"{len(rebal_dates)} monthly rebalances "
                f"[{rebal_dates[0].date()} → {rebal_dates[-1].date()}]")

    rows = []
    for rd in rebal_dates:
        members = None
        if rd in members_mat.index:
            row = members_mat.loc[rd]
            members = set(row.index[row.values])
        scores = compute_factor_scores(rd, prices, fundamentals, sectors,
                                       members=members, composite_factors=FACTORS)
        if scores.empty:
            continue
        di = dates.get_loc(rd)
        end = min(di + bt["rebalance_freq"], len(dates) - 1)
        fwd = (prices.iloc[end] / prices.iloc[di] - 1.0).reindex(scores.index)

        block = scores[FACTORS].copy()
        block["fwd_ret"] = fwd
        block["date"] = rd
        block["ticker"] = block.index
        rows.append(block.dropna(subset=["fwd_ret"]))

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=FACTORS, how="all")
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path)
    logger.info(f"Panel: {len(panel):,} stock-months, "
                f"{panel['date'].nunique()} months, {panel['ticker'].nunique()} names "
                f"→ {panel_path}")
    return panel


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    build_panel(rebuild=ap.parse_args().rebuild)
