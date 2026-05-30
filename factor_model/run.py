"""
Multi-factor equity model — walk-forward backtest.

Pipeline:
  1. S&P 500 universe + GICS sectors (reused from sector_model)
  2. Daily adjusted prices (reused fetch_prices)
  3. SEC EDGAR fundamentals: book equity, net income, assets, gross profit,
     revenue, shares (factor_model/data_fundamentals.py)
  4. Monthly rebalance: compute sector-neutral factor z-scores → composite →
     long-short quintile spread → hold → record
  5. Track each factor's standalone long-short return for attribution

Run from factor_model/ directory:
    python run.py
    python run.py --mode long_only
    python run.py --oos-start 2022-01-01

Benchmark: SPY (for the long-only mode) and zero (for market-neutral long-short).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# Reuse sector_model data infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sector_model"))
from data.universe import fetch_prices  # noqa: E402

# factor_model modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_fundamentals import fetch_factor_fundamentals, fetch_cik_map  # noqa: E402
from factors import compute_factor_scores, FACTORS  # noqa: E402
from construction import long_short_weights, long_only_weights, turnover  # noqa: E402
from universe_pit import (  # noqa: E402
    historical_universe, membership_matrix, fetch_sectors,
    fetch_prices_survivorship_free,
)


def run_factor_model(cfg: dict, out_path: str = "results/backtest.csv") -> pd.DataFrame:
    d   = cfg["data"]
    bt  = cfg["backtest"]
    pf  = cfg["portfolio"]
    cache = d["cache_dir"]

    pit = cfg.get("universe", {}).get("point_in_time", True)

    # ── Universe (point-in-time membership) ───────────────────────────────
    logger.info("═══ Stage 1: S&P 500 universe ═══")
    if pit:
        tickers = historical_universe(d["start_date"], d["end_date"], cache_dir=cache)
        logger.info(f"Historical universe (ever a member 2015-2024): {len(tickers)} tickers")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sector_model"))
        from data.sp500 import fetch_sp500_universe
        tickers = fetch_sp500_universe(cache_dir=cache)["Symbol"].tolist()

    # ── Prices ────────────────────────────────────────────────────────────
    logger.info("═══ Stage 2: Prices ═══")
    if pit:
        # Survivorship-free: keep delisted names (partial history)
        prices = fetch_prices_survivorship_free(
            tickers, d["start_date"], d["end_date"], cache_dir=cache)
    else:
        prices = fetch_prices(tickers, d["start_date"], d["end_date"], cache_dir=cache)
    valid = list(prices.columns)
    logger.info(f"Prices: {prices.shape[1]} of {len(tickers)} tickers have data")

    # Sectors: seed with Wikipedia GICS for current members, look up the rest
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sector_model"))
    from data.sp500 import fetch_sp500_universe
    seed = fetch_sp500_universe(cache_dir=cache).set_index("Symbol")["GICS Sector"].to_dict()
    sectors = fetch_sectors(valid, seed=seed, cache_dir=cache)

    # Point-in-time membership matrix (which stocks were members each day)
    members_mat = membership_matrix(prices.index, cache_dir=cache) if pit else None

    # ── Fundamentals ──────────────────────────────────────────────────────
    logger.info("═══ Stage 3: SEC EDGAR fundamentals ═══")
    cik_map = fetch_cik_map(cache_dir=cache)
    fundamentals = fetch_factor_fundamentals(
        valid, prices.index, cache_dir=f"{cache}/factor_fund", cik_map=cik_map,
    )
    logger.info(f"Fundamentals: {fundamentals.shape}")

    # ── SPY benchmark ─────────────────────────────────────────────────────
    spy = fetch_prices(["SPY"], d["start_date"], d["end_date"], cache_dir=f"{cache}/spy")
    spy_px = spy["SPY"] if "SPY" in spy.columns else spy.iloc[:, 0]
    spy_ret = np.log(spy_px / spy_px.shift(1)).dropna()

    # ── Walk-forward backtest ─────────────────────────────────────────────
    logger.info("═══ Stage 4: Walk-forward backtest ═══")
    rebal_freq = bt["rebalance_freq"]
    warmup     = bt["warmup_days"]
    tcost      = bt["transaction_cost"]
    init_cap   = bt["initial_capital"]
    mode       = pf["mode"]
    quantile   = pf["quantile"]
    comp_factors = cfg.get("factors", {}).get("composite", FACTORS)
    logger.info(f"Composite factors: {comp_factors}")

    dates = prices.index
    rebal_dates = dates[warmup::rebal_freq]

    oos_start = cfg.get("oos_start")
    if oos_start:
        oos_ts = pd.Timestamp(oos_start)
        rebal_dates = rebal_dates[rebal_dates >= oos_ts]
        logger.info(f"OOS mode: {len(rebal_dates)} rebalances from {oos_ts.date()}")

    capital   = float(init_cap)
    prev_w    = pd.Series(dtype=float)
    records   = []
    weight_fn = long_short_weights if mode == "long_short" else long_only_weights

    for i, rd in enumerate(rebal_dates):
        # Point-in-time members on this date
        members = None
        if members_mat is not None and rd in members_mat.index:
            row = members_mat.loc[rd]
            members = set(row.index[row.values])

        scores = compute_factor_scores(
            rd, prices, fundamentals, sectors,
            members=members, composite_factors=comp_factors,
        )
        if scores.empty:
            continue

        w = weight_fn(scores["composite"], quantile=quantile)
        if w.empty:
            continue

        # Transaction cost on turnover
        to = turnover(prev_w, w)
        capital *= (1.0 - to * tcost)

        # Hold until next rebalance
        date_idx = dates.get_loc(rd)
        end_idx  = min(date_idx + rebal_freq, len(dates) - 1)
        fwd = np.log(prices.iloc[end_idx] / prices.iloc[date_idx])  # log returns per stock
        fwd_simple = np.exp(fwd) - 1

        port_ret = float((w * fwd_simple.reindex(w.index).fillna(0)).sum())
        capital *= (1.0 + port_ret)

        # Per-factor standalone long-short return (attribution)
        factor_rets = {}
        for f in FACTORS:
            fw = long_short_weights(scores[f], quantile=quantile)
            factor_rets[f] = float((fw * fwd_simple.reindex(fw.index).fillna(0)).sum()) \
                             if not fw.empty else np.nan

        # IC: rank corr of composite vs forward return
        common = scores.index.intersection(fwd_simple.dropna().index)
        ic = float(scores["composite"].reindex(common).corr(
            fwd_simple.reindex(common), method="spearman")) if len(common) >= 10 else np.nan

        bench = float(spy_ret.iloc[date_idx:end_idx].sum()) if date_idx < len(spy_ret) else 0.0

        rec = {
            "date": rd, "capital": capital, "period_return": port_ret,
            "benchmark_return": bench, "turnover": to, "ic": ic,
            "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum()),
        }
        rec.update({f"ret_{f}": factor_rets[f] for f in FACTORS})
        records.append(rec)
        prev_w = w

    results = pd.DataFrame(records).set_index("date")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)

    # ── Summary ───────────────────────────────────────────────────────────
    r = results["period_return"]
    ppy = 252 / rebal_freq
    total  = capital / init_cap - 1
    sharpe = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 0 else 0
    ic     = results["ic"].dropna()
    logger.info(
        f"DONE [{mode}] | total={total:+.1%} | Sharpe={sharpe:.2f} | "
        f"IC={ic.mean():+.4f} (t={ic.mean()/(ic.std()/len(ic)**0.5):+.2f}) | "
        f"saved → {out_path}"
    )

    # Per-factor Sharpe
    logger.info("Factor long-short Sharpes:")
    for f in FACTORS:
        fr = results[f"ret_{f}"].dropna()
        if len(fr) > 3 and fr.std() > 0:
            logger.info(f"  {f:<10} Sharpe={fr.mean()/fr.std()*np.sqrt(ppy):+.2f}  "
                        f"total={(1+fr).prod()-1:+.1%}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-factor equity model")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--mode", choices=["long_short", "long_only"], default=None)
    p.add_argument("--oos-start", default=None)
    p.add_argument("--out", default="results/backtest.csv")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode:
        cfg["portfolio"]["mode"] = args.mode
    out = args.out
    if args.oos_start:
        cfg["oos_start"] = args.oos_start
        out = args.out.replace(".csv", f"_oos_{args.oos_start[:4]}.csv")

    run_factor_model(cfg, out_path=out)
