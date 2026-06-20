"""
Small-cap multi-factor sleeve — the same engine as factor_model, a different pond.

Thesis under test (the repo's recurring lesson, inverted): every equity result here
has been marginal-to-dead because US large-cap is the most arbitraged market on earth.
Factor premia — value, quality, size especially — are documented to be STRONGER where
there is less analyst coverage and slower arbitrage, i.e. in small-caps. If the method
was never the problem and the *pond* was, the identical machinery should produce a
cleaner cross-sectional signal on the S&P 600 than it did on the S&P 500.

Identical to factor_model by construction: same sector-neutral factor z-scores
(compute_factor_scores), same EDGAR fundamentals pipeline, same long-short quintile
construction, same costs framework, same dev/OOS split. The ONLY change is the
universe (S&P 600 small-cap) — so any difference in result is a pond effect, not a
method effect.

SURVIVORSHIP: current S&P 600 constituents (no free PIT small-cap membership). Prices
are kept survivorship-free-style (partial history retained). Judge this on IC and the
L/S spread (ranking-robust), NOT absolute return, which is optimistic here. See
universe.py for the full caveat.

Run from smallcap_factor/ directory:
    python run.py
    python run.py --oos-start 2025-01-01
"""

import argparse
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from universe_pit import fetch_prices_survivorship_free, fetch_sectors  # noqa: E402
from data_fundamentals import fetch_factor_fundamentals, fetch_cik_map  # noqa: E402
from factors import compute_factor_scores, FACTORS  # noqa: E402
from construction import long_short_weights, turnover  # noqa: E402
from universe import fetch_sp600_universe  # noqa: E402


def run_smallcap(cfg: dict, out_path: str = "results/backtest.csv") -> pd.DataFrame:
    d, bt, pf = cfg["data"], cfg["backtest"], cfg["portfolio"]
    cache = d["cache_dir"]
    comp = cfg.get("factors", {}).get("composite", FACTORS)

    logger.info("═══ Stage 1: S&P 600 small-cap universe ═══")
    uni = fetch_sp600_universe(cache_dir=cache)
    tickers = uni["Symbol"].tolist()

    logger.info("═══ Stage 2: Prices (survivorship-free-style) ═══")
    prices = fetch_prices_survivorship_free(tickers, d["start_date"], d["end_date"], cache_dir=cache)
    valid = list(prices.columns)
    logger.info(f"Prices: {len(valid)} of {len(tickers)} small-cap names have data")

    seed = uni.set_index("Symbol")["GICS Sector"].to_dict()
    sectors = fetch_sectors(valid, seed=seed, cache_dir=cache)

    logger.info("═══ Stage 3: SEC EDGAR fundamentals ═══")
    cik_map = fetch_cik_map(cache_dir=cache)
    fundamentals = fetch_factor_fundamentals(
        valid, prices.index, cache_dir=f"{cache}/factor_fund", cik_map=cik_map)
    logger.info(f"Fundamentals: {fundamentals.shape}")

    logger.info("═══ Stage 4: Walk-forward backtest ═══")
    rebal_freq, warmup = bt["rebalance_freq"], bt["warmup_days"]
    tcost, borrow_rate = bt["transaction_cost"], bt["borrow_rate_annual"]
    init_cap, quantile = bt["initial_capital"], pf["quantile"]

    dates = prices.index
    rebal_dates = dates[warmup::rebal_freq]
    oos_start = cfg.get("oos_start")
    if oos_start:
        oos_ts = pd.Timestamp(oos_start)
        rebal_dates = rebal_dates[rebal_dates >= oos_ts]
        logger.info(f"OOS mode: {len(rebal_dates)} rebalances from {oos_ts.date()}")

    capital = float(init_cap)
    prev_w = pd.Series(dtype=float)
    records = []

    for rd in rebal_dates:
        scores = compute_factor_scores(rd, prices, fundamentals, sectors,
                                       members=None, composite_factors=comp)
        if scores.empty:
            continue
        w = long_short_weights(scores["composite"], quantile=quantile)
        if w.empty:
            continue

        to = turnover(prev_w, w)
        capital *= (1.0 - to * tcost)

        di = dates.get_loc(rd)
        end = min(di + rebal_freq, len(dates) - 1)
        fwd = (prices.iloc[end] / prices.iloc[di] - 1.0)
        port_ret = float((w * fwd.reindex(w.index).fillna(0)).sum())

        short_gross = float(w[w < 0].abs().sum())
        borrow_cost = borrow_rate * (end - di) / 252.0 * short_gross
        port_ret -= borrow_cost
        capital *= (1.0 + port_ret)

        factor_rets = {}
        for f in FACTORS:
            fw = long_short_weights(scores[f], quantile=quantile)
            factor_rets[f] = float((fw * fwd.reindex(fw.index).fillna(0)).sum()) if not fw.empty else np.nan

        common = scores.index.intersection(fwd.dropna().index)
        ic = float(scores["composite"].reindex(common).corr(
            fwd.reindex(common), method="spearman")) if len(common) >= 10 else np.nan

        rec = {"date": rd, "capital": capital, "period_return": port_ret,
               "borrow_cost": borrow_cost, "turnover": to, "ic": ic,
               "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum())}
        rec.update({f"ret_{f}": factor_rets[f] for f in FACTORS})
        records.append(rec)
        prev_w = w

    results = pd.DataFrame(records).set_index("date")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)

    # ── Summary ──
    r = results["period_return"]
    ppy = 252 / rebal_freq
    total = capital / init_cap - 1
    sharpe = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 0 else 0.0
    eq = (1 + r).cumprod()
    maxdd = float((eq / eq.cummax() - 1).min())
    ic = results["ic"].dropna()
    ic_t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if len(ic) > 1 else np.nan
    logger.info(
        f"DONE | total={total:+.1%} | Sharpe={sharpe:.2f} | vol={r.std()*np.sqrt(ppy):.1%} | "
        f"maxDD={maxdd:+.1%} | IC={ic.mean():+.4f} (t={ic_t:+.2f}) | → {out_path}")
    logger.info("Factor long-short Sharpes (small-cap):")
    for f in FACTORS:
        fr = results[f"ret_{f}"].dropna()
        if len(fr) > 3 and fr.std() > 0:
            logger.info(f"  {f:<10} Sharpe={fr.mean()/fr.std()*np.sqrt(ppy):+.2f}  "
                        f"total={(1+fr).prod()-1:+.1%}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Small-cap multi-factor sleeve")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--oos-start", default=None)
    p.add_argument("--out", default="results/backtest.csv")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = args.out
    if args.oos_start:
        cfg["oos_start"] = args.oos_start
        out = args.out.replace(".csv", f"_oos_{args.oos_start[:4]}.csv")
    run_smallcap(cfg, out_path=out)
