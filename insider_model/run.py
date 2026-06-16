"""
Insider cluster-buy backtest — survivorship-free, optionally SPY-hedged.

Mechanics (mirrors earnings_model/run.py):
  - At each rebalance, score every point-in-time index member by trailing
    insider cluster-buying: distinct insiders with open-market purchases
    (Form 4 code 'P') filed within the trailing signal_window, STRICTLY
    before the rebalance date.
  - Long the qualifying names (>= min_buyers distinct buyers and >= min_value
    total $), equal weight, capped at max_names.
  - hedge="spy": short SPY 1:1 against the long book → market-neutral-ish
    (beta-1 hedge; residual beta of the names vs SPY remains).
  - Costs: transaction cost on turnover, borrow fee on the SPY short.

Run from insider_model/:
    python run.py
    python run.py --oos-start 2022-01-01
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from universe_pit import (  # noqa: E402
    historical_universe, membership_matrix, fetch_prices_survivorship_free,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from events import build_purchase_events, flag_routine_buyers  # noqa: E402


def run_insider(cfg: dict, out_path: str = "results/backtest.csv") -> pd.DataFrame:
    d, bt, sig, pf = cfg["data"], cfg["backtest"], cfg["signal"], cfg["portfolio"]
    cache = d["cache_dir"]

    # ── Universe (survivorship-free) ──────────────────────────────────────
    logger.info("═══ Stage 1: Survivorship-free universe ═══")
    tickers = historical_universe(d["start_date"], d["end_date"], cache_dir=cache)
    prices = fetch_prices_survivorship_free(tickers, d["start_date"], d["end_date"], cache_dir=cache)
    valid = set(prices.columns)
    members_mat = membership_matrix(prices.index, cache_dir=cache)
    logger.info(f"Universe: {len(valid)} tickers (survivorship-free)")

    # ── Insider purchase events ───────────────────────────────────────────
    logger.info("═══ Stage 2: Form 4 open-market purchases ═══")
    end_year = pd.Timestamp(d["end_date"]).year
    events = build_purchase_events(d["form345_start_year"], end_year, cache_dir=cache)
    events = events[events["ticker"].isin(valid)].copy()
    if sig.get("officer_director_only", True):
        events = events[events["is_officer_director"]]
    events = events[events["value"] >= 0]   # keep zero-price rows for counting; value filter is per-window
    # v1 (2026-06): drop routine same-month-every-year buyers (CMP 2012) —
    # opportunistic trades carry the information content.
    if sig.get("drop_routine", True):
        events = flag_routine_buyers(events)
        events = events[~events["is_routine"]]
    logger.info(f"Events: {len(events)} purchases, {events.ticker.nunique()} tickers")

    # ── SPY (hedge + benchmark) ───────────────────────────────────────────
    spy = fetch_prices_survivorship_free(["SPY"], d["start_date"], d["end_date"],
                                         cache_dir=f"{cache}/spy")
    spy_px = spy["SPY"] if "SPY" in spy.columns else spy.iloc[:, 0]

    # ── Walk-forward backtest ─────────────────────────────────────────────
    logger.info("═══ Stage 3: Cluster-buy backtest ═══")
    rebal_freq  = bt["rebalance_freq"]
    warmup      = bt["warmup_days"]
    tcost       = bt["transaction_cost"]
    borrow_rate = bt.get("borrow_rate_annual", 0.003)
    init_cap    = bt["initial_capital"]
    window      = sig["signal_window"]
    min_buyers  = sig["min_buyers"]
    min_value   = sig["min_value"]
    max_names   = sig["max_names"]
    hedge       = pf.get("hedge", "spy")
    min_names   = pf["min_names"]

    dates = prices.index
    rebal_dates = dates[warmup::rebal_freq]
    if cfg.get("oos_start"):
        oos_ts = pd.Timestamp(cfg["oos_start"])
        rebal_dates = rebal_dates[rebal_dates >= oos_ts]
        logger.info(f"OOS mode: {len(rebal_dates)} rebalances from {oos_ts.date()}")

    capital = float(init_cap)
    prev_w = pd.Series(dtype=float)
    records = []

    for rd in rebal_dates:
        win_start = dates[max(0, dates.get_loc(rd) - window)]
        # STRICTLY before rd: filings can land after the close (PEAD lesson)
        elig = events[(events["filed"] > win_start) & (events["filed"] < rd)]
        if elig.empty:
            continue

        members = set(members_mat.loc[rd].pipe(lambda r: r.index[r.values])) \
                  if rd in members_mat.index else valid
        last_px = prices.loc[:rd].iloc[-1]
        tradable = set(last_px.dropna().index)
        elig = elig[elig["ticker"].isin(members & tradable)]

        score = (elig.groupby("ticker")
                     .agg(n_buyers=("owner_cik", "nunique"), value=("value", "sum")))
        score = score[(score["n_buyers"] >= min_buyers) & (score["value"] >= min_value)]
        if len(score) < min_names:
            continue
        score = score.sort_values(["n_buyers", "value"], ascending=False).head(max_names)

        w = pd.Series(1.0 / len(score), index=score.index)

        # Turnover cost (long book only; SPY hedge turnover is negligible)
        all_idx = prev_w.index.union(w.index)
        to = float((w.reindex(all_idx).fillna(0) - prev_w.reindex(all_idx).fillna(0)).abs().sum())
        capital *= (1.0 - to * tcost)

        # Hold rebal_freq days
        di = dates.get_loc(rd)
        ei = min(di + rebal_freq, len(dates) - 1)
        fwd = (prices.iloc[ei] / prices.iloc[di] - 1).reindex(w.index).fillna(0.0)
        long_ret = float((w * fwd).sum())
        spy_ret = float(spy_px.iloc[ei] / spy_px.iloc[di] - 1) if len(spy_px) > ei else 0.0

        if hedge == "spy":
            borrow = borrow_rate * (ei - di) / 252.0          # on 1.0 short notional
            port_ret = long_ret - spy_ret - borrow
        else:
            borrow = 0.0
            port_ret = long_ret
        capital *= (1.0 + port_ret)

        records.append({
            "date": rd, "capital": capital, "period_return": port_ret,
            "long_return": long_ret, "benchmark_return": spy_ret,
            "borrow_cost": borrow, "turnover": to,
            "n_names": len(score),
            "avg_buyers": float(score["n_buyers"].mean()),
            "total_value": float(score["value"].sum()),
        })
        prev_w = w

    results = pd.DataFrame(records).set_index("date")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)

    r = results["period_return"]
    ppy = 252 / rebal_freq
    tot = capital / init_cap - 1
    sharpe = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 0 else 0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    corr_bm = float(r.corr(results["benchmark_return"]))
    logger.info(
        f"DONE | total={tot:+.1%} | Sharpe={sharpe:.2f} | maxDD={dd:+.1%} | "
        f"corr(SPY)={corr_bm:+.2f} | avg book={results['n_names'].mean():.0f} | → {out_path}"
    )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Insider cluster-buy model")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--oos-start", default=None)
    p.add_argument("--out", default="results/backtest.csv")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = args.out
    if args.oos_start:
        cfg["oos_start"] = args.oos_start
        out = out.replace(".csv", f"_oos_{args.oos_start[:4]}.csv")
    run_insider(cfg, out_path=out)
