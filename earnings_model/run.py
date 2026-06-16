"""
PEAD backtest — event-driven, survivorship-free, market-neutral.

Mechanics:
  - At each rebalance (every `rebalance_freq` trading days), the book holds
    every stock whose most recent earnings announcement falls within the
    trailing `drift_window` days — i.e. stocks still inside their post-earnings
    drift period.
  - Among that eligible set (restricted to point-in-time index members that are
    actually trading), rank by SUE; go long the top quantile, short the bottom
    quantile, equal-weighted, dollar-neutral.
  - A stock enters the book at its announcement and ages out `drift_window`
    days later — the standard PEAD holding pattern.

Survivorship-free from day one (shared/universe_pit.py): historical S&P 500
membership + delisted-name prices. This is the lesson from sector_model, whose
apparent alpha was mostly survivorship bias.

Run from earnings_model/ directory:
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
from events import build_events, fetch_cik_map  # noqa: E402


def run_pead(cfg: dict, out_path: str = "results/backtest.csv") -> pd.DataFrame:
    d, bt, pf = cfg["data"], cfg["backtest"], cfg["portfolio"]
    cache = d["cache_dir"]

    # ── Universe (survivorship-free) ──────────────────────────────────────
    logger.info("═══ Stage 1: Survivorship-free universe ═══")
    tickers = historical_universe(d["start_date"], d["end_date"], cache_dir=cache)
    prices = fetch_prices_survivorship_free(tickers, d["start_date"], d["end_date"], cache_dir=cache)
    valid = list(prices.columns)
    members_mat = membership_matrix(prices.index, cache_dir=cache)
    logger.info(f"Universe: {len(valid)} tickers (survivorship-free)")

    # ── Earnings events ───────────────────────────────────────────────────
    logger.info("═══ Stage 2: Earnings-surprise events (SUE) ═══")
    cik_map = fetch_cik_map(cache_dir=cache)
    events = build_events(valid, cache_dir=cache, cik_map=cik_map, end_date=d["end_date"])
    events = events[events["ticker"].isin(valid)].copy()
    logger.info(f"Events: {len(events)} announcements, {events.ticker.nunique()} tickers")

    # ── SPY benchmark ─────────────────────────────────────────────────────
    spy = fetch_prices_survivorship_free(["SPY"], d["start_date"], d["end_date"],
                                         cache_dir=f"{cache}/spy")
    spy_px = spy["SPY"] if "SPY" in spy.columns else spy.iloc[:, 0]
    spy_ret = np.log(spy_px / spy_px.shift(1)).dropna()

    # ── Walk-forward event-driven backtest ────────────────────────────────
    logger.info("═══ Stage 3: PEAD backtest ═══")
    rebal_freq   = bt["rebalance_freq"]
    drift_window = bt["drift_window"]          # trading days a stock stays in book
    warmup       = bt["warmup_days"]
    tcost        = bt["transaction_cost"]
    borrow_rate  = bt.get("borrow_rate_annual", 0.01)  # annualised fee on short notional
    init_cap     = bt["initial_capital"]
    quantile     = pf["quantile"]
    min_names    = pf["min_names"]

    dates = prices.index
    rebal_dates = dates[warmup::rebal_freq]
    oos_start = cfg.get("oos_start")
    if oos_start:
        oos_ts = pd.Timestamp(oos_start)
        rebal_dates = rebal_dates[rebal_dates >= oos_ts]
        logger.info(f"OOS mode: {len(rebal_dates)} rebalances from {oos_ts.date()}")

    capital = float(init_cap)
    prev_w  = pd.Series(dtype=float)
    records = []

    for rd in rebal_dates:
        win_start = dates[max(0, dates.get_loc(rd) - drift_window)]
        # Eligible: announced within the drift window, STRICTLY before rd.
        # (Fixed 2026-06: was `<= rd`. ann_date is the SEC filing date; filings
        # often land after the close, so trading the same day's close on that
        # information is not executable. Strict `<` guarantees the information
        # is public for at least one full session before we trade it.)
        elig = events[(events["ann_date"] > win_start) & (events["ann_date"] < rd)]
        if elig.empty:
            continue
        # Most recent announcement per ticker within the window
        elig = elig.sort_values("ann_date").drop_duplicates("ticker", keep="last")

        # Point-in-time members, trading on rd
        members = set(members_mat.loc[rd].pipe(lambda r: r.index[r.values])) \
                  if rd in members_mat.index else set(valid)
        last_px = prices.loc[:rd].iloc[-1]
        tradable = set(last_px.dropna().index)
        elig = elig[elig["ticker"].isin(members & tradable)]
        if len(elig) < min_names:
            continue

        # Rank by SUE → long top quantile, short bottom quantile
        sue = elig.set_index("ticker")["sue"].sort_values(ascending=False)
        n_side = max(1, int(len(sue) * quantile))
        longs, shorts = sue.index[:n_side], sue.index[-n_side:]
        w = pd.Series(0.0, index=sue.index)
        w[longs]  = +1.0 / n_side
        w[shorts] = -1.0 / n_side

        # Turnover cost
        all_idx = prev_w.index.union(w.index)
        to = float((w.reindex(all_idx).fillna(0) - prev_w.reindex(all_idx).fillna(0)).abs().sum())
        capital *= (1.0 - to * tcost)

        # Hold rebal_freq days
        di = dates.get_loc(rd)
        ei = min(di + rebal_freq, len(dates) - 1)
        fwd = (prices.iloc[ei] / prices.iloc[di] - 1).reindex(w.index).fillna(0.0)
        port_ret = float((w * fwd).sum())

        # Borrow fee on the short book (added 2026-06): annualised rate charged
        # pro-rata over the holding period on gross short notional. 1%/yr is a
        # general-collateral assumption; hard-to-borrow names cost more.
        short_gross = float(w[w < 0].abs().sum())
        borrow_cost = borrow_rate * (ei - di) / 252.0 * short_gross
        port_ret -= borrow_cost
        capital *= (1.0 + port_ret)

        bench = float(spy_ret.iloc[di:ei].sum()) if di < len(spy_ret) else 0.0
        records.append({
            "date": rd, "capital": capital, "period_return": port_ret,
            "borrow_cost": borrow_cost,
            "benchmark_return": bench, "turnover": to,
            "n_long": len(longs), "n_short": len(shorts), "n_eligible": len(sue),
            "avg_sue_long": float(sue[longs].mean()), "avg_sue_short": float(sue[shorts].mean()),
        })
        prev_w = w

    results = pd.DataFrame(records).set_index("date")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)

    r = results["period_return"]
    ppy = 252 / rebal_freq
    tot = capital / init_cap - 1
    sharpe = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 0 else 0
    dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    logger.info(
        f"DONE | total={tot:+.1%} | Sharpe={sharpe:.2f} | maxDD={dd:+.1%} | "
        f"avg book={results[['n_long','n_short']].sum(axis=1).mean():.0f} | → {out_path}"
    )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PEAD earnings-drift model")
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
    run_pead(cfg, out_path=out)
