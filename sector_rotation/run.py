"""
Sector-rotation sleeve — long-short sector ETFs, dollar-neutral.

Reuses the scoring ingredients of sector_model Layer 1 (12-1 momentum +
theory-grounded macro tilts via _MACRO_BETAS / fetch_macro_data) but builds
a spread portfolio instead of a long-only softmax allocation:

  composite = mom_w * z(momentum_12_1) + macro_w * z(macro_tilt)
  long top n_side ETFs at +1/n, short bottom n_side at -1/n.

Timing: signal uses closes through the rebalance date rd; the position is
entered at rd's close and P&L is measured close(rd) -> close(rd+freq), i.e.
it earns from the next session onward. No same-day information credit
(LOOKAHEAD_FINDING.md rules).

Run from sector_rotation/:
    python run.py
    python run.py --oos-start 2022-01-01
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from loguru import logger

# Reuse Layer 1's macro machinery (betas table + cached VIX/yield fetcher)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sector_model"))
from signals.sector_rotation import _MACRO_BETAS, fetch_macro_data  # noqa: E402


def fetch_etf_prices(tickers, start, end, cache_dir="cache") -> pd.DataFrame:
    p = Path(cache_dir) / f"etf_prices_{start}_{end}.parquet"
    if p.exists():
        logger.info(f"Loading cached ETF prices from {p}")
        return pd.read_parquet(p)
    logger.info(f"Fetching {len(tickers)} ETFs [{start} → {end}]...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.ffill(limit=5)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    prices.to_parquet(p)
    return prices


def run_rotation(cfg: dict, out_path: str = "results/backtest.csv") -> pd.DataFrame:
    d, sig, bt = cfg["data"], cfg["signal"], cfg["backtest"]
    etf_map = cfg["etfs"]                      # sector name -> ticker
    tickers = list(etf_map.values())
    sector_of = {v: k for k, v in etf_map.items()}

    logger.info("═══ Stage 1: Data ═══")
    prices = fetch_etf_prices(tickers + ["SPY"], d["start_date"], d["end_date"], d["cache_dir"])
    spy_px = prices["SPY"]
    px = prices[[t for t in tickers if t in prices.columns]]
    logret = np.log(px / px.shift(1))
    macro = fetch_macro_data(d["start_date"], d["end_date"], cache_dir=d["cache_dir"])

    logger.info("═══ Stage 2: Long-short rotation backtest ═══")
    lb, skip  = sig["momentum_lookback"], sig["momentum_skip"]
    zwin      = sig["macro_z_window"]
    mom_w, mac_w = sig["momentum_weight"], sig["macro_weight"]
    n_side    = sig["n_side"]
    min_secs  = sig["min_sectors"]
    freq      = bt["rebalance_freq"]
    tcost     = bt["transaction_cost"]
    borrow    = bt.get("borrow_rate_annual", 0.003)
    init_cap  = bt["initial_capital"]

    dates = px.index
    rebal_dates = dates[bt["warmup_days"]::freq]
    if cfg.get("oos_start"):
        oos_ts = pd.Timestamp(cfg["oos_start"])
        rebal_dates = rebal_dates[rebal_dates >= oos_ts]
        logger.info(f"OOS mode: {len(rebal_dates)} rebalances from {oos_ts.date()}")

    def zscore(s: pd.Series) -> pd.Series:
        sd = s.std()
        return (s - s.mean()) / sd if sd > 1e-8 else s * 0.0

    capital = float(init_cap)
    prev_w = pd.Series(dtype=float)
    records = []

    for rd in rebal_dates:
        t = dates.get_loc(rd)
        if t < lb:
            continue

        # 12-1 momentum: only ETFs with full lookback history (XLC/XLRE enter late)
        window = logret.iloc[t - lb: t - skip]
        ok = window.notna().mean() > 0.95
        mom = window.sum()[ok[ok].index]
        if len(mom) < min_secs:
            continue

        # Macro tilt: trailing z of yield curve & VIX x theory betas
        hist = macro.loc[:rd].tail(zwin)
        if len(hist) < zwin // 2:
            continue
        row = hist.iloc[-1]
        yc_z  = (row["yield_curve"] - hist["yield_curve"].mean()) / (hist["yield_curve"].std() + 1e-8)
        vix_z = (row["vix"] - hist["vix"].mean()) / (hist["vix"].std() + 1e-8)
        mac = pd.Series({tk: _MACRO_BETAS.get(sector_of[tk], (0.0, 0.0))[0] * yc_z
                             + _MACRO_BETAS.get(sector_of[tk], (0.0, 0.0))[1] * vix_z
                         for tk in mom.index})

        composite = mom_w * zscore(mom) + mac_w * zscore(mac)
        ranked = composite.sort_values(ascending=False)
        longs, shorts = ranked.index[:n_side], ranked.index[-n_side:]
        w = pd.Series(0.0, index=ranked.index)
        w[longs]  = +1.0 / n_side
        w[shorts] = -1.0 / n_side

        # Costs
        all_idx = prev_w.index.union(w.index)
        to = float((w.reindex(all_idx).fillna(0) - prev_w.reindex(all_idx).fillna(0)).abs().sum())
        capital *= (1.0 - to * tcost)

        # Hold freq days: close(rd) -> close(rd+freq)
        ei = min(t + freq, len(dates) - 1)
        fwd = (px.iloc[ei] / px.iloc[t] - 1).reindex(w.index).fillna(0.0)
        port_ret = float((w * fwd).sum())
        borrow_cost = borrow * (ei - t) / 252.0 * float(w[w < 0].abs().sum())
        port_ret -= borrow_cost
        capital *= (1.0 + port_ret)

        spy_ret = float(spy_px.iloc[ei] / spy_px.iloc[t] - 1)
        records.append({
            "date": rd, "capital": capital, "period_return": port_ret,
            "benchmark_return": spy_ret, "turnover": to, "borrow_cost": borrow_cost,
            "n_sectors": len(ranked),
            "longs": "|".join(longs), "shorts": "|".join(shorts),
        })
        prev_w = w

    results = pd.DataFrame(records).set_index("date")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)

    r = results["period_return"]
    ppy = 252 / freq
    sharpe = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 0 else 0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    logger.info(
        f"DONE | total={eq.iloc[-1]-1:+.1%} | Sharpe={sharpe:.2f} | maxDD={dd:+.1%} | "
        f"corr(SPY)={r.corr(results['benchmark_return']):+.2f} | "
        f"avg turnover={results.turnover.mean():.2f} | → {out_path}"
    )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sector-rotation long-short sleeve")
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
    run_rotation(cfg, out_path=out)
