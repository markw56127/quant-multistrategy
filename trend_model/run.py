"""
Time-series momentum (trend-following) sleeve — cross-asset futures.

Why this sleeve exists
----------------------
Every equity strategy in this repo competes in US large-cap, the most efficient
pond on earth, and the true-OOS test (OOS_FINDING.md) showed the equity-factor
book degrading from 0.75 to ~0 Sharpe out of sample. Time-series momentum is the
one anomaly that has survived a century of out-of-sample data (Moskowitz, Ooi &
Pedersen 2012; Hurst, Ooi & Pedersen 2017 backtested it to 1880) and is
structurally diversifying to equity factors. This is the most defensible
"real-money" candidate available to a retail account — and a positive,
cross-asset result to balance the (honestly-reported) equity negatives.

Method (classic AQR / MOP construction, lookahead-safe throughout)
------------------------------------------------------------------
At each monthly rebalance date t, using ONLY data through t:
  1. Signal_i = mean over lookbacks L of sign(P_i[t] / P_i[t-L] - 1)   ∈ [-1, 1]
  2. Per-instrument inverse-vol size: raw_w_i = Signal_i · σ_target / σ_i,
     where σ_i is trailing realized vol (window strictly before t).
  3. Hold to t+1; book return = Σ_i raw_w_i · r_i(t→t+1).
  4. Book-level vol targeting: leverage_t = port_vol_target / trailing_book_vol,
     where trailing_book_vol uses only realized book returns from periods < t
     (capped at max_leverage). Net of turnover transaction costs.

Lookahead discipline (this repo's identity): the signal reads the price AT t
(known at t); volatilities and the leverage multiplier read only data strictly
before t. Nothing peeks at the return it is about to earn.

Run from trend_model/ directory:
    python run.py
    python run.py --oos-start 2025-01-01
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from loguru import logger


def fetch_futures(tickers, start, end, cache_dir):
    """Download continuous front-month futures closes; keep names with ≥1y history."""
    cache = Path(cache_dir)
    p = cache / f"futures_{start}_{end}.parquet"
    if p.exists():
        logger.info(f"Loading cached futures prices from {p}")
        return pd.read_parquet(p)

    frames = {}
    for t in tickers:
        try:
            raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                logger.warning(f"  {t}: no data — skipped")
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if close.notna().sum() < 252:
                logger.warning(f"  {t}: <1y of data — skipped")
                continue
            frames[t] = close
        except Exception as e:
            logger.warning(f"  {t}: download failed ({e}) — skipped")

    prices = pd.DataFrame(frames).sort_index().ffill(limit=5)
    logger.info(f"Futures universe: {prices.shape[1]} of {len(tickers)} instruments kept")
    cache.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(p)
    return prices


def run_trend_model(cfg: dict, out_path: str = "results/backtest.csv") -> pd.DataFrame:
    d   = cfg["data"]
    bt  = cfg["backtest"]
    u   = cfg["universe"]
    cache = d["cache_dir"]

    tickers = [t for group in u.values() for t in group]
    logger.info("═══ Stage 1: Fetch cross-asset futures ═══")
    prices = fetch_futures(tickers, d["start_date"], d["end_date"], cache)
    # Subset to the configured universe (cache may hold a wider set of instruments)
    keep = [t for t in tickers if t in prices.columns]
    prices = prices[keep]
    logger.info(f"Active universe: {len(keep)} instruments — {keep}")
    clip = bt["return_clip"]
    # Roll-gap defense: a front-month continuous roll injects a one-day "return"
    # that was never tradeable. Clip daily returns so these artifacts enter
    # NEITHER the vol estimate NOR realized P&L. (Documented in README/config.)
    rets = prices.pct_change().clip(-clip, clip)
    dates = prices.index

    lookbacks   = cfg["signal"]["lookbacks_days"]
    rebal_freq  = bt["rebalance_freq"]
    warmup      = bt["warmup_days"]
    vol_window  = bt["vol_window"]
    vol_floor   = bt["inst_vol_floor"]
    max_w       = bt["max_inst_weight"]
    inst_vt     = bt["inst_vol_target"]
    port_vt     = bt["port_vol_target"]
    max_lev     = bt["max_leverage"]
    tcost       = bt["transaction_cost"]
    init_cap    = bt["initial_capital"]
    max_lb      = max(lookbacks)

    rebal_dates = dates[warmup::rebal_freq]
    oos_start = cfg.get("oos_start")
    if oos_start:
        oos_ts = pd.Timestamp(oos_start)
        rebal_dates = rebal_dates[rebal_dates >= oos_ts]
        logger.info(f"OOS mode: {len(rebal_dates)} rebalances from {oos_ts.date()}")

    logger.info("═══ Stage 2: Walk-forward trend backtest ═══")
    capital   = float(init_cap)
    prev_w    = pd.Series(0.0, index=prices.columns)
    unlev_hist = []          # past unlevered monthly book returns (strictly < t)
    records   = []

    for rd in rebal_dates:
        di = dates.get_loc(rd)
        if di < max_lb:
            continue

        # ── Signal: mean of trend signs over lookbacks (price at t is known) ──
        px_now = prices.iloc[di]
        sig = pd.Series(0.0, index=prices.columns)
        for L in lookbacks:
            past = prices.iloc[di - L]
            sig = sig.add(np.sign(px_now / past - 1.0), fill_value=0.0)
        sig /= len(lookbacks)

        # ── Per-instrument realized vol (clipped returns), window strictly before t ──
        win = rets.iloc[di - vol_window:di]               # rows di-vol_window .. di-1
        inst_vol = (win.std() * np.sqrt(252)).clip(lower=vol_floor)

        valid = px_now.notna() & inst_vol.notna() & (inst_vol > 0) & prices.iloc[di - max_lb].notna()
        raw_w = pd.Series(0.0, index=prices.columns)
        raw_w[valid] = (sig[valid] * inst_vt / inst_vol[valid]).clip(-max_w, max_w)

        # ── Hold to next rebalance; unlevered book return from CLIPPED daily
        # returns (a roll-gap day contributes no tradeable P&L) ──
        end_idx = min(di + rebal_freq, len(dates) - 1)
        hold = rets.iloc[di + 1:end_idx + 1]              # clipped daily returns over the hold
        fwd = ((1.0 + hold).prod() - 1.0).reindex(raw_w.index).fillna(0.0)
        unlev_ret = float((raw_w * fwd).sum())

        # ── Book vol targeting: leverage from past unlevered book vol only ──
        if len(unlev_hist) >= 6:
            book_vol = np.std(unlev_hist, ddof=1) * np.sqrt(12)
            lev = min(port_vt / book_vol, max_lev) if book_vol > 0 else 1.0
        else:
            lev = 1.0

        w = raw_w * lev
        to = float((w - prev_w.reindex(w.index).fillna(0.0)).abs().sum())
        cost = to * tcost
        net_ret = lev * unlev_ret - cost
        capital *= (1.0 + net_ret)

        records.append({
            "date": rd, "capital": capital, "period_return": net_ret,
            "unlev_return": unlev_ret, "leverage": lev, "turnover": to,
            "gross_exposure": float(w.abs().sum()),
            "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum()),
        })
        unlev_hist.append(unlev_ret)
        prev_w = w

    results = pd.DataFrame(records).set_index("date")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)

    # ── Summary ──
    r = results["period_return"]
    ppy = 252 / rebal_freq
    total  = capital / init_cap - 1
    sharpe = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 0 else 0.0
    eq = (1 + r).cumprod()
    maxdd = float((eq / eq.cummax() - 1).min())
    logger.info(
        f"DONE | total={total:+.1%} | Sharpe={sharpe:.2f} | "
        f"vol={r.std()*np.sqrt(ppy):.1%} | maxDD={maxdd:+.1%} | "
        f"avg lev={results['leverage'].mean():.2f} | → {out_path}"
    )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cross-asset time-series momentum sleeve")
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

    run_trend_model(cfg, out_path=out)
