"""
VIX term-structure sleeve — volatility risk premium with a regime switch.

Why this sleeve exists
----------------------
The volatility risk premium (implied vol persistently exceeds subsequently
realized vol) and VIX futures term-structure carry are among the best-
documented premia in the literature (Cheng 2019; Eraker & Wu 2017). This is
the first candidate return source for the book since carry failed — and it is
structurally unlike everything already tested (it trades vol, not direction).

Construction (pre-registered; the threshold is the theory value, not fitted)
----------------------------------------------------------------------------
At each weekly rebalance rd, using closes through rd:
  ratio = VIX / VIX3M
  ratio <  1.0 (contango):      hold SVXY (-0.5x short-vol ETP) — harvest VRP
  ratio >= 1.0 (backwardation): hold VXX (long-vol ETP) — crisis convexity
Position = min(1.0, vol_target / trailing-60d vol of the held ETP), rest cash.
Net of 10 bps/turnover; long-only, no borrow. Entry at rd close, P&L
close(rd) -> close(rd+5): no same-day information credit (LOOKAHEAD rules).

Benchmarks: static vol-targeted SVXY (is the *timing* worth anything beyond
the raw premium?) and SPY.

Dev window 2019-2024 (both ETPs in current form, no structural break inside
the sample; VXX series B starts 2018-01, SVXY deleveraged 2018-02).
OOS 2025+. Bar (pre-set): dev net Sharpe >= 0.5 AND OOS > 0 AND monthly corr
to trend_model < 0.5.

Run from vix_model/:
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


def fetch_closes(tickers: list, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    p = Path(cache_dir) / f"closes_{start}_{end}.parquet"
    if p.exists():
        logger.info(f"Loading cached closes from {p}")
        return pd.read_parquet(p)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes = closes.ffill(limit=5)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    closes.to_parquet(p)
    return closes


def run_vix_model(cfg: dict, out_path: str = "results/backtest.csv") -> pd.DataFrame:
    d, tk, bt = cfg["data"], cfg["tickers"], cfg["backtest"]
    thresh = cfg["signal"]["ratio_threshold"]

    logger.info("═══ Stage 1: Data ═══")
    px = fetch_closes(list(tk.values()), d["start_date"], d["end_date"], d["cache_dir"])
    ratio = px[tk["vix"]] / px[tk["vix3m"]]
    etps = [tk["short_vol"], tk["long_vol"]]
    rets = px[etps].pct_change()

    freq, vw = bt["rebalance_freq"], bt["vol_window"]
    vt, max_w, tcost = bt["vol_target"], bt["max_weight"], bt["transaction_cost"]
    dates = px.index
    start_i = dates.get_loc(dates[dates >= pd.Timestamp(bt["start"])][0])
    rebal_dates = dates[start_i::freq]
    if cfg.get("oos_start"):
        oos_ts = pd.Timestamp(cfg["oos_start"])
        rebal_dates = rebal_dates[rebal_dates >= oos_ts]
        logger.info(f"OOS mode: {len(rebal_dates)} rebalances from {oos_ts.date()}")

    logger.info("═══ Stage 2: Weekly term-structure backtest ═══")
    capital = float(bt["initial_capital"])
    prev_w = pd.Series(0.0, index=etps)
    records = []
    for rd in rebal_dates:
        di = dates.get_loc(rd)
        if di < vw or pd.isna(ratio.loc[rd]):
            continue
        held = tk["short_vol"] if ratio.loc[rd] < thresh else tk["long_vol"]
        vol60 = rets[held].iloc[di - vw:di].std() * np.sqrt(252)   # strictly prior
        if pd.isna(vol60) or vol60 <= 0:
            continue
        w = pd.Series(0.0, index=etps)
        w[held] = min(max_w, vt / vol60)

        to = float((w - prev_w).abs().sum())
        ei = min(di + freq, len(dates) - 1)
        fwd = (px[etps].iloc[ei] / px[etps].iloc[di] - 1).fillna(0.0)
        net = float((w * fwd).sum()) - to * tcost
        capital *= (1.0 + net)
        spy = float(px[tk["benchmark"]].iloc[ei] / px[tk["benchmark"]].iloc[di] - 1)
        records.append({
            "date": rd, "capital": capital, "period_return": net,
            "benchmark_return": spy, "ratio": float(ratio.loc[rd]),
            "held": held, "weight": float(w[held]), "turnover": to,
        })
        prev_w = w

    results = pd.DataFrame(records).set_index("date")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)

    r = results["period_return"]
    ppy = 252 / freq
    sharpe = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 0 else 0.0
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    frac_short_vol = float((results["held"] == tk["short_vol"]).mean())
    logger.info(
        f"DONE | total={eq.iloc[-1]-1:+.1%} | Sharpe={sharpe:.2f} | "
        f"vol={r.std()*np.sqrt(ppy):.1%} | maxDD={dd:+.1%} | "
        f"in short-vol {frac_short_vol:.0%} of weeks | → {out_path}"
    )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VIX term-structure sleeve")
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
    run_vix_model(cfg, out_path=out)
