"""
Carry sleeve — cross-asset futures, time-series carry (long positive carry,
short negative), on the same dev-selected equity-index + rates universe as
trend_model.

Why this sleeve exists
----------------------
Carry is the second futures anomaly with deep, multi-decade, multi-asset
evidence (Koijen, Moskowitz, Pedersen & Vrugt 2018, "Carry"), and it is
historically near-uncorrelated to time-series momentum — the textbook
complement to the one sleeve in this repo that survived OOS (trend_model).

Carry definitions (theory-grounded, computable from FREE data, nothing fitted)
------------------------------------------------------------------------------
Carry = the return earned if prices do not move (KMPV). For futures, the spot
leg accrues yield while the position finances at the short rate:

  equity index:  carry_i = trailing-12m dividend yield of the tracking ETF
                 (distributions / unadjusted price) - 3m T-bill.
                 (ES->SPY, NQ->QQQ, YM->DIA, RTY->IWM)
  rates:         carry_i = constant-maturity Treasury yield at the contract's
                 tenor - 3m T-bill (the term spread — the classic Fama-Bliss
                 bond-carry proxy; ZT->2y, ZF->5y, ZN->10y, ZB->30y, via FRED).

Signal_i = sign(carry_i). Position sizing, vol targeting, and costs are
inherited from trend_model verbatim: per-instrument inverse-vol sizing to a
10% vol budget (strictly-prior 60d window, clipped returns), book vol
targeting to 12% from strictly-prior realized book returns, 5 bps/turnover,
monthly rebalance. Yields and dividends are point-in-time (FRED prints and
ex-dates on/before the rebalance date).

Honest caveat: with 8 US-only instruments, time-series carry is essentially
two macro bets — the equity financing spread (div yield vs bills) and the
slope of the Treasury curve. The cross-sectional breadth of KMPV's global
result is not available here.

Run from carry_model/:
    python run.py
    python run.py --oos-start 2025-01-01
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from loguru import logger

# Load trend_model/run.py under a distinct module name (both sleeves' entry
# points are called run.py, so a bare `from run import ...` would collide)
_spec = importlib.util.spec_from_file_location(
    "trend_run", Path(__file__).resolve().parent.parent / "trend_model" / "run.py")
_trend_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trend_run)
fetch_futures = _trend_run.fetch_futures


def fetch_fred_yields(series: list, cache_dir: str) -> pd.DataFrame:
    """Daily constant-maturity Treasury yields (percent) from FRED's keyless CSV."""
    p = Path(cache_dir) / f"fred_{'_'.join(sorted(series))}.parquet"
    if p.exists():
        logger.info(f"Loading cached FRED yields from {p}")
        return pd.read_parquet(p)
    frames = {}
    for s in series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
        df = pd.read_csv(url, na_values=".")
        df.columns = ["date", s]
        frames[s] = df.set_index(pd.to_datetime(df["date"]))[s]
        logger.info(f"  FRED {s}: {frames[s].notna().sum()} obs")
    out = pd.DataFrame(frames).ffill(limit=5)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(p)
    return out


def fetch_div_yields(etfs: list, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """Trailing-12m distribution yield per ETF: rolling 252d dividend sum /
    UNADJUSTED close (adjusted closes would overstate historical yields)."""
    p = Path(cache_dir) / f"divyield_{start}_{end}.parquet"
    if p.exists():
        logger.info(f"Loading cached dividend yields from {p}")
        return pd.read_parquet(p)
    raw = yf.download(etfs, start=start, end=end, auto_adjust=False, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    out = {}
    for etf in etfs:
        divs = yf.Ticker(etf).dividends
        # ex-date timestamps arrive tz-aware at 09:30 ET — normalize to the
        # midnight-stamped dates the price index uses
        divs.index = divs.index.tz_localize(None).normalize()
        px = close[etf].dropna()
        div_daily = divs.reindex(px.index).fillna(0.0)
        out[etf] = div_daily.rolling(252, min_periods=200).sum() / px
        logger.info(f"  {etf}: div yield mean={out[etf].mean():.2%}")
    dy = pd.DataFrame(out)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    dy.to_parquet(p)
    return dy


def run_carry_model(cfg: dict, out_path: str = "results/backtest.csv") -> pd.DataFrame:
    d, bt, u = cfg["data"], cfg["backtest"], cfg["universe"]
    eq_map, rt_map = u["equities"], u["rates"]
    tickers = list(eq_map) + list(rt_map)
    fin_series = cfg["signal"]["financing_series"]

    logger.info("═══ Stage 1: Data (futures, FRED yields, ETF distributions) ═══")
    prices = fetch_futures(tickers, d["start_date"], d["end_date"], d["futures_cache_dir"])
    prices = prices[[t for t in tickers if t in prices.columns]]
    fred = fetch_fred_yields(sorted(set(list(rt_map.values()) + [fin_series])), d["cache_dir"])
    dy = fetch_div_yields(list(eq_map.values()), d["start_date"], d["end_date"], d["cache_dir"])

    dates = prices.index
    fin = (fred[fin_series] / 100.0).reindex(dates, method="ffill")
    carry = pd.DataFrame(index=dates, columns=prices.columns, dtype=float)
    for fut, etf in eq_map.items():
        carry[fut] = dy[etf].reindex(dates, method="ffill") - fin
    for fut, tenor in rt_map.items():
        carry[fut] = (fred[tenor] / 100.0).reindex(dates, method="ffill") - fin

    clip = bt["return_clip"]
    rets = prices.pct_change().clip(-clip, clip)   # same roll-gap defense as trend

    logger.info("═══ Stage 2: Walk-forward carry backtest ═══")
    vol_window, vol_floor = bt["vol_window"], bt["inst_vol_floor"]
    max_w, inst_vt = bt["max_inst_weight"], bt["inst_vol_target"]
    port_vt, max_lev = bt["port_vol_target"], bt["max_leverage"]
    tcost, init_cap = bt["transaction_cost"], bt["initial_capital"]
    rebal_freq = bt["rebalance_freq"]

    rebal_dates = dates[bt["warmup_days"]::rebal_freq]
    if cfg.get("oos_start"):
        oos_ts = pd.Timestamp(cfg["oos_start"])
        rebal_dates = rebal_dates[rebal_dates >= oos_ts]
        logger.info(f"OOS mode: {len(rebal_dates)} rebalances from {oos_ts.date()}")

    capital = float(init_cap)
    prev_w = pd.Series(0.0, index=prices.columns)
    unlev_hist, records = [], []

    for rd in rebal_dates:
        di = dates.get_loc(rd)
        if di < vol_window:
            continue

        sig = np.sign(carry.loc[rd])                       # +1 / -1 / 0(nan->0)
        sig = sig.fillna(0.0)

        win = rets.iloc[di - vol_window:di]                # strictly before t
        inst_vol = (win.std() * np.sqrt(252)).clip(lower=vol_floor)

        valid = prices.iloc[di].notna() & inst_vol.notna() & (inst_vol > 0)
        raw_w = pd.Series(0.0, index=prices.columns)
        raw_w[valid] = (sig[valid] * inst_vt / inst_vol[valid]).clip(-max_w, max_w)

        end_idx = min(di + rebal_freq, len(dates) - 1)
        hold = rets.iloc[di + 1:end_idx + 1]
        fwd = ((1.0 + hold).prod() - 1.0).reindex(raw_w.index).fillna(0.0)
        unlev_ret = float((raw_w * fwd).sum())

        if len(unlev_hist) >= 6:
            book_vol = np.std(unlev_hist, ddof=1) * np.sqrt(12)
            lev = min(port_vt / book_vol, max_lev) if book_vol > 0 else 1.0
        else:
            lev = 1.0

        w = raw_w * lev
        to = float((w - prev_w.reindex(w.index).fillna(0.0)).abs().sum())
        net_ret = lev * unlev_ret - to * tcost
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

    r = results["period_return"]
    ppy = 252 / rebal_freq
    sharpe = (r.mean() / r.std()) * np.sqrt(ppy) if r.std() > 0 else 0.0
    eq_curve = (1 + r).cumprod()
    maxdd = float((eq_curve / eq_curve.cummax() - 1).min())
    logger.info(
        f"DONE | total={capital / init_cap - 1:+.1%} | Sharpe={sharpe:.2f} | "
        f"vol={r.std() * np.sqrt(ppy):.1%} | maxDD={maxdd:+.1%} | "
        f"avg lev={results['leverage'].mean():.2f} | → {out_path}"
    )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cross-asset futures carry sleeve")
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
    run_carry_model(cfg, out_path=out)
