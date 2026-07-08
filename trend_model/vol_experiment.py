"""
Pre-registered volatility-forecasting experiment for trend_model sizing.

Motivation: direction is barely predictable, volatility genuinely is. The
sleeve sizes positions by inverse vol (trailing 60d) — if a better vol
forecast exists, sizing improves with zero change to the signal. This also
gives the "segmented curve fitting" idea its honest test: applied to the one
quantity where local-segment estimation has statistical traction.

Estimators (per instrument, strictly-prior data only, at each monthly
rebalance; all forecast the NEXT-21-day annualized vol):
  rolling60   trailing 60d std (current baseline in run.py)
  ewma        RiskMetrics EWMA, lambda = 0.94 (canonical value, not fitted)
  garch       GARCH(1,1) MLE on trailing 3y (min 300 obs), 21d-ahead mean vol
  segmented   causal changepoint vol: Inclan-Tiao ICSS on the trailing 252d,
              recursing into the right-hand (most recent) segment, vol from
              the current segment only (>= 20 obs). No future data touches a
              boundary.

Scoring, fixed before running:
  Stage 1 (forecast skill): QLIKE = log(f2) + rv2/f2 and RMSE of annualized
    vol, on the COMMON sample where all estimators produce a forecast;
    Newey-West t (lag 1) on per-date mean loss differential vs rolling60.
  Stage 2 (economic value): identical trend backtest with each estimator in
    per-instrument sizing (book-level vol targeting untouched); dev/OOS net
    Sharpe and vol-target tracking = mean |rolling-6m realized book vol - 12%|.
  Adoption bar: QLIKE win with DM t > 2 AND dev Sharpe within 0.05 of
    baseline AND tracking not worse. Missing forecasts fall back to rolling60
    so variants differ only where the estimator is defined.

Run from trend_model/:  python vol_experiment.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

warnings.filterwarnings("ignore")

# ── config (inherited from config.yaml / run.py, nothing new fitted) ──
cfg = yaml.safe_load(open("config.yaml"))
TICKERS = [t for g in cfg["universe"].values() for t in g]
CLIP = cfg["backtest"]["return_clip"]
LOOKBACKS = cfg["signal"]["lookbacks_days"]
REBAL, WARMUP = cfg["backtest"]["rebalance_freq"], cfg["backtest"]["warmup_days"]
VOL_FLOOR, MAX_W = cfg["backtest"]["inst_vol_floor"], cfg["backtest"]["max_inst_weight"]
INST_VT, PORT_VT = cfg["backtest"]["inst_vol_target"], cfg["backtest"]["port_vol_target"]
MAX_LEV, TCOST = cfg["backtest"]["max_leverage"], cfg["backtest"]["transaction_cost"]
OOS = pd.Timestamp("2025-01-01")
EWMA_LAMBDA = 0.94
GARCH_WIN, GARCH_MIN = 756, 300
ICSS_WIN, ICSS_MIN_SEG, ICSS_CRIT = 252, 20, 1.358   # 95% critical value

prices = pd.read_parquet("cache/futures_2015-01-01_2026-06-30.parquet")[TICKERS]
rets = prices.pct_change().clip(-CLIP, CLIP)
dates = prices.index
rebal_dates = [rd for rd in dates[WARMUP::REBAL] if dates.get_loc(rd) >= max(LOOKBACKS)]


def vol_rolling60(r: np.ndarray) -> float:
    return float(np.nanstd(r[-60:], ddof=1) * np.sqrt(252))


def vol_ewma(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    if len(r) < 60:
        return np.nan
    var = np.var(r[:30])
    for x in r[30:]:
        var = EWMA_LAMBDA * var + (1 - EWMA_LAMBDA) * x * x
    return float(np.sqrt(var * 252))


def vol_garch(r: np.ndarray) -> float:
    from arch import arch_model
    r = r[~np.isnan(r)][-GARCH_WIN:]
    if len(r) < GARCH_MIN:
        return np.nan
    try:
        res = arch_model(r * 100, vol="GARCH", p=1, q=1).fit(disp="off", show_warning=False)
        f = res.forecast(horizon=REBAL, reindex=False).variance.values[0]
        return float(np.sqrt(f.mean() * 252) / 100)
    except Exception:
        return np.nan


def _icss_break(x: np.ndarray) -> int:
    """Most significant variance changepoint (Inclan-Tiao); -1 if none."""
    c = np.cumsum(x * x)
    if c[-1] <= 0:
        return -1
    T = len(x)
    k = np.arange(1, T + 1)
    D = c / c[-1] - k / T
    j = int(np.argmax(np.abs(D[:-1])))
    return j + 1 if np.abs(D[j]) * np.sqrt(T / 2.0) > ICSS_CRIT else -1


def vol_segmented(r: np.ndarray) -> float:
    """Vol of the CURRENT variance segment, boundaries from past data only."""
    x = r[~np.isnan(r)][-ICSS_WIN:]
    if len(x) < 2 * ICSS_MIN_SEG:
        return np.nan
    seg = x
    while len(seg) >= 2 * ICSS_MIN_SEG:
        b = _icss_break(seg)
        if b <= 0 or len(seg) - b < ICSS_MIN_SEG:
            break
        seg = seg[b:]                      # recurse into the most recent segment
    return float(np.std(seg, ddof=1) * np.sqrt(252))


ESTIMATORS = {"rolling60": vol_rolling60, "ewma": vol_ewma,
              "garch": vol_garch, "segmented": vol_segmented}


def main() -> None:
    logger.info(f"Forecasting at {len(rebal_dates)} rebalances × {len(TICKERS)} instruments")
    fc = {v: pd.DataFrame(index=rebal_dates, columns=TICKERS, dtype=float) for v in ESTIMATORS}
    rv = pd.DataFrame(index=rebal_dates, columns=TICKERS, dtype=float)

    for rd in rebal_dates:
        di = dates.get_loc(rd)
        hist = rets.iloc[:di]              # strictly before t, as in run.py
        fwd = rets.iloc[di + 1: min(di + REBAL, len(dates) - 1) + 1]
        for tk in TICKERS:
            h = hist[tk].values
            if np.isnan(h[-60:]).all():
                continue
            for v, fn in ESTIMATORS.items():
                fc[v].at[rd, tk] = fn(h)
            f = fwd[tk].dropna()
            if len(f) >= 15:
                rv.at[rd, tk] = f.std(ddof=1) * np.sqrt(252)

    # ── Stage 1: forecast skill on the common sample ──
    common = rv.notna()
    for v in ESTIMATORS:
        common &= fc[v].notna()
    n = int(common.values.sum())
    print(f"\n═══ Stage 1: forecast skill (common sample, n={n}) ═══")
    print(f"{'estimator':10s} {'QLIKE':>8s} {'RMSE(vol)':>10s} {'DM t vs rolling60':>18s}")
    qlike = {}
    for v in ESTIMATORS:
        f2 = (fc[v][common] ** 2)
        rv2 = (rv[common] ** 2)
        qlike[v] = np.log(f2) + rv2 / f2
        rmse = float(np.sqrt(((fc[v][common] - rv[common]) ** 2).stack().mean()))
        dm = ""
        if v != "rolling60":
            d = (qlike["rolling60"] - qlike[v]).mean(axis=1).dropna()  # >0 → v better
            # Newey-West (lag 1) t on the mean per-date loss differential
            u = d - d.mean()
            g0 = float((u * u).mean())
            g1 = float((u * u.shift(1)).dropna().mean())
            se = np.sqrt(max(g0 + 2 * (1 - 1 / 2) * g1, 1e-12) / len(d))
            dm = f"{float(d.mean() / se):+18.2f}"
        print(f"{v:10s} {float(qlike[v].stack().mean()):8.4f} {rmse:10.4f} {dm}")

    # ── Stage 2: identical trend backtest, sizing vol swapped ──
    print(f"\n═══ Stage 2: trend book with each sizing estimator ═══")
    print(f"{'estimator':10s} {'dev':>6s} {'OOS':>6s} {'full':>6s} {'realized vol':>13s} {'|6m vol-12%|':>13s}")
    books = {}
    for v in ESTIMATORS:
        prev_w = pd.Series(0.0, index=TICKERS)
        unlev_hist, recs = [], []
        for rd in rebal_dates:
            di = dates.get_loc(rd)
            px_now = prices.iloc[di]
            sig = pd.Series(0.0, index=TICKERS)
            for L in LOOKBACKS:
                sig = sig.add(np.sign(px_now / prices.iloc[di - L] - 1.0), fill_value=0.0)
            sig /= len(LOOKBACKS)
            iv = fc[v].loc[rd].fillna(fc["rolling60"].loc[rd]).clip(lower=VOL_FLOOR)
            valid = px_now.notna() & iv.notna() & (iv > 0) & prices.iloc[di - max(LOOKBACKS)].notna()
            raw_w = pd.Series(0.0, index=TICKERS)
            raw_w[valid] = (sig[valid] * INST_VT / iv[valid]).clip(-MAX_W, MAX_W)
            ei = min(di + REBAL, len(dates) - 1)
            fwd = ((1 + rets.iloc[di + 1: ei + 1]).prod() - 1).reindex(TICKERS).fillna(0.0)
            unlev = float((raw_w * fwd).sum())
            lev = min(PORT_VT / (np.std(unlev_hist, ddof=1) * np.sqrt(12)), MAX_LEV) \
                if len(unlev_hist) >= 6 else 1.0
            w = raw_w * lev
            net = lev * unlev - float((w - prev_w).abs().sum()) * TCOST
            recs.append((rd, net))
            unlev_hist.append(unlev)
            prev_w = w
        r = pd.Series(dict(recs))
        books[v] = r
        sh = lambda x: float(x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else 0.0
        roll6 = r.rolling(6).std() * np.sqrt(12)
        track = float((roll6 - PORT_VT).abs().mean())
        print(f"{v:10s} {sh(r[r.index < OOS]):+6.2f} {sh(r[r.index >= OOS]):+6.2f} "
              f"{sh(r):+6.2f} {float(r.std() * np.sqrt(12)):13.1%} {track:13.2%}")

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(books).to_csv("results/vol_experiment_books.csv")
    pd.concat({v: fc[v] for v in ESTIMATORS}, axis=1).to_csv("results/vol_experiment_forecasts.csv")
    logger.info("→ results/vol_experiment_books.csv, results/vol_experiment_forecasts.csv")


if __name__ == "__main__":
    main()
