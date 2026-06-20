"""
Statistical-arbitrage (pairs) sleeve — Ornstein-Uhlenbeck mean reversion.

Why this sleeve exists
----------------------
Every other sleeve in this repo is trend-following in spirit (factor momentum,
PEAD drift, cross-asset TSMOM). The book had ZERO mean-reversion exposure. Pairs
trading fills that structural gap, and it is negatively correlated with trend by
construction — it tends to make money exactly when trend chops sideways. It is
also the canonical *tradeable* application of a stochastic differential equation:
the spread between two cointegrated names is modeled as an OU process

    dS_t = theta (mu - S_t) dt + sigma dW_t

— mean-reverting to mu with speed theta — and we trade deviations from mu. (This
is the SDE done in its right home, vs. the Fokker-Planck sleeve that died to
lookahead, LOOKAHEAD_FINDING.md.)

Method (Engle-Granger + OU, lookahead-safe throughout)
------------------------------------------------------
Rolling formation/trading split. For each formation window (strictly before the
trading window that follows it):
  1. Candidate pairs are formed WITHIN GICS sector only (shared economic driver →
     cointegration is plausible, not spurious; cuts C(616,2) to a tractable set).
  2. Cheap correlation pre-filter on formation-window log prices, then the
     Engle-Granger test: OLS  logP_A = alpha + beta logP_B + s ,  ADF on residual
     s. Keep pairs with ADF p < adf_pvalue_max and beta > 0.
  3. OU fit on the residual via its AR(1) representation:
        s_t = a + b s_{t-1} + e  ->  theta = -ln(b),  half_life = ln2 / theta,
        mu = a/(1-b),  sigma_eq = std(s) over the formation window.
     Keep pairs whose half-life is in [half_life_min, half_life_max] (mean-reverts
     well inside the trading window). Rank by ADF p-value, keep top max_pairs.
  4. FREEZE (alpha, beta, mu, sigma_eq) and trade the NEXT trading window only:
        z_t = (logP_A[t] - alpha - beta logP_B[t] - mu) / sigma_eq
     z > +entry: short the spread (short A / long B); z < -entry: long the spread.
     Exit when |z| < exit_z; hard stop when |z| > stop_z (relationship broke).
     Legs are beta-neutral, gross 1: w_A = 1/(1+beta), w_B = -beta/(1+beta).

Lookahead discipline (this repo's identity): every pair parameter is estimated on
data strictly before the trading window. During trading, z_t reads prices known
at t, the position set at t earns the return from t -> t+1, and a leg that delists
(price -> NaN) force-closes the pair. Nothing peeks at the return it will earn.

Run from statarb_model/ directory:
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
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from universe_pit import (  # noqa: E402
    historical_universe, fetch_prices_survivorship_free, fetch_sectors,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pair formation: cointegration + OU fit on a formation window
# ─────────────────────────────────────────────────────────────────────────────
def _ols(y: np.ndarray, x: np.ndarray):
    """OLS y = a + b x. Returns (a, b, residuals)."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return coef[0], coef[1], resid


def _ou_from_residual(s: np.ndarray):
    """
    OU parameters from the AR(1) fit of the cointegration residual.
    Returns dict(theta, mu, half_life, sigma_eq) or None if not mean-reverting.
    """
    s0, s1 = s[:-1], s[1:]
    a, b, _ = _ols(s1, s0)
    if not (0.0 < b < 1.0):          # b>=1: no reversion; b<=0: oscillatory/garbage
        return None
    theta = -np.log(b)
    half_life = np.log(2.0) / theta
    mu = a / (1.0 - b)
    sigma_eq = float(np.std(s, ddof=1))
    if sigma_eq <= 0:
        return None
    return {"theta": theta, "mu": mu, "half_life": half_life, "sigma_eq": sigma_eq}


def form_pairs(logpx: pd.DataFrame, sectors: pd.Series, fcfg: dict) -> list:
    """
    Select cointegrated, OU-mean-reverting pairs on a single formation window.
    `logpx` is the log-price panel for the formation window only (rows = days).
    Returns a list of frozen-parameter dicts, best (lowest ADF p) first.
    """
    # Names with complete history over the whole formation window and price floor
    full = logpx.columns[logpx.notna().all(axis=0)]
    px_level = np.exp(logpx[full].iloc[-1])
    full = full[px_level >= fcfg["min_price"]]
    lp = logpx[full]

    pairs = []
    for sector, names in sectors.loc[sectors.index.isin(full)].groupby(sectors).groups.items():
        names = [n for n in names if n in full]
        if len(names) < 2:
            continue
        block = lp[names]
        corr = block.corr()                       # cheap gate before costly ADF
        cols = list(names)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                if corr.loc[a, b] < fcfg["corr_prefilter"]:
                    continue
                ya, xb = block[a].values, block[b].values
                alpha, beta, resid = _ols(ya, xb)
                if beta <= 0:                      # need a positive hedge ratio
                    continue
                try:
                    pval = adfuller(resid, maxlag=1, regression="c", autolag=None)[1]
                except Exception:
                    continue
                if pval > fcfg["adf_pvalue_max"]:
                    continue
                ou = _ou_from_residual(resid)
                if ou is None:
                    continue
                if not (fcfg["half_life_min"] <= ou["half_life"] <= fcfg["half_life_max"]):
                    continue
                pairs.append({"a": a, "b": b, "alpha": alpha, "beta": beta,
                              "pval": pval, **ou})

    pairs.sort(key=lambda p: p["pval"])
    return pairs[: fcfg["max_pairs"]]


# ─────────────────────────────────────────────────────────────────────────────
# Trade one frozen pair over a trading window -> daily net return series
# ─────────────────────────────────────────────────────────────────────────────
def trade_pair(pair: dict, logpx: pd.DataFrame, rets: pd.DataFrame, bt: dict) -> pd.Series:
    """
    Daily net return (per unit pair capital) over the trading-window index of
    `logpx`. Parameters in `pair` are frozen from the formation window. Position
    chosen at day t earns leg returns from t -> t+1; a NaN leg force-closes.
    """
    a, b = pair["a"], pair["b"]
    alpha, beta, mu, sigma = pair["alpha"], pair["beta"], pair["mu"], pair["sigma_eq"]
    wA, wB = 1.0 / (1.0 + beta), -beta / (1.0 + beta)     # beta-neutral, gross 1
    entry, exit_z, stop = bt["entry_z"], bt["exit_z"], bt["stop_z"]
    tcost = bt["transaction_cost"]

    idx = logpx.index
    spread = logpx[a] - alpha - beta * logpx[b] - mu
    z = spread / sigma
    rA, rB = rets[a], rets[b]

    out = pd.Series(0.0, index=idx)
    pos = 0          # -1 short spread, +1 long spread, 0 flat
    dead = False     # stop-out kills the pair for the rest of the window
    for k in range(len(idx) - 1):
        t = idx[k]
        zt = z.iloc[k]
        # delisting / data gap on either leg -> force flat, charge exit if needed
        if np.isnan(zt) or np.isnan(rA.iloc[k + 1]) or np.isnan(rB.iloc[k + 1]):
            if pos != 0:
                out.iloc[k] -= tcost * (abs(wA) + abs(wB))
                pos = 0
            continue

        new_pos = pos
        if dead:
            new_pos = 0
        elif pos == 0:
            if zt > entry:
                new_pos = -1
            elif zt < -entry:
                new_pos = +1
        else:  # in a position
            if abs(zt) < exit_z:
                new_pos = 0
            elif abs(zt) > stop:
                new_pos = 0
                dead = True

        # transaction cost on the change in signed leg weights (L1)
        if new_pos != pos:
            out.iloc[k] -= tcost * abs(new_pos - pos) * (abs(wA) + abs(wB))
        pos = new_pos

        # held position earns the next day's beta-neutral leg return
        if pos != 0:
            out.iloc[k] += pos * (wA * rA.iloc[k + 1] + wB * rB.iloc[k + 1])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward driver
# ─────────────────────────────────────────────────────────────────────────────
def run_statarb(cfg: dict, out_path: str = "results/backtest.csv") -> pd.DataFrame:
    d, fcfg, bt = cfg["data"], cfg["formation"], cfg["backtest"]
    cache = d["cache_dir"]

    logger.info("═══ Stage 1: Survivorship-free universe & prices ═══")
    tickers = historical_universe(d["start_date"], d["end_date"], cache_dir=cache)
    prices = fetch_prices_survivorship_free(tickers, d["start_date"], d["end_date"], cache_dir=cache)
    sectors = fetch_sectors(list(prices.columns), cache_dir=cache)
    logpx = np.log(prices.where(prices > 0))
    rets = prices.pct_change()
    dates = prices.index
    logger.info(f"Universe: {prices.shape[1]} names, {len(dates)} trading days "
                f"[{dates[0].date()} → {dates[-1].date()}]")

    F, T = fcfg["formation_days"], fcfg["trading_days"]
    oos_start = cfg.get("oos_start")
    oos_ts = pd.Timestamp(oos_start) if oos_start else None

    logger.info("═══ Stage 2: Rolling formation / OU trading ═══")
    book_daily = pd.Series(0.0, index=dates)     # equal-weight across pair slots
    records = []   # per-window diagnostics

    start = F
    while start + T <= len(dates):
        f_slice = slice(start - F, start)            # formation: strictly before trading
        t_slice = slice(start, start + T)            # trading window
        t_idx = dates[t_slice]

        if oos_ts is not None and t_idx[-1] < oos_ts:
            start += T
            continue

        pairs = form_pairs(logpx.iloc[f_slice], sectors, fcfg)
        if pairs:
            slot_rets = [trade_pair(p, logpx.iloc[t_slice], rets.iloc[t_slice], bt)
                         for p in pairs]
            win_ret = pd.concat(slot_rets, axis=1).mean(axis=1)   # equal weight
            book_daily.loc[win_ret.index] = win_ret.values

        n = len(pairs)
        avg_hl = float(np.mean([p["half_life"] for p in pairs])) if n else float("nan")
        records.append({
            "form_end": dates[start].date(), "trade_to": t_idx[-1].date(),
            "n_pairs": n, "avg_half_life": avg_hl,
        })
        logger.info(f"  form→{dates[start].date()}  trade→{t_idx[-1].date()}  "
                    f"pairs={n:>2}  avg half-life={avg_hl:4.1f}d")
        start += T

    # Trim to the traded span, build the capital curve
    if oos_ts is not None:
        book_daily = book_daily.loc[book_daily.index >= oos_ts]
    traded = book_daily.loc[book_daily.ne(0).cumsum() > 0]    # from first active day
    init_cap = bt["initial_capital"]
    capital = init_cap * (1.0 + traded).cumprod()

    results = pd.DataFrame({
        "capital": capital,
        "period_return": traded,
    })
    results.index.name = "date"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)

    # ── Summary ──
    r = traded
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0.0
    eq = (1 + r).cumprod()
    maxdd = float((eq / eq.cummax() - 1).min())
    total = float(eq.iloc[-1] - 1)
    formed = pd.DataFrame(records)
    logger.info(
        f"DONE | total={total:+.1%} | Sharpe={sharpe:.2f} | "
        f"vol={r.std()*np.sqrt(252):.1%} | maxDD={maxdd:+.1%} | "
        f"avg pairs/window={formed['n_pairs'].mean():.1f} | → {out_path}"
    )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OU statistical-arbitrage (pairs) sleeve")
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

    run_statarb(cfg, out_path=out)
