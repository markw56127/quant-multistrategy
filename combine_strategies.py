"""
Multi-strategy combination harness.

Combines the net (cost-adjusted) return streams of our market-neutral
strategies and measures the portfolio-level result. The thesis: two roughly
uncorrelated Sharpe-~0.5-0.8 strategies combine into a higher Sharpe than
either alone — the core mechanism behind multi-strategy quant funds.

Strategies (each long-short, market-neutral, survivorship-free):
  - factor_model   : value + quality composite   (monthly)
  - earnings_model : PEAD / SUE drift            (monthly)

Both are resampled to a common month-end grid (they rebalance on different
calendars), then combined under two schemes:
  - equal weight   : 50/50 capital
  - risk parity    : inverse-volatility weighted (equalises risk contribution)

Run: python combine_strategies.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STRATS = {
    "factor_vq": ROOT / "factor_model"   / "results" / "backtest.csv",
    "pead":      ROOT / "earnings_model" / "results" / "backtest.csv",
    # Shelved candidate sleeves (all ~zero or negative standalone Sharpe):
    #   "insider":    insider_model/results/backtest.csv    (two attempts, 0.13 / -0.01)
    #   "sector_rot": sector_rotation/results/backtest.csv  (0.03, vol 14.8%)
    #   "statarb":    statarb_model/results/backtest.csv    (-0.59 net, -0.30 GROSS:
    #                 no mean-reversion edge in US large-cap; Do & Faff 2010)
    # See their READMEs. The book is the two real sleeves above until a
    # candidate clears standalone Sharpe >= ~0.3 with low correlation.
}
# Sleeves whose results don't exist yet are skipped automatically.
STRATS = {k: p for k, p in STRATS.items() if p.exists()}
ANNUALISE = 12   # monthly grid


def net_monthly_returns(path: Path) -> pd.Series:
    """Net return series resampled to month-end from a strategy's capital curve."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Month-end equity, then monthly returns (captures all costs via capital)
    eq = df["capital"].resample("ME").last().ffill()
    return eq.pct_change().dropna()


def stats(r: pd.Series, label: str) -> dict:
    sharpe = r.mean() / r.std() * np.sqrt(ANNUALISE) if r.std() > 0 else 0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    total = eq.iloc[-1] - 1
    return {"strategy": label, "total": total, "ann_sharpe": sharpe,
            "max_dd": dd, "ann_vol": r.std() * np.sqrt(ANNUALISE), "n": len(r)}


def main():
    rets = {k: net_monthly_returns(p) for k, p in STRATS.items()}
    panel = pd.DataFrame(rets).dropna()
    print(f"Common months: {len(panel)}  ({panel.index[0].date()} → {panel.index[-1].date()})\n")

    # Individual stats
    rows = [stats(panel[k], k) for k in panel.columns]
    print("── Individual (net, market-neutral) ──")
    for s in rows:
        print(f"  {s['strategy']:<10} total={s['total']:>+7.1%}  Sharpe={s['ann_sharpe']:.2f}  "
              f"vol={s['ann_vol']:.1%}  maxDD={s['max_dd']:+.1%}")

    # Correlation — the key to diversification benefit
    print("\n  Correlation matrix (lower = more diversification benefit):")
    print(panel.corr().round(2).to_string().replace("\n", "\n  "))
    print()

    # Combined portfolios
    print("── Combined portfolios ──")
    # Equal weight
    ew = panel.mean(axis=1)
    s_ew = stats(ew, "50/50")

    # Risk parity — FIXED 2026-06: weights now use only PAST information.
    # Previously inverse-vol was computed on the full sample (mild lookahead:
    # month t's weights knew the whole history's vols). Now: expanding-window
    # vol with a 12-month minimum, weights applied to the FOLLOWING month.
    MIN_OBS = 12
    vol_hist = panel.expanding(min_periods=MIN_OBS).std().shift(1)
    inv_hist = 1.0 / vol_hist
    w_hist = inv_hist.div(inv_hist.sum(axis=1), axis=0)
    w_hist = w_hist.fillna(1.0 / panel.shape[1])   # equal-weight until history accrues
    rp = (panel * w_hist).sum(axis=1)
    s_rp = stats(rp, "risk-parity")

    # Vol targeting (added 2026-06): scale the combined stream to a constant
    # target vol using trailing realised vol (shifted — no current-month info).
    # Leverage capped at 2x; uncapped leverage on a backtest is fantasy.
    TARGET_VOL = 0.08   # 8% annualised
    MAX_LEV    = 2.0
    realised = rp.expanding(min_periods=MIN_OBS).std().shift(1) * np.sqrt(ANNUALISE)
    lev = (TARGET_VOL / realised).clip(upper=MAX_LEV).fillna(1.0)
    rp_vt = rp * lev
    s_vt = stats(rp_vt, "rp+voltarget")

    final_w = dict(w_hist.iloc[-1].round(2))
    for s, wdesc in [(s_ew, "50/50"),
                     (s_rp, f"riskparity(expanding) latest={final_w}"),
                     (s_vt, f"vol-target {TARGET_VOL:.0%}, lev≤{MAX_LEV:.0f}x, latest lev={lev.iloc[-1]:.2f}")]:
        print(f"  {s['strategy']:<12} total={s['total']:>+7.1%}  Sharpe={s['ann_sharpe']:.2f}  "
              f"vol={s['ann_vol']:.1%}  maxDD={s['max_dd']:+.1%}   [{wdesc}]")

    # Diversification summary
    best_single = max(rows, key=lambda x: x["ann_sharpe"])
    print(f"\n── Diversification benefit ──")
    print(f"  Best single strategy Sharpe: {best_single['ann_sharpe']:.2f} ({best_single['strategy']})")
    print(f"  Combined (risk-parity) Sharpe: {s_rp['ann_sharpe']:.2f}")
    lift = s_rp["ann_sharpe"] - best_single["ann_sharpe"]
    print(f"  Sharpe lift from combining: {lift:+.2f}  "
          f"({'diversification works' if lift > 0 else 'no benefit'})")


if __name__ == "__main__":
    main()
