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
}
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
    corr = panel.corr().iloc[0, 1]
    print(f"\n  Correlation between strategies: {corr:+.2f}")
    print(f"  (Lower = more diversification benefit when combined)\n")

    # Combined portfolios
    print("── Combined portfolios ──")
    # Equal weight
    ew = panel.mean(axis=1)
    s_ew = stats(ew, "50/50")
    # Risk parity (inverse-vol, computed on full sample — simple static version)
    inv = 1.0 / panel.std()
    w = inv / inv.sum()
    rp = (panel * w).sum(axis=1)
    s_rp = stats(rp, "risk-parity")
    for s, wdesc in [(s_ew, "50/50"), (s_rp, f"riskparity {dict(w.round(2))}")]:
        print(f"  {s['strategy']:<12} total={s['total']:>+7.1%}  Sharpe={s['ann_sharpe']:.2f}  "
              f"vol={s['ann_vol']:.1%}  maxDD={s['max_dd']:+.1%}")

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
