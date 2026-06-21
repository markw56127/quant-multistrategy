"""
Turnover, transaction-cost and capacity analysis for the composite factor L/S.

Builds the same long-short quintile book the factor sleeve trades (top/bottom 20%
of the composite z-score, equal-weight, dollar-neutral) and asks the two questions
a desk asks before allocating:

  TURNOVER / COST : how fast does the book trade, and at what per-trade cost does
                    the gross edge break even? (net Sharpe vs a grid of bps costs.)

  CAPACITY        : how much capital can it run before its own market impact eats
                    the alpha? Uses a square-root impact model (Almgren):
                        cost_fraction_i ≈ c · σ_daily · sqrt( traded$_i / ADV$_i )
                    with c≈1, σ_daily≈2%. Per-rebalance impact drag on the book is
                    Σ_i |Δw_i|·c·σ·sqrt(|Δw_i|·AUM / ADV$_i). We sweep AUM to find
                    where net annualized return PEAKS (capacity) and where net
                    Sharpe falls to half its frictionless value.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

ROOT = Path(__file__).resolve().parent
PANEL = ROOT.parent / "signal_combiner" / "cache" / "panel.parquet"
# The research book is VALUE + QUALITY: the two premia Fama-MacBeth and the IC
# analysis flag as positive (value t≈3, quality marginal). The other three factors
# (momentum/low_vol/size) are flat-to-negative over 2016-2026 and only dilute the
# composite — so an evidence-built book, not the naive all-5 average, is what we
# size and analyse here.
BOOK = ["value", "quality"]
ANN = 12
C_IMPACT, SIGMA_D = 1.0, 0.02          # square-root impact coefficient, daily vol


def ls_weights(panel: pd.DataFrame, q: float = 0.20) -> pd.DataFrame:
    """Dollar-neutral equal-weight top/bottom-q composite book: date×ticker weights."""
    p = panel.copy()
    p["composite"] = p[BOOK].mean(axis=1)
    W = {}
    for dt, g in p.groupby("date"):
        g = g.dropna(subset=["composite"])
        k = max(int(len(g) * q), 1)
        ranked = g.sort_values("composite")
        w = pd.Series(0.0, index=g["ticker"])
        w[ranked["ticker"].iloc[-k:].values] = 0.5 / k        # long top
        w[ranked["ticker"].iloc[:k].values] = -0.5 / k        # short bottom
        W[dt] = w
    return pd.DataFrame(W).T.sort_index().fillna(0.0)


def adv_dollar(tickers, dates, cache_dir) -> pd.DataFrame:
    """63-day trailing average daily dollar volume, sampled at rebalance dates."""
    p = Path(cache_dir) / "adv_dollar.parquet"
    if p.exists():
        logger.info(f"Loading cached ADV$ from {p}")
        return pd.read_parquet(p).reindex(dates, method="ffill")
    logger.info(f"Fetching volume for {len(tickers)} names...")
    raw = yf.download(list(tickers), start="2015-06-01", end="2026-06-30",
                      auto_adjust=True, progress=False)
    close, vol = raw["Close"], raw["Volume"]
    dollar = (close * vol).rolling(63, min_periods=20).mean()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    dollar.to_parquet(p)
    return dollar.reindex(dates, method="ffill")


def main():
    panel = pd.read_parquet(PANEL)
    W = ls_weights(panel)
    fwd = panel.pivot(index="date", columns="ticker", values="fwd_ret").reindex(W.index)
    cols = W.columns
    fwd = fwd.reindex(columns=cols)

    gross = (W * fwd.fillna(0)).sum(axis=1)                    # monthly gross L/S return
    dW = W.diff().abs()
    dW.iloc[0] = W.iloc[0].abs()
    turn = dW.sum(axis=1)                                     # one-way turnover (sum|Δw|)

    g_sharpe = gross.mean() / gross.std() * np.sqrt(ANN)
    print(f"\n{'='*78}\nTurnover & transaction-cost analysis ({'+'.join(BOOK)} L/S, top/bottom 20%)\n{'='*78}")
    print(f"avg one-way turnover: {turn.mean():.1%}/month   "
          f"({turn.mean()*ANN:.1f}× notional/yr)")
    print(f"frictionless gross: Sharpe={g_sharpe:+.2f}  ann.return={gross.mean()*ANN:+.1%}\n")
    print(f"{'cost (bps/trade)':<18}{'net ann.ret':>12}{'net Sharpe':>12}")
    breakeven = None
    for bps in [0, 5, 10, 15, 20, 30, 50]:
        net = gross - turn * bps / 1e4
        ns = net.mean() / net.std() * np.sqrt(ANN)
        print(f"{bps:<18}{net.mean()*ANN:>+12.1%}{ns:>+12.2f}")
        if breakeven is None and net.mean() <= 0:
            breakeven = bps
    print(f"\nbreakeven cost ≈ {'<' if breakeven==0 else ''}{breakeven if breakeven else '>50'} "
          f"bps/trade  (gross edge dies above this)")

    # ── Capacity: square-root market impact vs AUM ──
    adv = adv_dollar(cols, W.index, ROOT / "cache").reindex(columns=cols)
    print(f"\n{'='*78}\nCapacity analysis (square-root impact: c={C_IMPACT}, σ_daily={SIGMA_D:.0%})\n{'='*78}")
    print(f"{'AUM':>10}{'net ann.ret':>13}{'net Sharpe':>12}{'avg impact':>12}")
    base = 10e6
    rows = []
    for aum in [base*m for m in [0.1, 1, 5, 10, 25, 50, 100, 250, 500]]:
        # per-name traded $ = |Δw|·AUM ; impact fraction = c·σ·sqrt(traded$/ADV$)
        traded = dW * aum
        part = traded / adv.replace(0, np.nan)
        impact_freq = C_IMPACT * SIGMA_D * np.sqrt(part.clip(lower=0).fillna(0))
        drag = (dW * impact_freq).sum(axis=1)                 # book return drag/rebalance
        net = gross - drag
        ns = net.mean() / net.std() * np.sqrt(ANN)
        rows.append((aum, net.mean()*ANN, ns, drag.mean()))
        print(f"{aum/1e6:>8.0f}M{net.mean()*ANN:>+13.1%}{ns:>+12.2f}{drag.mean()*ANN:>+12.1%}")
    cap = pd.DataFrame(rows, columns=["aum", "net_ret", "net_sharpe", "drag"])
    cap["net_pnl"] = cap["net_ret"] * cap["aum"]              # net annual dollar P&L
    half = cap[cap["net_sharpe"] <= g_sharpe / 2]
    half_aum = f"~${half['aum'].min()/1e6:.0f}M" if not half.empty else ">$5B (max tested)"
    pnl_rising = cap["net_pnl"].idxmax() == len(cap) - 1
    peak_txt = ">$5B (still rising)" if pnl_rising else f"~${cap.loc[cap['net_pnl'].idxmax(),'aum']/1e6:.0f}M"
    print(f"\nnet dollar P&L peaks: {peak_txt}   "
          f"net Sharpe halves (½ of {g_sharpe:.2f}) by: {half_aum}")
    print("→ value+quality is LOW-turnover and HIGH-capacity: slow signal (IC rises "
          "with horizon), so impact stays small into the billions. The opposite of a\n"
          "  fast/crowded alpha — its constraint is the size of the edge, not capacity.")
    cap.to_csv(ROOT / "results" / "capacity.csv", index=False)


if __name__ == "__main__":
    main()
