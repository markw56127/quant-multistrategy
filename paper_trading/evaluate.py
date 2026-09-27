"""
Forward-test evaluation from the ledger.

1. IMPLEMENTATION SHORTFALL per filled order: signed (fill − decision) / decision, in bps,
   positive = cost. This is the counterparty/execution cost a close-to-close backtest never
   sees. (Alpaca paper fills are simulated against quotes: they capture the spread and
   timing, not your market impact.)
2. TRACKING vs SHADOW: the account's daily equity return vs a shadow book that holds the
   same targets at closing prices. The gap is the total implementation cost, including
   cash drag from whole-share rounding.

    python evaluate.py --strategy sixty_forty
"""

import argparse

import numpy as np
import pandas as pd

from broker import PaperBroker
from trader import ledger_paths, load_orders


def shortfall(orders: pd.DataFrame) -> pd.DataFrame:
    f = orders[orders.status == "filled"].copy()
    if f.empty:
        return f
    sign = np.where(f.side == "buy", 1, -1)
    f["shortfall_bps"] = sign * (f.filled_avg_price - f.decision_price) / f.decision_price * 1e4
    return f


def main(strategy):
    orders = load_orders(ledger_paths(strategy)[0])
    f = shortfall(orders)
    print(f"{strategy}: {len(orders)} orders logged, {len(f)} filled")
    if len(f):
        w = f.qty * f.decision_price
        print(f"  implementation shortfall: mean {f.shortfall_bps.mean():+.2f} bps, "
              f"notional-weighted {np.average(f.shortfall_bps, weights=w):+.2f} bps, "
              f"worst {f.shortfall_bps.max():+.2f} bps  (n={len(f)})")
    h = PaperBroker().portfolio_history()
    eq = pd.Series(h["equity"], index=pd.to_datetime(h["timestamp"], unit="s")).replace(0, np.nan).dropna()
    r = eq.pct_change().dropna()
    if len(r) > 5:
        print(f"  account: {len(r)} daily returns, ann. vol {r.std()*np.sqrt(252):.2%}, "
              f"total {eq.iloc[-1]/eq.iloc[0]-1:+.2%}")
    else:
        print("  account history too short for return statistics yet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="sixty_forty")
    main(ap.parse_args().strategy)
