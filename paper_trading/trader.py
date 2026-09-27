"""
Daily paper-trading run: reconcile past orders → decide whether to rebalance → plan →
submit → log.

    python trader.py --strategy sixty_forty            # dry run if no keys (prints the plan)
    python trader.py --strategy sixty_forty --live-paper   # submits to the Alpaca PAPER account

Rules:
  * Rebalance on the first run of each calendar month (and on the very first run).
    Otherwise only reconcile and log.
  * Orders are whole-share market orders (time_in_force=day), sells before buys. Trades
    under MIN_TRADE_USD are skipped.
  * client_order_id = {strategy}-{date}-{symbol}. Alpaca rejects duplicate ids, so a
    double run on the same day cannot double-trade.
  * Every order logs its decision price (latest trade at submit time). The fill is
    reconciled on later runs, which is what evaluate.py uses to measure implementation
    shortfall.
"""

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from broker import PaperBroker, have_keys
from strategies import STRATEGIES

ROOT = Path(__file__).resolve().parent
MIN_TRADE_USD = 50.0
DRY_EQUITY = 100_000.0
ORDER_COLS = ["date", "client_order_id", "strategy", "symbol", "side", "qty", "decision_price",
              "status", "filled_qty", "filled_avg_price", "filled_at"]


def plan_orders(targets: dict, equity: float, positions: dict, prices: dict, min_usd: float = MIN_TRADE_USD):
    """Pure function: whole-share orders that move `positions` to `targets` (weights of
    equity). Target quantities round toward zero, so buys never exceed equity. Symbols
    held but not targeted are sold. Returns [(symbol, side, qty)] with sells first."""
    orders = []
    for sym in sorted(set(targets) | set(positions)):
        px = prices[sym]
        tgt = math.floor(targets.get(sym, 0.0) * equity / px)
        d = tgt - positions.get(sym, 0.0)
        qty = int(abs(d))
        if qty == 0 or qty * px < min_usd:
            continue
        orders.append((sym, "sell" if d < 0 else "buy", qty))
    return sorted(orders, key=lambda o: o[1] != "sell")


def ledger_paths(strategy):
    d = ROOT / "ledger" / strategy
    d.mkdir(parents=True, exist_ok=True)
    return d / "orders.csv", d / "runs.csv"


def load_orders(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=ORDER_COLS)


def reconcile(broker, orders: pd.DataFrame) -> pd.DataFrame:
    open_ = ~orders.status.isin(["filled", "canceled", "expired", "rejected"])
    for i in orders.index[open_]:
        o = broker.order_by_client_id(orders.at[i, "client_order_id"])
        orders.at[i, "status"] = o["status"]
        orders.at[i, "filled_qty"] = float(o.get("filled_qty") or 0)
        orders.at[i, "filled_avg_price"] = float(o["filled_avg_price"]) if o.get("filled_avg_price") else None
        orders.at[i, "filled_at"] = o.get("filled_at")
    return orders


def needs_rebalance(orders: pd.DataFrame, today: str) -> bool:
    if orders.empty:
        return True
    return str(orders.date.max())[:7] != today[:7]


def run(strategy: str, submit: bool):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fn = STRATEGIES[strategy]
    orders_path, runs_path = ledger_paths(strategy)
    orders = load_orders(orders_path)

    if not submit:
        import yfinance as yf
        targets = fn(pd.Timestamp(today))
        closes = yf.download(list(targets), period="5d", auto_adjust=True, progress=False)["Close"].iloc[-1]
        prices = {s: float(closes[s]) for s in targets}
        plan = plan_orders(targets, DRY_EQUITY, {}, prices)
        print(f"DRY RUN ({strategy}, ${DRY_EQUITY:,.0f} hypothetical, no positions). Targets:")
        for s, w in targets.items():
            print(f"  {s:<5} {w:6.1%}  @ {prices[s]:.2f}")
        print("Orders that would be sent:")
        for s, side, q in plan:
            print(f"  {side:<4} {q:>5} {s:<5} (~${q * prices[s]:,.0f})")
        print(f"cash left ≈ ${DRY_EQUITY - sum(q * prices[s] for s, side, q in plan if side == 'buy'):,.0f}")
        return

    broker = PaperBroker()
    orders = reconcile(broker, orders)
    clock, acct = broker.clock(), broker.account()
    equity = float(acct["equity"])
    action = "reconcile-only"
    if clock["is_open"] and needs_rebalance(orders, today):
        targets = fn(pd.Timestamp(today))
        positions = broker.positions()
        syms = set(targets) | set(positions)
        prices = {s: broker.latest_trade(s) for s in syms}
        plan = plan_orders(targets, equity, positions, prices)
        new = []
        for sym, side, qty in plan:
            cid = f"{strategy}-{today}-{sym}"
            o = broker.submit_market(sym, qty, side, cid)
            new.append({"date": today, "client_order_id": cid, "strategy": strategy, "symbol": sym, "side": side,
                        "qty": qty, "decision_price": prices[sym], "status": o["status"],
                        "filled_qty": 0.0, "filled_avg_price": None, "filled_at": None})
        orders = pd.concat([orders, pd.DataFrame(new, columns=ORDER_COLS)], ignore_index=True)
        action = f"rebalanced ({len(new)} orders)"
    elif not clock["is_open"]:
        action = "market closed: reconcile-only"
    orders.to_csv(orders_path, index=False)
    run_row = pd.DataFrame([{"ts_utc": datetime.now(timezone.utc).isoformat(), "equity": equity,
                             "cash": float(acct["cash"]), "action": action}])
    run_row.to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False)
    print(f"{strategy}: equity ${equity:,.2f}; {action}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="sixty_forty", choices=list(STRATEGIES))
    ap.add_argument("--live-paper", action="store_true",
                    help="submit to the Alpaca PAPER account (requires keys); otherwise dry run")
    a = ap.parse_args()
    if a.live_paper and not have_keys():
        raise SystemExit("no Alpaca paper keys found; see paper_trading/README.md")
    run(a.strategy, submit=a.live_paper)
