"""Tests for the paper trader's safety rails and order math. No network, no keys needed."""

import pandas as pd

from broker import PaperBroker
from trader import plan_orders, needs_rebalance, ORDER_COLS
from evaluate import shortfall


def test_refuses_live_endpoint():
    for url in ["https://api.alpaca.markets", "https://paper-api.alpaca.markets.evil.com"]:
        try:
            PaperBroker(base_url=url)
        except ValueError:
            continue
        raise AssertionError(f"accepted non-paper URL {url}")


def test_plan_from_cash():
    o = plan_orders({"SPY": 0.6, "IEF": 0.4}, 100_000, {}, {"SPY": 500.0, "IEF": 95.0})
    assert o == [("IEF", "buy", 421), ("SPY", "buy", 120)]          # floor(40000/95)=421, 60000/500=120


def test_buys_never_exceed_equity():
    px = {"SPY": 771.35, "IEF": 97.13, "GLD": 250.7}
    o = plan_orders({"SPY": 0.5, "IEF": 0.3, "GLD": 0.2}, 10_000, {}, px)
    assert sum(q * px[s] for s, side, q in o if side == "buy") <= 10_000


def test_sells_first_and_exits_untargeted():
    o = plan_orders({"SPY": 1.0}, 100_000, {"TLT": 50, "SPY": 100}, {"SPY": 500.0, "TLT": 90.0})
    assert o[0] == ("TLT", "sell", 50) and o[1] == ("SPY", "buy", 100)


def test_small_trades_threshold():
    assert plan_orders({"SPY": 0.6}, 100_000, {"SPY": 119}, {"SPY": 500.0}, min_usd=1000) == []
    assert plan_orders({"SPY": 0.6}, 100_000, {"SPY": 119}, {"SPY": 500.0}) == [("SPY", "buy", 1)]


def test_monthly_rebalance_rule():
    empty = pd.DataFrame(columns=ORDER_COLS)
    assert needs_rebalance(empty, "2026-09-28")
    o = pd.DataFrame([{"date": "2026-09-28"}])
    assert not needs_rebalance(o, "2026-09-30")
    assert needs_rebalance(o, "2026-10-01")


def test_shortfall_sign():
    o = pd.DataFrame([
        {"status": "filled", "side": "buy", "qty": 10, "decision_price": 100.0, "filled_avg_price": 100.10},
        {"status": "filled", "side": "sell", "qty": 10, "decision_price": 100.0, "filled_avg_price": 99.90},
        {"status": "new", "side": "buy", "qty": 10, "decision_price": 100.0, "filled_avg_price": None}])
    f = shortfall(o)
    assert len(f) == 2 and (abs(f.shortfall_bps - 10.0) < 1e-9).all()   # both cost 10 bps
