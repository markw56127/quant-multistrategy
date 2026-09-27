# paper_trading — forward testing on an Alpaca PAPER account

**Status:** built and tested 2026-09-26 (dry-run verified, 7 tests). **Not yet connected:**
it needs your Alpaca paper keys (setup below). No scheduler is installed.

## Why paper trading, and what it can and can't tell us

- **It CAN** give a **forward out-of-sample test**: every day is data nobody fit to, which
  solves the limited-OOS-budget problem. It also **measures implementation shortfall**:
  each order logs its decision price, and `evaluate.py` compares that to the fill. That
  is the execution cost a close-to-close backtest never sees. And it exercises the full
  operational pipeline before any real money would be involved.
- **It CAN'T** simulate *your own* market impact. Paper orders never reach the real book,
  so fills are simulated against quotes. At small size, impact is negligible and the
  spread and timing are what matter, and those it does capture.

## Safety rails

- `broker.py` is **paper-only by construction**. The base URL is pinned to
  `paper-api.alpaca.markets`, and anything else raises (tested).
- The default mode is a **dry run** (prints the plan). Submitting requires `--live-paper`
  *and* keys.
- `client_order_id = {strategy}-{date}-{symbol}`. Alpaca rejects duplicates, so a double
  run on one day can't double-trade.
- Whole-share market orders, sells before buys. Target quantities round toward zero, so
  buys never exceed equity (tested).

## Strategies

No alpha candidate has passed a bar yet, so the first strategies are **premia
baselines** from `../premia_portfolio`, used to validate the pipeline:

| id | what | rebalance |
|---|---|---|
| `sixty_forty` | 60% SPY / 40% IEF | monthly (first run of each month) |
| `risk_parity_vt` | Track A risk parity on 9 ETFs, daily-1y covariance, 10% vol target, capped at 1× | monthly |

One strategy per paper account. Alpaca allows several paper accounts, each with its own
keys. A future strategy that passes a pre-registered bar gets added to `strategies.py`
and its own account.

## Setup (one time, about 5 minutes)

1. Create a free account at alpaca.markets and switch to the **Paper** dashboard.
2. Generate **paper** API keys. Put them in `paper_trading/.env`, which is git-ignored:
   ```
   APCA_API_KEY_ID=PK...
   APCA_API_SECRET_KEY=...
   ```
3. Dry run (no keys needed): `python trader.py --strategy sixty_forty`
4. First real paper run, during market hours: `python trader.py --strategy sixty_forty --live-paper`
5. Later: `python evaluate.py --strategy sixty_forty`

## Daily schedule (optional, after step 4 works)

The same launchd pattern as `../options_data`: a weekday 12:30 PT run, 30 minutes before
the close, so market orders fill near the closing price. Not installed yet. It will be
added once the keys are in place and a manual `--live-paper` run has succeeded.

## Files

| file | role |
|---|---|
| `broker.py` | paper-only Alpaca REST client (account, clock, positions, latest trade, orders) |
| `strategies.py` | target-weight functions |
| `trader.py` | daily run: reconcile → rebalance decision → plan → submit → ledger |
| `evaluate.py` | implementation shortfall and account statistics from the ledger |
| `test_trader.py` | safety and order-math tests (no network) |
| `ledger/{strategy}/` | append-only `orders.csv` and `runs.csv`, created on first `--live-paper` run |
