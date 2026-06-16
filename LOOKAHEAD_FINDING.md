# Lookahead Bias — The Second Big Finding

Date: 2026-06-11 (companion to SURVIVORSHIP_FINDING.md)

## Summary

The pinn_rl model's results were entirely lookahead artifacts. Three distinct
leaks were found and fixed; after the fix the model has no edge and is
retired. A milder same-day-execution leak was also found and fixed in the
PEAD model, cutting its honest net return roughly in half.

## The pinn_rl leaks

1. **One-day return credit (env + backtest engine).** The agent observed
   features built from day t's close and was paid day t's return — a return
   already realized inside its own observation. With 5-day rebalancing, 1 in
   5 days of P&L was known at decision time. Same bug in both
   `rl/environment.py` and `backtest/engine.py`.
2. **Future returns in the feature matrix.** `post_earn_ret_5d` (the realized
   NEXT 5 days of returns) was placed at announcement+1 and forward-filled
   into the VAE input. The code comment said "valid for TRAINING" — it wasn't;
   it flowed into the live state.
3. **Today's snapshot broadcast to history.** `forwardEps` / `pegRatio` from
   a current yfinance call were constant features across 2015-2024.

## Evidence

| Run | Sharpe | Max DD | Verdict |
|-----|--------|--------|---------|
| "OOS" with leaks (oos_results.csv) | **16.9** | −0.6% | impossible → leak signature |
| In-sample, engine leak absent (old code path) | −0.16 | −12% | nothing there |
| Retrained after all fixes (2026-06) | training Sharpe −1 to −3.8, no learning trend | — | **no edge, retired** |

Synthetic proof of the engine fix: a perfect same-day-foresight signal earned
+372% under the old rebalance ordering and +1% under the fixed ordering.

## The PEAD timing leak (milder, same family)

`ann_date <= rebalance_date` allowed entry at the close of the SEC filing
date, but filings often post after hours — and the docstring's claimed
"+1 day shift in _quarterly_series" did not actually exist in the code.
Combined with adding 1%/yr short borrow fees: net total +39.2% → +18.5%,
monthly-grid Sharpe 0.55 → 0.31. Still real, much thinner.

## Honest state of the book (2026-06)

factor value+quality (0.75) + PEAD v1 (0.31), expanding-window risk parity:
**Sharpe 0.87, maxDD −7.5%**, net, survivorship-free, no full-sample fitting
anywhere (combiner weights were also fixed: expanding-window inverse-vol,
not full-sample). Insider sleeve v0 too weak to include (see its README).

## Rules going forward

1. A position decided with day-t information earns returns from t+1, never t.
   Event timestamps are raw filing dates until proven otherwise — require
   strictly-before, don't trust docstrings about shifts.
2. Any backtest Sharpe > ~2 on daily equities is a bug hunt, not a result.
3. No feature may contain values dated after the row it sits in.
4. Nothing (weights, normalizers, vols) may be fit on the full sample.
