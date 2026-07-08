# vix_model — VIX term-structure sleeve (volatility risk premium)

**Status:** SHELVED (2026-07) — failed its pre-registered bar on the single
run. The most informative part of the result is that the *timing signal
subtracts value* from the raw premium. Recorded below; no iteration.

## Pre-registered test

Signal: VIX/VIX3M ratio at the weekly close (threshold 1.0 — the theory
value, nothing fitted). Contango → long SVXY (−0.5x short-vol ETP);
backwardation → long VXX. Position = 12% vol budget / trailing 60d ETP vol,
capped at 100% notional; 10 bps per unit turnover; long-only, no borrow.
Dev 2019–2024 (both ETPs in their current form — VXX series B from 2018-01,
SVXY deleveraged 2018-02 — so no structural break inside the sample);
OOS 2025+. **Bar (pre-set): dev net Sharpe ≥ 0.5 AND OOS > 0 AND monthly
corr to trend_model < 0.5.**

## Result: FAIL

| book | dev (2019-24) | OOS (2025-26) | maxDD |
|---|---|---|---|
| timed (term-structure switch) | **+0.38** | **−0.01** | −32.5% |
| static short-vol (always SVXY, same sizing) | +0.49 | +0.20 | −23.8% |

Corr: +0.25 to trend book, +0.38 to SPY (short-vol is implicit equity beta).
The sleeve held SVXY 93% of weeks; the 7% spent in VXX — the entire point of
the switch — was net-negative whipsaw. The timing rule is worse than doing
nothing, in both windows.

## Honest reading

- **The VRP itself is real but thin here**: static −0.5x short-vol earns
  ~0.49 dev / 0.20 OOS — below every adoption bar in this repo, correlated
  to SPY, and with an understated left tail (see below).
- **The tail risk is not in the sample by construction.** The dev window
  starts 2019 because the instruments changed after Volmageddon (2018-02-05
  — the event that destroyed XIV and halved SVXY's leverage). A short-vol
  backtest that structurally cannot contain its own worst-case event
  understates risk; sizing on these numbers would be exactly the mistake
  this repo exists to prevent.
- **Index-ratio timing at a weekly grid is too slow/blunt**: term-structure
  inversions are brief and violent; by the time VIX > VIX3M at a Friday
  close, most of the VXX payoff has happened. A daily-grid or futures-basis
  version is a *different, pre-registrable* experiment — but it would need
  VX futures data and a reason to believe the timing edge exists at all,
  which this result argues against.

## Run

```bash
cd vix_model
python run.py
python run.py --oos-start 2025-01-01
```
