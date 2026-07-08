# carry_model — cross-asset futures carry (eq + rates)

**Status:** DEAD (2026-07) — failed its pre-registered bar on the first and
only run. Negative result recorded below; do not iterate on this universe.

## Pre-registered test (bar fixed before running, nothing fitted)

Carry is the second futures anomaly with deep multi-asset evidence (Koijen,
Moskowitz, Pedersen & Vrugt 2018) and is historically near-uncorrelated to
TSMOM — the textbook complement to `trend_model`. This sleeve tested the
version of it computable from **free data** on the same dev-selected
equity-index + rates universe:

- **equity carry** = trailing-12m distribution yield of the tracking ETF
  (SPY/QQQ/DIA/IWM, distributions ÷ unadjusted price) − 3m T-bill
- **rates carry** = constant-maturity Treasury yield at the contract tenor − 3m
  T-bill (term spread — the Fama-Bliss bond-carry proxy; FRED DGS2/5/10/30)
- signal = sign(carry), time-series; sizing / vol targeting / costs inherited
  from `trend_model` **verbatim** (nothing re-fit)
- **Bar (pre-set):** dev (≤2024) net Sharpe ≥ 0.3 AND OOS (2025+) > 0 AND
  monthly corr to trend_model < 0.5

## Result: FAIL

| book | dev (2016-2024) | OOS (2025-26) |
|---|---|---|
| carry (all 8) | **+0.08** | **−1.20** |
| equity-carry only | +0.18 | −1.14 |
| rates-carry only | −0.38 | −0.60 |

Monthly corr to trend_model: **−0.04** — the diversification claim held, but a
negative-return stream diversifies you into losses: adding carry 50/50 would
have dragged the OOS book from +0.38 to −0.88.

## Why it failed (post-mortem, not an excuse to re-tune)

- **US-only time-series carry is two macro bets, not a carry portfolio.**
  KMPV's result is cross-sectional across ~50 global instruments; with 8 US
  instruments, "carry" collapses to (a) is the equity financing spread
  positive, (b) is the Treasury curve steep. That is an inverted-curve timing
  rule, and it has been on the wrong side since 2022: bills > dividend yields
  kept the sleeve short equity futures through the 2023-26 bull (−14% to −23%
  per year, five years running).
- **The rates leg never worked even in dev** (−0.38) — the term spread's sign
  is too slow-moving to be a monthly timing signal on four correlated
  duration points of one curve.
- The honest conclusion mirrors `statarb_model`: the anomaly is real in the
  literature, but **the free-data, US-only expression of it carries no edge**.
  A real carry sleeve needs global futures term-structure data (paid).

Caveat recorded at run time: 2025-26 is not a pure OOS window for a sleeve
built in mid-2026; but no parameter here was fitted to any period — every
number is inherited from trend_model or is a definition from the literature.

## Run

```bash
cd carry_model
python run.py
python run.py --oos-start 2025-01-01
```

Reuses `../trend_model/cache` for futures prices; fetches FRED yields
(keyless CSV endpoint) and ETF distributions via yfinance, then caches.
