# Sector Rotation Sleeve (long-short sector ETFs)

**Status:** DEAD (2026-07) — v0 failed its first true-OOS test (Sharpe −0.67),
and three pre-registered reformulations each failed their pre-set bars.
Recorded below; do not iterate further on this universe.

## Reformulation experiments (2026-07, `experiments_v1.py`)

To answer "is the construction the problem, or the idea?", four runs were
pre-registered — variants and pass bars fixed before running, one run each,
no iteration (insider_model precedent). Data extended through 2026-06;
dev = 2015-2024, OOS = 2025+ (never seen during development).

| Variant | Bar (pre-set) | dev | OOS | Verdict |
|---|---|---|---|---|
| v0 blend, first true-OOS run | OOS Sharpe > 0 | +0.13 | **−0.67** | **FAIL** |
| v1 vol-adjusted momentum, macro dropped | dev ≥ 0.3 AND OOS > 0 | −0.21 | −1.15 | FAIL |
| v2 per-sector TSMOM (sign of own 12-1, inv-vol) | dev ≥ 0.3 AND OOS > 0 AND corr(trend) < 0.6 | 0.27 | +0.83 | FAIL (dev 0.27 < 0.3) |
| v3 dual momentum (long-only top-3, cash gate) | beat SPY Sharpe in dev AND OOS | 0.80 vs SPY 0.94 | 1.17 vs 1.14 | FAIL (dev) |

Reading, honestly:
- **v0's OOS failure (−0.67) is the decisive result** — same pattern as PEAD:
  a marginal dev signal that inverts out of sample.
- **Cross-sectional sector momentum is dead, not mis-constructed.** Vol-adjusting
  the rank (v1) made it *worse*. The rank IC underneath is t ≈ 0.7 over 120
  months — there is no portfolio construction that rescues a zero IC.
- **v2 is the only near-miss**, and it is not a new edge: it is time-series
  momentum expressed on sector ETFs (dev corr to `trend_model` 0.42, OOS
  corr(SPY) 0.82 — mostly long-beta in the OOS window). The repo already
  harvests that premium in `trend_model` with better instruments. Its OOS
  strength *confirms* TREND_FINDING; it does not justify a redundant sleeve.
- **v3 confirms long-only rotation is beta**: it never beat buy-and-hold SPY
  in development (0.80 vs 0.94, corr 0.74+). The "switch into the winning
  sector" story, implemented long-only, collects market return minus costs.

Numbers drift slightly vs. the v0 record below (fresh yfinance adjusted
closes, window now includes 2015 and 2025-26); conclusions are unchanged.
Per-variant series: `results/experiment_<variant>.csv`.

## v0 result (2016-2024, monthly grid, net)

Total **−6.1%**, Sharpe **0.03**, vol 14.8%, maxDD −30.4%. Correlation to
factor_vq −0.25 (ideal) but the decision rule was standalone Sharpe ≥ ~0.3
AND low correlation; it fails the first condition decisively. Adding it
dropped the risk-parity book from 0.87 to 0.65. Shelved per the rule —
no parameter iteration, since the macro-beta table was already the
flagged overfitting risk and tuning it against the same history would
manufacture a result.

Post-mortem note: this is consistent with sector momentum's documented
decay post-2015 and the 2020/2022 sector whipsaws (energy, tech). It also
retroactively suggests Layer 1 was contributing less to the original
sector_model than assumed — the survivorship post-mortem cleared it of
*bias*, not of weakness.

## Concept

Extracts Layer 1 of `sector_model` — the part the survivorship post-mortem
cleared ("operates on sector ETFs — no survivorship issue... probably still
fine") — and converts it from a long-only allocator into a dollar-neutral
spread: long the top-N sector ETFs, short the bottom-N, by composite score.

## Signal (reused from sector_model/signals/sector_rotation.py)

1. **Sector momentum (12-1 month)** — Moskowitz & Grinblatt (1999): past
   12-month return skipping the last month, cross-sectionally z-scored.
2. **Macro tilt** — theory-grounded (NOT fitted) sector betas to the yield
   curve and VIX, applied to trailing-252d z-scores of each macro series.
   Financials like steep curves; utilities/REITs hate rising rates;
   defensives like high VIX; cyclicals like low VIX.

`composite = 0.6 * z(momentum) + 0.4 * z(macro)` — weights inherited from
sector_model, not re-optimized here.

## Why it can't have the old problems

- **No survivorship bias possible**: the 11 SPDR ETFs are the universe;
  none ever delisted. XLC (2018) and XLRE (2015) enter when they have
  enough history — point-in-time by construction.
- **No same-day leak**: signal uses closes through the rebalance date;
  entry at that close (executable for ETFs near the close), P&L measured
  from the next bar onward via close-to-close forward returns.
- **Cheap to trade**: pennies-wide spreads, ~30bp GC borrow on shorts.

## Honest caveats

- The 0.6/0.4 weights and macro betas, while theory-motivated, were written
  while looking at the same 2015-2024 history the backtest runs on. The
  macro-beta table is the part most at risk of quiet overfitting. Treat
  the first OOS year as the real test.
- 11 assets, monthly → ~108 independent-ish bets over 9 years. Wide error
  bars on any Sharpe estimate; don't over-read a good number.
- Decision rule, set BEFORE running (same as insider_model):
  **standalone monthly-grid Sharpe ≥ ~0.3 AND low correlation to
  factor_vq/PEAD, else shelve.**

## Run

```bash
cd sector_rotation
python run.py
python run.py --oos-start 2022-01-01
```

First run downloads 11 ETFs + SPY + VIX/yields via yfinance (~seconds), then
caches. Output lands in `results/backtest.csv`; `combine_strategies.py`
picks it up automatically.
