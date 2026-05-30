# Multi-Factor Equity Model

**Status:** ACTIVE — this is the model we're building next

## Concept

The workhorse of quantitative equity investing (AQR, Dimensional, LSV, BlackRock
Scientific Active Equity). Rank stocks on several well-documented, academically
validated factors, combine into a composite score, build a portfolio from the
spread.

The core insight that fixes our `sector_model` problem: **factors have
different regime strengths and partially cancel each other's weaknesses.**
Value works when momentum crashes (2022); momentum works when value stagnates
(2020-2021). Running them together gives a smoother equity curve than any single
factor — diversification across *factors*, not just stocks.

## The factors (each with academic grounding)

| Factor | Definition | Reference |
|--------|-----------|-----------|
| **Value** | Book/Price, Earnings/Price, sales/EV | Fama & French (1992) |
| **Momentum** | 12-1 month return | Jegadeesh & Titman (1993) |
| **Quality** | ROE, earnings stability, low accruals, gross profitability | Novy-Marx (2013) |
| **Low Volatility** | inverse of trailing volatility | Ang et al. (2006) |
| **Size** | small-cap premium (use carefully) | Banz (1981) |

## Methodology

1. **Universe:** Russell 1000 or full S&P 500 (start with S&P 500 — reuse the
   `sector_model` universe and EDGAR pipeline).
2. **Factor computation:** cross-sectional z-score each factor at each rebalance.
3. **Combination:** equal-weight composite to start (don't optimise weights —
   that's where overfitting creeps in). Equal-weighting factors is remarkably
   robust (DeMiguel et al. 2009, "1/N" result).
4. **Portfolio:** long top quintile, optionally short bottom quintile
   (see `longshort_model`).
5. **Neutralisation:** sector-neutralise so you're not just making a sector bet
   (e.g., value tilts you into financials/energy by construction). Demean each
   factor within sector before combining.

## What to reuse from sector_model

- `data/sec_edgar.py` — fundamentals (EPS, revenue, shares, margins)
- `data/universe.py` — price fetching, missing-data handling
- `data/sp500.py` — universe + GICS sector labels (needed for sector-neutralisation)
- Walk-forward backtest skeleton from `train_sp500.py`

## What's genuinely different from sector_model

- No two-layer rotation. Factors ARE the alpha; no separate macro layer.
- Cross-sectional ranking across the *whole universe* at once, not within sectors.
- Sector-neutralisation instead of sector-selection.
- Fama-MacBeth regression option: regress forward returns on factor exposures
  cross-sectionally each period → get factor returns AND t-stats directly.
  This tells you which factors are actually paying off and is the standard
  academic validation method.

## First milestone

Reproduce the classic result: a value+momentum+quality equal-weight composite,
long-short, sector-neutral, should give a Sharpe of 0.8-1.2 gross of costs over
2015-2024. If we can't reproduce the textbook result, something is wrong with
the pipeline.
