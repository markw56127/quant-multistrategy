# Multi-Factor Equity Model

**Status:** v0 BUILT — value+quality composite, survivorship-corrected.

## Results (v0, survivorship-free, 2016-2024, long-short market-neutral)

| Composite | Total | Sharpe | Max DD | Notes |
|-----------|-------|--------|--------|-------|
| **value + quality** | **+33.1%** | **0.61** | −9.3% | positive 7/9 yrs, +6.6% in 2022 crash |
| value + quality + momentum | +5.2% | 0.14 | −10.3% | momentum drags |
| value only | +46.8% | 0.41 | −25.5% | higher return, much deeper drawdown |

Per-factor long-short Sharpes (survivorship-free):
value +0.41, quality +0.27, size +0.26, low_vol −0.16, momentum −0.27.

**Key finding — survivorship bias is real and large.** The size factor read
+187% (Sharpe 1.44, positive every single year) on the *current*-constituent
universe. After (a) point-in-time membership and (b) keeping delisted names'
prices, it collapsed to a realistic +23% (Sharpe 0.26) — matching the
literature that the size premium is ~dead post-1980s. Value and quality
survived the correction; they are the genuine signal.

**Known limitation:** ~19% of historical members (145/755) have no free price
data (full delistings/bankruptcies — disproportionately the worst outcomes),
so a residual upward bias remains. Proper fix needs CRSP/Compustat.

## Original plan (below) — mostly realised


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
