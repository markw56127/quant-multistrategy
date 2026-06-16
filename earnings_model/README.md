# Earnings PEAD Model

**Status:** v1 — survivorship-free, market-neutral, execution-honest (2026-06).

## Results (v1, 2016-2024, monthly rebalance, NET of costs + borrow)

| | v0 (pre-correction) | **v1 (honest)** |
|---|---------------------|-----------------|
| Total (net) | +39.2% | **+18.5%** |
| Sharpe (20d grid) | 0.50 | **0.51** |
| Sharpe (monthly grid, combiner) | 0.55 | **0.31** |
| Max DD | −19.2% | **−16.7%** |

**2026-06 corrections** (see ../LOOKAHEAD_FINDING.md):

1. *Strict filing-date entry.* Eligibility was `ann_date <= rebalance_date`,
   but ann_date is the raw SEC filing date (the claimed +1-day shift in
   _quarterly_series never existed) and filings often land after the close —
   so same-day entry captured drift that wasn't executable. Now `< rd`.
2. *Borrow costs.* 1%/yr general-collateral fee on short notional
   (~9 pts of total return over 9 years).

Roughly half the v0 net return was these two artifacts. What remains is
still a real, positive, market-neutral drift signal — just a thinner one.

- 17,473 earnings announcements across 563 tickers.
- Built survivorship-free from day one (shared/universe_pit.py).

**Cost sensitivity matters here.** Weekly rebalancing churned a 100-name book
(0.59 turnover/wk → 3%/yr drag → net Sharpe collapsed to 0.25). The 60-day
drift doesn't need weekly trading; monthly rebalancing cut turnover ~4× and
recovered net Sharpe to 0.50. PEAD is a real signal but turnover-sensitive.

**SUE = seasonal random walk** (no analyst data): expected EPS = same fiscal
quarter prior year; SUE = unexpected / trailing-8-quarter std. Self-consistent
across the FY/quarterly XBRL reporting wrinkle.

## Original plan (below) — realised


## Concept

Post-Earnings Announcement Drift (PEAD): stocks with large positive earnings
surprises continue to drift upward for 20-60 trading days after the
announcement, and large negative surprises drift down. Documented since Ball &
Brown (1968), still works because most institutional capital reacts too slowly
and there are limits to arbitrage.

## Signal

```
surprise = (actual_EPS - consensus_estimate) / pre_announcement_price_volatility
```

Standardised Unexpected Earnings (SUE) is the classic formulation. Enter long
on the top decile of positive surprises, short the bottom decile, hold 20-60
days, exit.

## Why this fits our infrastructure

- SEC EDGAR (`sector_model/data/sec_edgar.py`) already gives us exact filing
  dates — the event timestamp PEAD requires.
- We already compute a naive consensus proxy (trailing 4-quarter average) in
  `_build_edgar_features`. Real analyst consensus would be better.

## Missing piece

- **Consensus estimates.** yfinance has sparse/unreliable analyst estimates.
  Free options: Zacks (scraping, fragile), Finnhub free tier (limited),
  Nasdaq Data Link. Paid: I/B/E/S, FactSet.
- The naive trailing-average proxy works but is weaker — it can't capture
  guidance revisions between quarters.

## Key references

- Ball & Brown (1968) — original PEAD documentation
- Bernard & Thomas (1989) — PEAD persists, drift magnitude
- Chordia & Shivakumar (2006) — PEAD vs price momentum interaction

## Architecture sketch

- Event-driven, not calendar-rebalanced
- Universe: full S&P 500 (or Russell 1000)
- Position entry on announcement+1, exit on announcement+40
- Overlapping positions (many stocks announce in the same 2-week window)
- Market-neutral via long top / short bottom surprise deciles
