# Earnings PEAD Model

**Status:** Idea / not started

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
