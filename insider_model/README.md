# Insider Cluster-Buy Model (SEC Form 4)

**Status:** SHELVED (2026-06) — two honest attempts, both ~zero. Negative
result recorded; do not iterate further on this universe.

## v1 results (routine filter + 126d window + no $ floor)

Total **−5.8%**, Sharpe **−0.01**, maxDD −27.4% (monthly grid, net,
SPY-hedged). Worse than v0 (+7.3%, 0.13). The CMP filter and extra breadth
did not surface signal — consistent with the conclusion that the insider
effect lives in small caps, where information asymmetry is real and analyst
coverage is thin. In the S&P 500 it is arbitraged flat.

**Why we stop here instead of tuning more:** two parameterizations have been
tried against the same 9 years of data. A third, fourth, fifth attempt that
finally "works" would be multiple-testing artifact, not alpha. The honest
options are a different universe (Russell 2000 — needs survivorship-free
small-cap data we don't have for free) or a different sleeve entirely.

The infrastructure (Form 345 fetcher, cluster signal, routine filter) is
sound and reusable if small-cap data ever becomes available.

## v0 results (2016-2024, SPY-hedged, net)

Total +7.3%, Sharpe 0.13, maxDD −25.9%, corr(SPY) +0.14, avg book only
18 names. Correlations to the other sleeves were ideal (factor +0.13,
PEAD −0.07) but the standalone edge was too thin: adding it DROPPED the
3-sleeve risk-parity Sharpe to 0.68 vs 0.87 for the 2-sleeve book.
Consistent with the literature: the insider effect concentrates in small
caps; the S&P 500 is the hardest place to harvest it. **Not in the
combined book until it clears ~0.3 standalone.**

## v1 changes

- **Routine-buyer filter** (Cohen, Malloy & Pomorski 2012): drop buys from
  insiders who bought in the same calendar month 3 years running —
  opportunistic trades carry the information. `signal.drop_routine: true`.
- **Breadth**: window 63→126 trading days, $50k value floor removed
  (v0's 18-name book was mostly noise-vulnerable concentration).

## Concept

Corporate insiders (officers, directors, 10%+ holders) must report their
trades on Form 4 within 2 business days. Open-market *purchases* are a
documented predictive signal (Lakonishok & Lee 2001; Cohen, Malloy & Pomorski
2012 "Decoding Inside Information"): insiders buy with their own money when
they believe the stock is cheap. Sells are noise (diversification, taxes,
option exercises) and are ignored.

The strongest formulation is **cluster buying**: multiple distinct insiders
buying within a short window. One insider buying can be idiosyncratic;
three insiders buying the same month is conviction.

## Why this sleeve

- Reuses everything: PIT universe (`shared/universe_pit.py`), survivorship-free
  prices, the PEAD backtest skeleton.
- Driven by *behavior*, not price or fundamentals → plausibly uncorrelated
  with both the value+quality and PEAD sleeves (the whole point of sleeve #3:
  combined risk-parity Sharpe was 1.05 on two sleeves).
- Free, structured, timestamped data: SEC publishes quarterly flattened
  Form 3/4/5 data sets (2006Q1–present).

## Data

Quarterly ZIPs from
`https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{YYYY}q{Q}_form345.zip`
(verified live 2026-06). Each contains `SUBMISSION.tsv` (accession → ticker,
filing date), `NONDERIV_TRANS.tsv` (transaction code, shares, price), and
`REPORTINGOWNER.tsv` (owner CIK → distinct-buyer counting).

We keep only: `TRANS_CODE == 'P'` (open-market purchase) with
`TRANS_ACQUIRED_DISP_CD == 'A'`, non-derivative table only.

## Signal

At each rebalance date `rd` (monthly — turnover lesson from PEAD):

```
window   = trailing signal_window trading days, filings STRICTLY before rd
n_buyers = distinct reporting-owner CIKs with a 'P' purchase in window
value    = total $ purchased in window
signal   = n_buyers (cluster size), tiebreak by value
```

Long the qualifying names (`n_buyers >= min_buyers`, `value >= min_value`),
equal-weighted, capped at `max_names`. Market-neutralised by shorting SPY
against the long book (config `hedge: spy`), with borrow cost charged.

Timing discipline (lesson from PEAD): the event timestamp is the SEC
**filing date**, and eligibility requires `filed < rd` — filings can land
after the close, so same-day execution is not assumed.

## Run

```bash
cd insider_model
python run.py                          # full backtest
python run.py --oos-start 2022-01-01   # OOS evaluation
```

First run downloads ~40 quarterly ZIPs (~10 MB each) → cached. To skip
re-downloading prices/membership, copy from the PEAD model first:

```bash
mkdir -p cache && cp ../earnings_model/cache/sp500_historical_components.csv \
  ../earnings_model/cache/prices_sf_*.parquet cache/
```

## Honest expectations

Long-only insider strategies historically earn ~3-7%/yr over market on the
cluster-buy subset, concentrated in small caps — inside the S&P 500 the
effect is weaker (more analyst coverage, less information asymmetry). Net
Sharpe in the 0.3-0.6 range would be a success for a third sleeve; what
matters is correlation to the other two, not standalone Sharpe.

## Key references

- Lakonishok & Lee (2001) — Are Insider Trades Informative?
- Cohen, Malloy & Pomorski (2012) — Decoding Inside Information
- Jeng, Metrick & Zeckhauser (2003) — Estimating the Returns to Insider Trading
