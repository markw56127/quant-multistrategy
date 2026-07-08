# Trend sleeve: the first edge in this repo that survived out-of-sample

**Date:** 2026-06-15
**Sleeve:** `trend_model/` — cross-asset time-series momentum (TSMOM) in futures.
**Headline:** A dev-selected equity-index + rates trend book went **DEV Sharpe
+0.61 → OOS (2025+) Sharpe +0.38**, the first positive true-OOS result in the
project. But read the caveats — 18 months and a concentrated source.

> ## ADDENDUM (2026-07): roll-free replication does NOT confirm the OOS number
>
> The known data caveat ("=F series are not back-adjusted") was tested instead
> of just disclosed. `trend_model/validate_etf_proxy.py` reruns the IDENTICAL
> construction on roll-free total-return ETF proxies (SPY/QQQ/DIA/IWM,
> SHY/IEI/IEF/TLT) in excess-of-cash terms — same signal, sizing, vol target,
> costs. Result (monthly corr between books +0.77):
>
> | book | DEV | OOS |
> |---|---|---|
> | futures =F (as reported below) | +0.61 | +0.38 |
> | ETF proxy, roll-free | **+0.31** | **−0.19** |
> | futures signal / ETF P&L | +0.86 | +0.14 |
> | ETF signal / futures P&L | +0.33 | +0.03 |
>
> Reading: the **dev-period trend premium survives the data change** (positive
> in all four signal/P&L combinations, 0.31–0.86), but the **OOS survival does
> not** — it spans −0.19 to +0.38 depending purely on data source. Per-instrument
> drift between =F and clean excess returns runs to **+1.7%/yr (NQ)**; the ±15%
> clip never catches normal-sized roll gaps, so they flow into both signal and
> P&L. Equity trend remains the stronger leg in both sources (futures eq-only
> +0.50 dev / +0.49 OOS; ETF eq-only +0.26 / −0.14); rates trend is negative OOS
> in both.
>
> **Status downgrade: "survived OOS" → "dev-robust, OOS unresolved."** The
> honest statement is now: dev Sharpe 0.3–0.6 and OOS somewhere in [−0.2, +0.4],
> with the true value unknowable on free data. Resolution requires genuinely
> back-adjusted futures series (Norgate/CSI-class, paid) — free sources
> (Quandl CHRIS, etc.) are discontinued. Until then this sleeve is a tracked
> paper sleeve with a data asterisk, not a deployable edge. The original
> write-up below is retained unchanged as the record of what was believed
> on 2026-06-15 and why.

## Why we built it

Every prior strategy fished US large-cap equities, the most efficient pond on
earth, and the equity-factor book failed its true-OOS test (0.75 → −0.05, see
`OOS_FINDING.md`). Time-series momentum is the one anomaly with a century of
out-of-sample support (Moskowitz-Ooi-Pedersen 2012; Hurst-Ooi-Pedersen 2017) and
is structurally diversifying. It is the most defensible "real-money" candidate
available to a retail account, and a positive cross-asset result to balance the
honestly-reported equity negatives.

## What we found

**1. The naive all-27-instrument book nets ~0 (Sharpe −0.28).** Equal risk across
every asset class lets the dead sleeves drag down the live ones.

**2. The trend premium is concentrated, and the split is visible on dev data
alone.** Per-asset-class TSMOM Sharpe, decided on DEVELOPMENT data only (≤2024):

| class | DEV Sharpe | OOS Sharpe |
|---|---|---|
| equity index | **+0.47** | +0.50 |
| rates        | **+0.46** | −0.40 |
| FX           | −0.10 | −0.92 |
| metals       | −0.41 | +1.00 |
| agriculture  | −0.20 | −0.49 |
| energy       | −0.05 | −1.86 |

Only equity and rates are positive in development. That selection uses **zero
OOS information** and is theory-consistent: equity and bond trend are the most-
cited, most-robust TSMOM sleeves. (Metals' +1.00 OOS is 18 months of noise off a
−0.41 dev base — exactly the kind of in-sample-blind we refuse to chase.)

**3. The dev-selected EQ+RT book held up out of sample.**

| book | DEV Sharpe | OOS Sharpe | full |
|---|---|---|---|
| all-27 naive | −0.29 | −0.20 | −0.28 |
| **EQ+RT (dev-selected)** | **+0.61** | **+0.38** | **+0.58** |

Full sample: +120% total, 15.0% vol, −20% maxDD, avg leverage 1.3x. Compare the
equity-factor book that died OOS (0.75 → −0.05): this one degraded from 0.61 to
0.38 — *within one standard error*, and **still positive**.

## Read the caveats (this is not a victory lap)

- **18 OOS months → SE on the Sharpe ≈ ±0.82.** +0.38 is "behaving as a ~0.5-0.6
  Sharpe sleeve would," not "proven." Do not size on the point estimate.
- **The OOS edge is concentrated in equity trend.** Rates trend was *negative*
  OOS (−0.40, rates chopped sideways in 2025-26); equity trend (+0.50 OOS)
  carried the book. A two-legged sleeve currently standing on one leg.
- **Selection risk remains.** Picking {eq, rt} from six classes is defensible
  (dev-only + strong priors) but is still a choice; the honest test is whether
  it persists as more OOS data accrues.
- **Free data.** Front-month continuous (yfinance "=F"), roll-gap defended by
  clipping daily returns to ±15% (a roll is not tradeable P&L). The worst roll
  glitches (CL=F +306% on 2020-04-20, NG=F's 23 spikes) were in the commodity
  contracts we dropped — convenient, and real. Equity/rates futures roll cleanly.

## What it means for the two goals

- **Resume:** the first OOS-surviving, cross-asset result in the repo — a strong,
  honest bullet that also demonstrates the dev/OOS discipline interviewers screen
  for. Selection was made on dev data, not mined on the test set.
- **Real money:** the most legitimate candidate found so far, but 18 months and a
  one-legged OOS mean it is "promote to a tracked paper sleeve and watch," not
  "deploy and size." The clean next step is better data (back-adjusted continuous
  contracts) to confirm the commodity/FX classes are genuinely dead vs. roll-noise
  casualties, and more OOS months on eq+rates.

## Reproduce
```
cd trend_model && python run.py                 # canonical eq+rt book
python run.py --oos-start 2025-01-01            # OOS-only rebalances
# To re-run the full all-27 book: uncomment the four excluded groups in config.yaml
```
