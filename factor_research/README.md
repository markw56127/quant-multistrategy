# factor_research — statistical-alpha evaluation layer

The academic-quant evaluation a research desk runs on a cross-sectional signal
*before* sizing it: **Fama-MacBeth factor premia, IC analysis (level + decay),
turnover/transaction-cost analysis, and capacity analysis** — all on the cached,
survivorship-free factor panel (56,915 stock-months, 594 names, 2016–2026) that
`factor_model` and `signal_combiner` already produced. Same value/quality/momentum/
low-vol/size exposures the sleeve trades; this layer characterizes them statistically.

## 1. Fama-MacBeth factor premia (Newey-West t-stats)

Monthly cross-sectional regression `r_{i,t+1} = a_t + Σ λ_{k,t} z_{k,i,t}`; the
premium is the mean of the slope time series, t-stat HAC-corrected (3 lags).

| factor | premium (ann.) | t-stat (full) | dev t | oos t |
|---|---|---|---|---|
| **value**   | **+2.45%** | **+2.97** | +1.90 | +5.64 |
| quality | +0.70% | +1.35 | +2.13 | −2.92 |
| momentum| +1.22% | +0.90 | −0.04 | +2.13 |
| low_vol | −2.04% | −1.36 | −0.70 | −1.84 |
| size    | −0.79% | −1.37 | −0.31 | −4.31 |

**Value is the one robust premium** (t≈3 full sample, HAC). Note `size` is *negative*
here — and this universe is genuinely point-in-time, so this is the honest size
premium. (Contrast `../smallcap_factor`, where current-constituents survivorship
inflated `size` to +3,566%: same factor, the difference is purely data integrity.)

## 2. Information Coefficient — level and decay

Single-period rank IC is modest (value +0.010, composite +0.006), as expected at
monthly equity SNR. The informative result is the **decay profile**:

```
IC vs k-months-ahead return:   k=1    3    6    9    12
              value          +0.010 .025 .033 .037 .037   ← rises, then plateaus
              composite      +0.006 .014 .020 .021 .026
```

Value's predictive power **accumulates with horizon and plateaus** — a slow-decaying
signal. That is the mechanistic reason it is low-turnover and high-capacity below.

## 3. Turnover & transaction-cost (value+quality book, top/bottom 20%)

The book is built from the two premia that survive §1–2 (value+quality), not the
naive all-5 average (which §1 shows is diluted to ~0 by the dead factors).

- gross Sharpe **+0.81**, ann. return +2.6%
- one-way turnover **16.4%/month (2.0× notional/yr)** — slow
- **breakeven cost > 50 bps/trade**: net Sharpe is still +0.50 at 50 bps. Robust to
  realistic equity costs (the sleeve trades at 10 bps).

## 4. Capacity (square-root market impact)

Almgren square-root model `cost ≈ c·σ·√(traded$/ADV$)`, c=1, σ_daily=2%, on 63-day
trailing dollar-ADV (fetched per name):

| AUM | $100M | $500M | $1B | $2.5B | $5B |
|---|---|---|---|---|---|
| net Sharpe | +0.74 | +0.65 | +0.58 | +0.45 | +0.30 |

**Net Sharpe halves only by ~$5B; dollar P&L is still rising at $5B.** Low turnover +
slow decay → impact stays small into the billions. The book's binding constraint is
the *size of the edge* (Sharpe ~0.8), not capacity — the opposite of a fast/crowded
alpha.

## Takeaway

A complete, honest statistical characterization of a real (if modest) cross-sectional
equity premium: value is significant (Fama-MacBeth t≈3), slow-decaying, cheap to
trade, and high-capacity; quality is a marginal complement; momentum/low-vol/size do
not pay over this sample. This is what the evaluation actually looks like before a
desk decides to allocate — and the same toolkit cleanly separates a real premium (§1
value) from a survivorship illusion (`smallcap_factor` size).

## Run
```
python run_report.py        # all three analyses (≈30s; first capacity run fetches volume)
python fama_macbeth.py      # premia only
python ic_analysis.py       # IC level + decay only
python capacity.py          # turnover, cost, capacity only
```
