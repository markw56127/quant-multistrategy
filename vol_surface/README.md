# vol_surface — daily implied-volatility surfaces from the banked chains

**Status:** built 2026-10-08. Runs daily via launchd (`com.markwang.forward-tests`, 15:30 PT).
6 synthetic-truth tests in `test_surface.py`.

## What it does (per chain, per expiry): `surface.py`

1. **Quote filter:** bid > 0, ask > bid, relative spread < 50%, mid ≥ 0.05, at least 5
   days to expiry. For SPX, the PM-settled `SPXW` series is kept when the AM-settled
   `SPX` shares a third-Friday date: they are different contracts at different prices.
2. **Forward and discount from put–call parity:** a robust regression of C − P = D·F − D·K
   over strikes within ±5% of spot (3-MAD outlier pass). This absorbs dividends, borrow
   and rates with no assumptions. If the implied rate falls outside 0–10% (short-dated
   quote noise), D is fixed at a 4% rate and the forward still comes from parity.
3. **Own implied vols:** Black-76 on the forward, from OTM mids. Yahoo's IV column is
   not used: 8% of SPX contracts carry IV < 1%, and it differs from ours by a median 1.4
   vol points.
4. **No-arbitrage checks:** monotonicity and convexity of call prices in strike, counted
   only when a violation exceeds the bid/ask spread; Gatheral's butterfly-density
   condition on the fitted smile; calendar monotonicity of total variance. The fit checks
   cover only the quoted k-range, never SVI's extrapolation.
5. **SVI per expiry:** raw SVI, w(k) = a + b(ρ(k − m) + √((k − m)² + s²)), fitted in vol
   space with a soft-L1 loss (robust to stale quotes) and positivity as a penalty.
6. **Daily summary:** at 30 and 90 days, with total variance interpolated linearly in T at
   fixed k, **only inside each expiry's quoted k-range** (else NaN):
   - ATM vol
   - **skew** = IV(−sd) − IV(+sd)
   - **curvature** = ½[IV(−sd) + IV(+sd)] − ATM, with **sd = σ_ATM√T** (±1 standard
     deviation of the expected move, ≈ 16-delta put vs call)
   - **term slope** = ATM90 − ATM30
7. **Capture flag:** `regular` when the chain was fetched during that day's session,
   else `after_hours`. After-hours chains have stale or one-sided quotes, so treat them
   as diagnostics only.

## Data problems found and fixed while building it

| problem | symptom | fix |
|---|---|---|
| AM- and PM-settled SPX listed under one expiry | inflated contract counts; D = 0.91 | keep `SPXW` |
| stale deep-strike quotes | parity D = 0.93 on 2026-12-31 (≈30% rate) | ±5% band plus robust pass → D = 0.989 |
| mid-quote noise on 5-point strikes | 3,041 "convexity violations" | count only violations larger than the spread → 161 |
| checks run on SVI extrapolation | 37/42 butterfly failures | check only the quoted range → 0 |
| positivity enforced by rejection | short expiries left unfitted | penalty inside the fit → 42/42 fitted |
| summary read from SVI's extrapolated wings | AT&T "skew" 95 vol points; fixed ±10% often unquoted | quoted-range guard plus vol-scaled moneyness → 3 vol points |
| after-hours snapshots (2026-09-30, 10-06) | SPY ATM 20% vs ~12.6%; no stock surfaces | `capture` flag; signals use regular captures only |

## Outputs (write-only per day: safe on the iCloud-synced Desktop)

- `results/params/{date}.csv`: one row per underlying (SPX, SPY, QQQ, IWM and the 50
  stocks) with the summary parameters and fit/arbitrage diagnostics
- `results/svi/{date}/{underlying}.csv`: per-expiry forward, discount factor, SVI
  parameters, fit error and checks

`python run_daily.py` processes any snapshot day not yet built (`--rebuild` refits
everything; `--resummarize` recomputes summaries from the saved SVI fits without refitting). Expect about 1 minute per index-only day and about 10 minutes per day with
stock chains.

## First readings

- **SPX and SPY agree** to within about 0.1 vol point on ATM vol, skew and curvature
  every day, from independent contracts (European cash-settled vs American ETF). That is
  the pipeline's best validation.
- **Index skew ≈ 4.5–5 vol points at ±1 sd vs single-stock median ≈ 1.4:** index skew
  prices correlation and crash risk (Bakshi–Kapadia–Madan 2003). Stock smiles are
  flatter. QQQ has the steepest index skew.
- **The term structure slopes upward** (SPX ATM 30d ≈ 13% → 90d ≈ 14%): a calm-market
  shape. IWM is nearly flat.
- **Variance risk premium:** over 13 days, SPY realized 10.1% (5-minute bars plus the
  overnight gap; 38% of variance is overnight) against VIX at 15.4%, so implied ran 1.5×
  realized.
