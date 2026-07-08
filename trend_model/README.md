# trend_model — cross-asset time-series momentum (TSMOM)

Trend-following sleeve in futures. The one anomaly with a century of out-of-sample
support (Moskowitz-Ooi-Pedersen 2012), built to the same lookahead-safe,
survivorship-aware standard as the rest of the repo. See `../TREND_FINDING.md`
for the full result, caveats, **and the 2026-07 addendum that downgrades it**.

## Result (summary)
Dev-selected equity-index + rates book on yfinance "=F" series: **DEV Sharpe
+0.61 → OOS (2025+) +0.38**, full sample +120% / 15% vol / −20% maxDD. The naive
all-27-instrument book nets ~0 — the trend premium over 2015-2026 lives in
equities and rates, not commodities/FX.

> **VALIDATION UPDATE (2026-07): dev-robust, OOS unresolved.** Rerunning the
> identical construction on roll-free total-return ETF proxies in excess-of-cash
> terms (`validate_etf_proxy.py`) gives **dev +0.31 / OOS −0.19** (book corr
> +0.77). The dev premium survives the data change; the OOS number does not —
> it depends on which data source you believe (=F vs roll-free: OOS +0.38 vs
> −0.19). Per-instrument =F drift vs clean excess returns reaches +1.7%/yr (NQ);
> the ±15% clip never catches normal-sized roll gaps. "First OOS survivor"
> is therefore too strong; the honest claim is a real dev-period trend premium
> with an OOS sign that free data cannot resolve. Details in TREND_FINDING.md.

## Method (lookahead-safe throughout)
At each monthly rebalance t, using only data through t:
1. **Signal** = mean over 3/6/12-month lookbacks of `sign(P[t]/P[t-L]-1)` ∈ [−1,1].
2. **Per-instrument inverse-vol sizing**: `w_i = signal_i · 0.10 / σ_i`, σ_i from a
   trailing 60-day window strictly before t (floored at 6% ann., capped |w|≤0.25).
3. **Book vol targeting** to 12% ann. using only realised book returns from periods
   < t (leverage capped at 3x).
4. Net of 5 bps/turnover. P&L and vol both use **clipped daily returns** (±15%) so a
   front-month roll gap contributes neither risk nor fake P&L.

## Universe
Dev-selected to equity-index (ES/NQ/YM/RTY) + rates (ZN/ZB/ZF/ZT) futures. The
other four asset classes (FX, metals, ag, energy) had negative trend Sharpe in the
development period (≤2024) and are excluded — decision made with zero OOS info; see
`config.yaml` comments and `TREND_FINDING.md`.

## Data caveat
yfinance front-month continuous ("=F") series — not back-adjusted, so roll gaps are
noise (defended by clipping, which only catches the monster gaps). This caveat has
now been **tested, not just disclosed**: `validate_etf_proxy.py` replicates the book
on roll-free ETF excess returns and finds the OOS result is data-source-dependent
(see summary above). Free back-adjusted futures don't exist (Quandl CHRIS is dead;
Norgate/CSI are paid) — buying clean data is the mandatory next step before sizing
real money, and until then this is a paper sleeve with an asterisk.

## Run
```
python run.py                        # canonical eq+rt book → results/backtest.csv
python run.py --oos-start 2025-01-01 # OOS-only
python validate_etf_proxy.py         # roll-free ETF replication (the 2026-07 audit)
```

## Vol-sizing experiment (2026-07): no change adopted

Pre-registered comparison of four sizing-vol estimators (`vol_experiment.py`;
bar: QLIKE win with DM t > 2 AND dev Sharpe within 0.05 AND tracking not
worse). Forecast skill for next-21d vol, 946 common forecasts:
GARCH(1,1) best (QLIKE −4.005 vs rolling60 −3.877, DM t = +1.68), EWMA-0.94
second (t = +1.25), **causal changepoint/segmented vol WORSE than the plain
60d window (t = −1.10)** — discarding pre-break data costs more variance than
the bias it removes, a clean negative result for segmentation-style ideas
even at the task most favorable to them. Economically the book is
indistinguishable across estimators (full Sharpe 0.56–0.58; book-level vol
targeting washes out per-instrument sizing differences) — a robustness
property worth having on record. GARCH missed the significance bar (1.68 < 2)
on 118 monthly observations → baseline stays; artifacts in
`results/vol_experiment_*.csv`.

## Clean-data pipeline (built 2026-07, awaiting a Databento key)

The path to resolving the OOS question, ready to run once `DATABENTO_API_KEY`
is in `.env` (pay-as-you-go; daily bars only, cost-preflighted, typically
covered by signup credits):

```
pip install databento
python fetch_databento.py        # per-contract daily bars → cache/contracts_daily.parquet
python build_backadjusted.py     # volume-crossover roll + Panama adjustment
                                 #   → cache/backadjusted_returns.parquet + roll_calendar.csv
# then uncomment data.backadjusted_returns in config.yaml and rerun run.py
```

The stitcher emits *tradeable* daily returns (point change of the held
contract over the prior actual close — no roll gap ever enters signal or
P&L) and a roll calendar for audit. `build_backadjusted.py --self-test`
verifies the roll/adjustment logic on synthetic contracts; the run.py
consumption path is regression-tested to leave the yfinance book unchanged
when no clean data is present.
