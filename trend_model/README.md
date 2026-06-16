# trend_model — cross-asset time-series momentum (TSMOM)

Trend-following sleeve in futures. The one anomaly with a century of out-of-sample
support (Moskowitz-Ooi-Pedersen 2012), built to the same lookahead-safe,
survivorship-aware standard as the rest of the repo. **It is the first sleeve here
that survived a true out-of-sample test** — see `../TREND_FINDING.md` for the full
result and caveats.

## Result (summary)
Dev-selected equity-index + rates book: **DEV Sharpe +0.61 → OOS (2025+) +0.38**,
full sample +120% / 15% vol / −20% maxDD. The naive all-27-instrument book nets
~0 — the trend premium over 2015-2026 lives in equities and rates, not commodities/FX.

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
noise (defended by clipping). The worst glitches were in the dropped commodity
contracts. Cleaner back-adjusted data is the right next step before sizing real money.

## Run
```
python run.py                        # canonical eq+rt book → results/backtest.csv
python run.py --oos-start 2025-01-01 # OOS-only
```
