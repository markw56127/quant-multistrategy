# Split-adjustment lookahead in market cap: the dev value premium was an artifact

**Date:** 2026-09-26
**Found by:** reviewing the fundamentals pipeline before building new characteristics
**Audit:** `factor_research/split_audit.py` (non-destructive; writes a corrected panel
to `factor_research/cache/panel_splitfix.parquet`)

## The bug

`factor_model/factors.py` computes

```
market_cap = adjusted_price × EDGAR shares_outstanding
```

- **yfinance adjusted prices** divide every past price by all splits (and dividends)
  that happened *later*, up to the download date.
- **EDGAR `EntityCommonStockSharesOutstanding`** is the as-reported share count,
  with no adjustment.

So a stock that **will** split k:1 shows a past market cap k× too small:

| | implied market cap (original) | corrected | actual (approx.) |
|---|---|---|---|
| NVDA, 2017-06 | $2.1B | $85.9B | ~$85B |
| AMZN, 2017-06 | $25.1B | $501.5B | ~$480B |
| AAPL, 2017-06 | $185.3B | $798.7B | ~$800B |
| TSLA, 2017-06 | $3.7B | $55.9B | ~$55B |
| AMZN, 2023-06 | $61.8B | $1,236.5B | ~$1.2T |

B/P, E/P and S/P all divide by market cap, so future splitters look like deep
value. Companies split **after their price has risen**. The value factor was
therefore partly sorting on future returns. Dividend adjustment adds a smaller
version of the same effect (future high payers look cheaper). 146 names split
during the sample, and their mean value-score distortion at the start is 0.85σ,
against 0.33σ for the rest.

## The correction

```
raw_close(t)  = split-adjusted Close(t) × Π(splits after t)     # Close is not dividend-adjusted
shares_now(t) = shares_reported(filing f) × Π(splits in (f, t])
market_cap(t) = raw_close(t) × shares_now(t)
```

Applied to 588 of 594 names. Six fully delisted names have no Yahoo data and are
left as they were. Every other step of the pipeline is identical.

## Impact

Fama–MacBeth premia (annualized, Newey–West t):

| factor | dev (<2025) original | **dev corrected** | OOS (2025+) original | OOS corrected |
|---|---|---|---|---|
| **value** | +1.56%, t = +1.90 | **−0.60%, t = −0.51** | +7.74%, t = +5.64 | +7.04%, t = +4.37 |
| quality | +1.15%, t = +2.13 | +1.20%, t = +2.24 | −2.05%, t = −2.92 | −2.13%, t = −3.03 |
| size | −0.18%, t = −0.31 | −1.08%, t = −1.76 | −4.48%, t = −4.31 | −4.42%, t = −4.02 |
| momentum, low_vol | essentially unchanged (no market cap involved) | | | |

Quintile L/S gross Sharpe:

| book | dev original | **dev corrected** | OOS original | OOS corrected |
|---|---|---|---|---|
| value + quality | 0.87 | **0.31** | 0.53 | 0.40 |
| value alone | 0.39 | 0.15 | 1.74 | 1.42 |

**The mechanism, tested directly:** the component removed by the correction
(original value score − corrected) earns **+3.75%/yr, t = +2.98** in dev. That
is more than the entire original dev value premium. The dev value premium *was*
the lookahead.

OOS value survives the correction (t = 4.37). That fits the mechanism: few
splits happened between 2025 and the adjustment date, so the OOS window was
barely contaminated. But it is 18 months, and value had a strong 2025.

## What this invalidates (dev-period claims only)

- `factor_model`: the dev Sharpe 0.61–0.62 for value+quality.
- `factor_research`: "value is the one robust premium (t ≈ 3)". The full-sample t
  falls to 0.43. IC decay, turnover and capacity were all computed on the
  contaminated panel.
- `signal_combiner` bake-off inputs, and `OOS_FINDING.md`'s dev column for factor_vq.
- `smallcap_factor` uses the same market-cap construction; not yet re-audited.
- `dynamic_trading` M1–M6: every parameter and result was built on the
  contaminated panel. M5's "dev pass" is void.

What stands: the OOS numbers are close to right, momentum and low_vol are
unaffected, and every sleeve that doesn't use EDGAR market cap is unaffected.

## Also found

`data_fundamentals._build_factor_fundamentals` back-fills shares outstanding
before a company's first filing (`.ffill().bfill()`). That is lookahead too, but
it is negligible here. It only affects the window before a name's first 10-Q,
mostly post-2015 IPOs.

## Status: PATCHED (2026-09-26)

- `shared/market_cap.py` builds market cap as above. `factor_model/run.py`,
  `signal_combiner/build_panel.py` and `smallcap_factor/run.py` add it after
  fetching fundamentals. `factors.py` raises if `market_cap` is missing, so the old
  construction cannot silently return.
- `data_fundamentals.py` no longer back-fills shares. The cached EDGAR files
  still contain back-filled values, but those only cover dates before a name's
  first filing. S&P 500 inclusion requires 12 months of seasoning after IPO, so
  that window never falls inside the point-in-time universe.
- The rebuilt `signal_combiner/cache/panel.parquet` matches the audit panel to
  1e-15. The pre-fix panel is kept as `panel_presplitfix.parquet` so historical
  `dynamic_trading` numbers stay reproducible.

Rerun results:

| | before | **after** |
|---|---|---|
| factor_vq (value+quality) net Sharpe, dev / OOS | 0.62 / 0.25 | **0.16 / 0.25** |
| buffered factor_vq, dev / OOS | 0.44 / 0.92 | 0.03 / 0.55 |
| combined book (factor_vq + PEAD), dev / OOS | 0.75 / −0.05 | **0.34 / −0.15** |
| factor_research value IC (full) | +0.010 | **−0.004** |
| factor_research v+q gross Sharpe; capacity at $1B | 0.81; 0.58 | **0.32; 0.15** |
| signal_combiner, dev L/S Sharpe (ew / ridge / xgb) | positive | **−0.38 / −0.17 / −0.19** |
| smallcap_factor value / size L/S Sharpe | 0.44 / 3.21 | 0.31 / 2.92 (size is still survivorship) |

**Not yet re-audited:** `sector_model/data/sec_edgar.py` builds `log_market_cap`
the same way (price × as-reported shares). It is a feature in the older
sector_model and not in any scored sleeve, but any revival must switch to
`shared/market_cap.py`.
