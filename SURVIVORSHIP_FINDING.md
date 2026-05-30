# Survivorship Bias — The Big Finding

Date: 2026-05-29

## Summary

The factor_model exposed survivorship bias; retrofitting the correction into
sector_model revealed that **most of sector_model's apparent alpha was
survivorship bias, not signal.**

## Evidence 1 — Factor model size factor

Long-short size factor, three universes:

| Universe | Size return | Sharpe |
|----------|-------------|--------|
| Current S&P 500 constituents (biased) | +187% | 1.44 (positive 9/9 years!) |
| Point-in-time membership only | +97% | 0.87 |
| + survivorship-free prices (delisted kept) | +23% | 0.26 |

A "+187% factor, positive every single year" is impossible for a real
long-small/short-large factor in the mega-cap-dominated 2015-2024. It was
bias. Corrected → +23%, matching the literature (size premium ~dead post-1980s).

## Evidence 2 — Sector model, clean vs biased

Same two-layer model, current-constituent universe vs survivorship-free:

| Metric | v1 (biased) | survivorship-free |
|--------|-------------|-------------------|
| Total | +148.4% | +60.8% |
| Excess vs SPY | **+46.9%** | **−40.8%** |
| Sharpe | 1.46 | 0.69 |
| IR | 0.74 | −0.19 |
| IC t-stat | 1.85 | 0.57 |

The headline excess **flipped sign**: +47% → −41%. On a clean universe the
sector model underperforms SPY.

Year-by-year (survivorship-free): 2020 +23.5%, 2021 −17.0%, 2022 −15.1%,
2023 +7.3%, 2024 −16.2% excess. Some real signal (2020, 2023) but swamped
by the years where survivor stock-picking was actually losing.

## Interpretation

- **Layer 1 (sector rotation)** operates on sector ETFs — no survivorship
  issue. It is probably still fine.
- **Layer 2 (within-sector stock picking)** is where the bias lived. Picking
  the top momentum names among *survivors* looks like skill because the
  losers (which a real model would also have picked and lost on) were deleted
  from the universe.

## Caveats

- ~19% of historical members (145/755) still have no free price data
  (full delistings/bankruptcies) — residual upward bias remains even in the
  "survivorship-free" number, so the true clean result may be slightly worse.
- Non-trading returns filled with 0 for the rolling-OLS panel; bounded
  distortion (membership filter excludes those stocks from selection).

## Consequence for all future work

Every model must use the shared point-in-time universe from day one
(`shared/universe_pit.py`). Current-constituent backtests are not just
optimistic — they can invert the sign of the result.
