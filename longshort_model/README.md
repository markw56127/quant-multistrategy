# Long-Short Market-Neutral Wrapper

**Status:** Technique (REALISED in the factor/PEAD/trend sleeves), not a standalone model

> **CORRECTION (2026-06):** an earlier version of this doc claimed the long-only
> sector_model had "OOS Sharpe ~1.05" and that long-short would push it to "1.5–2.0".
> That 1.05 was a **survivorship-biased** artifact — once the universe was made
> point-in-time and delisted names retained, the sector_model's excess return over SPY
> flipped from +47% to −41% ([../SURVIVORSHIP_FINDING.md](../SURVIVORSHIP_FINDING.md)),
> and the genuinely market-neutral long-short books built afterward (factor +0.25 OOS,
> trend +0.38 OOS, PEAD −0.69 OOS — [../OOS_FINDING.md](../OOS_FINDING.md)) land nowhere
> near 1.5–2.0. The construction technique below is sound and was applied to those
> sleeves; the inflated projection was not.

## Concept

The long-only sleeves carry full market beta. In a market-neutral long-short
construction, you buy the top-ranked names AND short the bottom-ranked names in equal
dollar amounts, so the market exposure cancels out.

## Why it matters

- Removes systematic market risk → isolates the cross-sectional signal from whether
  the market goes up or down (the factor, PEAD, and trend sleeves are all built this
  way). It raises *risk-adjusted* return only to the extent the ranking has real
  cross-sectional skill — it is not free Sharpe, as the OOS results above show.
- The 2022 problem partially dissolves: if the signal correctly ranks stocks *within*
  a falling market, shorting the bottom decile makes money even when everything is down.
- This is the standard construction for most quant equity funds.

## Construction

```
long_weight[i]  = +1/N for top-N ranked stocks
short_weight[i] = -1/N for bottom-N ranked stocks
gross_exposure  = 2.0 (1.0 long + 1.0 short)
net_exposure    = 0.0 (market neutral)
```

## Caveats

- Shorting has real-world frictions: borrow costs, hard-to-borrow names,
  short squeezes, margin requirements.
- Backtests must subtract borrow fees (typically 0.3-3% annualised, much
  higher for small/hot names).
- This is a *wrapper technique* — apply it to the factor_model or earnings_model
  rather than building it standalone.

## Apply to

`factor_model` first — multi-factor signals rank cleanly into long/short
deciles and the factor literature is built around long-short spreads.
