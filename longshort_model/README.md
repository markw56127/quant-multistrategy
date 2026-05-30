# Long-Short Market-Neutral Wrapper

**Status:** Idea / a technique to apply to other models, not a standalone model

## Concept

Everything we've built so far is long-only and benchmarked against SPY, which
means we carry full market beta. In a market-neutral long-short construction,
you buy the top-ranked names AND short the bottom-ranked names in equal dollar
amounts. The market exposure cancels out.

## Why it matters

- Removes systematic market risk → Sharpe ratios jump. Our long-only sector
  model gets OOS Sharpe ~1.05; the same signal as long-short would likely be
  1.5-2.0 because you're no longer exposed to whether the market goes up.
- The 2022 problem partially dissolves: if your signal correctly ranks stocks
  *within* a falling market, shorting the bottom decile makes money even when
  everything is down.
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
