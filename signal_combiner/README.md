# signal_combiner — linear vs XGBoost bake-off

Does a more flexible model combine the five factor signals (value, momentum,
quality, low_vol, size) into a better cross-sectional return forecast than the
naive equal-weight composite the factor sleeve already trades? Four combiners,
identical point-in-time features, identical purged/embargoed walk-forward CV.

**Verdict (2026-06): no. XGBoost does not beat the regularized linear model, and
learned linear combiners do not robustly beat the naive composite — they are
regime-fragile.** The simple equal-weight composite remains the right default.

## Result

Purged + embargoed expanding walk-forward (min 36mo train, 1mo embargo), monthly
top/bottom-20% long-short, GROSS of costs. Split dev (<2025) vs true OOS (2025+):

| model | DEV IC (t) | DEV L/S Sharpe | OOS IC (t) | OOS L/S Sharpe |
|---|---|---|---|---|
| EW composite | +0.004 (0.3) | −0.04 | +0.001 (0.1) | −0.17 |
| OLS          | −0.005 (−0.3)| −0.14 | +0.048 (1.5) | +1.04 |
| Ridge        | −0.005 (−0.3)| −0.15 | +0.047 (1.5) | +1.04 |
| **XGBoost**  | +0.004 (0.4) | +0.11 | +0.011 (0.6) | +0.44 |

## How to read it (don't get excited by the OOS +1.04)

- **The headline answer is robust: XGBoost loses.** It trails ridge/OLS in OOS
  (+0.44 vs +1.04) and barely clears EW in dev. Added flexibility does not help at
  equity-factor signal-to-noise — variance dominates, trees fit noise. This is the
  whole reason real equity quant runs on linear/Fama-MacBeth combiners.

- **The linear OOS pop is a regime, not a discovered edge.** The *same* ridge is
  −0.15 over the 71-month dev window and +1.04 over 18 OOS months. The learned
  coefficients (value +0.0015, quality +0.0007, momentum −0.0009) are economically
  sensible — long value/quality, short momentum — so what you are seeing is the
  value/quality premium being *out* of favor 2016–2024 and *in* favor 2025–26. The
  combiner reweights factors; it cannot create edge that is not in them.

- **Over the reliable (4×-longer) dev sample, fitting HURT:** OLS/ridge (−0.14/
  −0.15) underperformed the naive equal-weight composite (−0.04). Non-stationary
  factor premia make learned weights stale. The naive composite has no weights to
  overfit, which is exactly why it is robust.

- **18 months is not enough to overturn 71.** OOS IC t-stat ≈ 1.5 (not significant
  at 5%); OOS L/S Sharpe SE ≈ ±0.82. The point estimate of +1.04 is ~1.3 SE from
  zero. We do not promote ridge over the composite on this evidence.

## Takeaway

The right combiner at this SNR is the simplest one. XGBoost is the wrong tool here
and the experiment proves it cleanly on properly-leakage-controlled CV — the most
credible thing one can show about ML in equities is *when it does not help.* What
little OOS edge appears is the value/quality factor premium already documented in
`../OOS_FINDING.md`, repackaged through a reweighting, not new alpha.

## Files / run
```
python build_panel.py            # build (or --rebuild) the cached factor panel
python bakeoff.py                # run the 4-way CV bake-off → results/bakeoff.csv
```
Panel: 56,915 stock-months × 126 months × 594 names, reusing factor_model's exact
survivorship-free universe + EDGAR fundamentals + sector-neutral z-scores, so the
bake-off varies only the combiner.
