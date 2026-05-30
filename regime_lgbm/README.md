# Regime-Conditional LightGBM

**Status:** Idea / not started

## Concept

Direct fix for the failure we hit in `sector_model`: a single LightGBM trained
across all history learns the wrong lesson from dominant periods. Trained over
2018-2021, it learned "high momentum = crash" from the COVID drawdown and then
systematically picked losers in 2022-2023 (IC t-stat went to -0.93).

The fix professional shops use: **train separate models per regime, apply the
one matching current conditions.**

## Approach

1. Label every historical day with a regime using the HMM we already have
   (`sector_model/signals/sector.py`, SectorRegimeModel) or a simpler
   VIX + yield-curve clustering:
   - Regime A: low-VIX / risk-on / momentum-dominant
   - Regime B: high-VIX / risk-off / quality-and-defensive-dominant
   - Regime C: rising-rate / value-rotation

2. Train a separate LightGBM on each regime's data only.

3. At prediction time, detect current regime and apply the matching model.
   Optionally blend models weighted by regime probability (soft assignment)
   to avoid discontinuities at regime boundaries.

## Why it should work

The 2022 problem was never that the features lacked signal (IC t-stat was 1.85
in-sample). It was that the *relationship* between features and returns flips
sign across regimes. A regime-conditional model captures exactly that.

## Risk

Data fragmentation: splitting 10 years across 3 regimes leaves ~3 years per
model. May be too little for LightGBM. Mitigations: soft assignment (every
sample contributes to every model, weighted by regime probability), or use
regime as an interaction feature rather than a hard split.

## Reuses

- `sector_model/signals/cross_section.py` — CrossSectionalModel (LightGBM wrapper)
- `sector_model/signals/sector.py` — HMM regime model
- All SEC EDGAR features already built
