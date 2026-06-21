"""
Fama-MacBeth (1973) cross-sectional factor-premia estimation.

The canonical academic-quant procedure, and the specific thing missing from the
rest of the repo (the bake-off ran a POOLED regression; Fama-MacBeth runs a fresh
cross-sectional regression each period and tests the time series of slopes):

  1. For each month t, regress the cross-section of forward returns on the factor
     exposures:   r_{i,t+1} = a_t + Σ_k λ_{k,t} · z_{k,i,t} + ε_{i,t}
     The slope λ_{k,t} is the realized "factor premium" earned that month.
  2. The estimate of each premium is the time-series mean of λ_{k,t}; its
     significance is a t-stat on that mean, using NEWEY-WEST (HAC) standard errors
     so serial correlation / heteroskedasticity in the monthly premia don't
     overstate significance.

Inputs are the cached factor panel (sector-neutral z-scores) from signal_combiner,
so the exposures are identical to what the factor sleeve trades.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PANEL = Path(__file__).resolve().parent.parent / "signal_combiner" / "cache" / "panel.parquet"
FEATS = ["value", "momentum", "quality", "low_vol", "size"]
OOS = pd.Timestamp("2025-01-01")


def _nw_tstat(series: pd.Series, lags: int = 3) -> tuple:
    """Mean of a premium series and its Newey-West (HAC) t-stat."""
    y = series.dropna().values
    X = np.ones((len(y), 1))
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def fama_macbeth(panel: pd.DataFrame, feats=FEATS) -> pd.DataFrame:
    """Per-month cross-sectional regressions → time series of factor premia."""
    rows = {}
    for dt, g in panel.groupby("date"):
        X = sm.add_constant(g[feats].fillna(0.0))
        y = g["fwd_ret"].values
        if len(g) < len(feats) + 5:
            continue
        beta = sm.OLS(y, X).fit().params
        rows[dt] = beta
    return pd.DataFrame(rows).T.sort_index()        # index=date, cols=[const]+feats


def report(panel: pd.DataFrame = None):
    panel = panel if panel is not None else pd.read_parquet(PANEL)
    lam = fama_macbeth(panel)

    print(f"\n{'='*78}\nFama-MacBeth factor premia  (monthly cross-sectional regressions)")
    print(f"{'='*78}")
    print("Premium = mean monthly slope (return per +1 z-score); t via Newey-West (3 lag)\n")
    segments = {"FULL": lam,
                "DEV  (<2025)": lam[lam.index < OOS],
                "OOS  (2025+)": lam[lam.index >= OOS]}
    hdr = f"{'factor':<10}" + "".join(f"{s:>20}" for s in segments)
    print(hdr)
    print(f"{'':<10}" + "".join(f"{'premium    t-stat':>20}" for _ in segments))
    out = {}
    for f in FEATS:
        line = f"{f:<10}"
        for seg, df in segments.items():
            prem, t = _nw_tstat(df[f])
            # annualize the monthly premium for readability
            line += f"{prem*12:>+11.2%}{t:>+9.2f}"
            out[(f, seg)] = (prem * 12, t)
        print(line)
    # annualized premium shown; t-stat is on the monthly mean (scale-free)
    print(f"\n(premia annualized ×12; |t|>1.96 ≈ 5% significant. n_months: "
          f"full={len(lam)}, dev={segments['DEV  (<2025)'].shape[0]}, "
          f"oos={segments['OOS  (2025+)'].shape[0]})")

    res = pd.DataFrame(
        {f: {seg: out[(f, seg)] for seg in segments} for f in FEATS}).T
    res.to_csv(Path(__file__).resolve().parent / "results" / "fama_macbeth.csv")
    return lam


if __name__ == "__main__":
    report()
