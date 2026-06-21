"""
Information Coefficient (IC) analysis — predictive strength and its decay.

  - IC_t = cross-sectional Spearman rank correlation between a factor's exposure at
    t and the realized forward return. Mean IC measures raw predictive power; the
    IC information ratio (mean/std) and its t-stat measure reliability.
  - IC DECAY: IC of the exposure at t against the return realized k months later,
    for k = 1..12. A signal that decays fast must be traded fast (high turnover,
    capacity-limited); a slow-decaying signal is cheaper to harvest. This directly
    motivates the turnover and capacity analyses.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PANEL = Path(__file__).resolve().parent.parent / "signal_combiner" / "cache" / "panel.parquet"
FEATS = ["value", "momentum", "quality", "low_vol", "size"]


def _xs_ic(panel: pd.DataFrame, factor: str, target: str) -> pd.Series:
    """Monthly cross-sectional rank IC of `factor` vs `target`."""
    def ic(g):
        s = g[[factor, target]].dropna()
        if len(s) < 10:
            return np.nan
        return spearmanr(s[factor], s[target]).correlation
    return panel.groupby("date").apply(ic)


def _multi_horizon(panel: pd.DataFrame, kmax: int = 12) -> pd.DataFrame:
    """Add fwd_k columns: compounded forward return k rebalances ahead, per ticker."""
    wide = panel.pivot(index="date", columns="ticker", values="fwd_ret").sort_index()
    df = panel.copy()
    for k in range(1, kmax + 1):
        # product of (1+r) over the next k one-month forward returns, shifted to align
        fwdk = (1 + wide).rolling(k).apply(np.prod, raw=True).shift(-(k - 1)) - 1
        long = fwdk.stack(future_stack=True).rename(f"fwd{k}").reset_index()
        df = df.merge(long, on=["date", "ticker"], how="left")
    return df


def report(panel: pd.DataFrame = None):
    panel = panel if panel is not None else pd.read_parquet(PANEL)
    panel = panel.copy()
    panel["composite"] = panel[FEATS].mean(axis=1)

    print(f"\n{'='*78}\nInformation Coefficient analysis\n{'='*78}")
    print(f"{'factor':<11}{'mean IC':>9}{'IC IR':>8}{'IC t':>8}{'hit%':>7}   (monthly, n=126)")
    rows = {}
    for f in FEATS + ["composite"]:
        ics = _xs_ic(panel, f, "fwd_ret").dropna()
        n = len(ics)
        ir = ics.mean() / ics.std() if ics.std() > 0 else 0
        rows[f] = {"mean_ic": ics.mean(), "ic_ir": ir, "ic_t": ir * np.sqrt(n),
                   "hit": (ics > 0).mean()}
        print(f"{f:<11}{ics.mean():>+9.4f}{ir:>+8.2f}{ir*np.sqrt(n):>+8.2f}{(ics>0).mean():>6.0%}")

    # IC decay for the composite (and value, the strongest single factor)
    print(f"\n── IC decay (mean cross-sectional IC vs return k months ahead) ──")
    mh = _multi_horizon(panel, kmax=12)
    print(f"{'horizon kmo':<12}" + "".join(f"{k:>6}" for k in range(1, 13)))
    for f in ["composite", "value"]:
        decay = [ _xs_ic(mh, f, f"fwd{k}").dropna().mean() for k in range(1, 13) ]
        print(f"{f:<12}" + "".join(f"{d:>+6.3f}" for d in decay))
    print("\n(IC vs k-month-ahead return is cumulative; a flattening/plateau means the "
          "signal keeps predicting at longer horizons → slower decay → more capacity.)")

    pd.DataFrame(rows).T.to_csv(Path(__file__).resolve().parent / "results" / "ic_analysis.csv")


if __name__ == "__main__":
    report()
