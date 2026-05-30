"""
Portfolio construction from factor composite scores.

Long-short market-neutral quintile spread:
  - Long  the top quintile  (highest composite scores), equal-weighted
  - Short the bottom quintile (lowest composite scores), equal-weighted
  - Dollar-neutral: equal capital long and short, so market beta ≈ 0

This is the standard academic factor-portfolio construction. The long-short
spread isolates the factor signal from market direction, which is exactly
what we want — and it's why factor Sharpe ratios are reported on the spread,
not the long leg alone.

A long-only variant (top quintile vs benchmark) is also provided for
comparison against the sector_model, which was long-only.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def long_short_weights(
    scores: pd.Series,
    quantile: float = 0.20,
) -> pd.Series:
    """
    Build dollar-neutral long-short weights from composite scores.

    Returns a weight Series (positive = long, negative = short) summing to ~0,
    with gross exposure 2.0 (1.0 long + 1.0 short).
    """
    s = scores.dropna()
    if len(s) < 10:
        return pd.Series(dtype=float)

    n_side = max(1, int(len(s) * quantile))
    ranked = s.sort_values(ascending=False)
    longs  = ranked.index[:n_side]
    shorts = ranked.index[-n_side:]

    w = pd.Series(0.0, index=s.index)
    w[longs]  = +1.0 / n_side
    w[shorts] = -1.0 / n_side
    return w


def long_only_weights(
    scores: pd.Series,
    quantile: float = 0.20,
) -> pd.Series:
    """Top-quintile long-only, equal-weighted, gross exposure 1.0."""
    s = scores.dropna()
    if len(s) < 10:
        return pd.Series(dtype=float)
    n = max(1, int(len(s) * quantile))
    longs = s.sort_values(ascending=False).index[:n]
    w = pd.Series(0.0, index=s.index)
    w[longs] = 1.0 / n
    return w


def turnover(prev: pd.Series, new: pd.Series) -> float:
    """One-sided turnover between two weight vectors."""
    all_idx = prev.index.union(new.index)
    p = prev.reindex(all_idx).fillna(0.0)
    n = new.reindex(all_idx).fillna(0.0)
    return float((n - p).abs().sum())
