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


def buffered_long_short_weights(
    scores: pd.Series,
    prev_w: pd.Series,
    quantile: float = 0.20,
    buffer_mult: float = 2.0,
) -> pd.Series:
    """
    Long-short quintile spread with a hold buffer (turnover reduction).

    Entry band is the plain top/bottom `quantile`; a name already held stays
    in the book until it leaves the wider top/bottom `quantile * buffer_mult`
    band (2x is the standard practitioner convention, cf. MSCI index buffer
    zones). This trades away the churn of names oscillating around the
    quantile boundary while keeping the same signal. Each side is normalized
    to gross 1.0, so the book stays dollar-neutral.

    Requires quantile * buffer_mult <= 0.5 so the two hold bands cannot overlap.
    """
    s = scores.dropna()
    if len(s) < 10:
        return pd.Series(dtype=float)
    if quantile * buffer_mult > 0.5:
        raise ValueError("quantile * buffer_mult must be <= 0.5 (bands would overlap)")

    n_entry = max(1, int(len(s) * quantile))
    n_hold  = max(1, int(len(s) * quantile * buffer_mult))
    ranked  = s.sort_values(ascending=False)

    top_entry = set(ranked.index[:n_entry])
    top_hold  = set(ranked.index[:n_hold])
    bot_entry = set(ranked.index[-n_entry:])
    bot_hold  = set(ranked.index[-n_hold:])

    # Names that left the scoreable universe drop out automatically (& s.index)
    prev_longs  = set(prev_w[prev_w > 0].index) & set(s.index)
    prev_shorts = set(prev_w[prev_w < 0].index) & set(s.index)

    longs  = top_entry | (prev_longs & top_hold)
    shorts = bot_entry | (prev_shorts & bot_hold)

    w = pd.Series(0.0, index=s.index)
    w[list(longs)]  = +1.0 / len(longs)
    w[list(shorts)] = -1.0 / len(shorts)
    return w


def turnover(prev: pd.Series, new: pd.Series) -> float:
    """One-sided turnover between two weight vectors."""
    all_idx = prev.index.union(new.index)
    p = prev.reindex(all_idx).fillna(0.0)
    n = new.reindex(all_idx).fillna(0.0)
    return float((n - p).abs().sum())
