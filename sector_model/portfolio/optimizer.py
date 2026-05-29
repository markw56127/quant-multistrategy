"""
Portfolio construction: translate alpha scores into long-only position weights.

Strategy:
  - Pick the top N stocks by alpha score
  - Size by inverse volatility (equal risk contribution per position)
  - Scale gross exposure continuously by HMM bull probability
  - Remainder sits in cash (no margin, no shorting)

Exposure scaling:
  gross = MIN_EXPOSURE + (MAX_EXPOSURE - MIN_EXPOSURE) * bull_prob

  This avoids the hard threshold problem where the portfolio snaps from
  100% to 35% exposure as soon as one state tips to bear. Instead, exposure
  drifts smoothly as regime evidence accumulates. With bull_prob=1.0 the
  portfolio is fully deployed; with bull_prob=0.0 it sits at 35% (floor).
"""

import numpy as np
import pandas as pd
from loguru import logger


MIN_EXPOSURE = 0.35   # floor: still deployed even in worst bear
MAX_EXPOSURE = 1.00   # ceiling: fully deployed in pure bull


def construct_weights(
    scores:    pd.Series,
    vol:       pd.Series,
    bull_prob: float,
    n_long:    int   = 8,
    max_pos:   float = 0.20,
) -> pd.Series:
    """
    Build a long-only weight vector from cross-sectional alpha scores.

    Args:
        scores    : alpha score per stock (higher = more attractive long)
        vol       : realized vol per stock (inverse-vol sizing)
        bull_prob : HMM posterior probability of bull regime (0–1)
        n_long    : number of long positions to hold
        max_pos   : hard cap on any single position weight
    Returns:
        weights   : Series summing to <= gross_target. No short positions.
    """
    gross_target = MIN_EXPOSURE + (MAX_EXPOSURE - MIN_EXPOSURE) * float(bull_prob)

    ranked = scores.rank(ascending=False)
    longs  = ranked[ranked <= n_long].index

    if len(longs) == 0:
        logger.debug("No valid scores — holding cash")
        return pd.Series(0.0, index=scores.index)

    v     = vol.reindex(longs).fillna(vol.mean())
    inv_v = 1.0 / (v + 1e-8)
    raw_w = inv_v / inv_v.sum()
    weights = pd.Series(0.0, index=scores.index)
    weights[longs] = raw_w * gross_target

    weights = weights.clip(upper=max_pos)
    if weights.sum() > 0:
        weights = weights / weights.sum() * gross_target

    logger.debug(
        f"Portfolio: longs={longs.tolist()} | bull_prob={bull_prob:.2f} "
        f"| gross={weights.sum():.2f} | cash={1 - weights.sum():.2f}"
    )
    return weights
