"""
Portfolio construction: translate alpha scores into long-only position weights.

Strategy:
  - Pick the top N stocks by alpha score
  - Size by inverse volatility (equal risk contribution per position)
  - Scale gross exposure by regime: full in bull, reduced in chop, defensive in bear
  - Remainder sits in cash (no margin, no shorting)

Why inverse-vol over equal-weight:
  High-beta semis (NVDA, AMD) would otherwise dominate an equal-weight book.
  Inverse-vol gives each position roughly equal *risk* contribution.

Why no short selling:
  Shorting individual names has unlimited downside and requires margin.  In a
  secular sector bull market, shorting the "worst" stock still loses money.
  Regime-based cash allocation achieves the same defensive effect without
  the tail risk.
"""

import numpy as np
import pandas as pd
from loguru import logger


# Gross equity exposure by regime (remainder is cash)
REGIME_EXPOSURE = {
    0: 0.35,   # bear  — mostly cash, small long book
    1: 0.70,   # chop  — moderate exposure
    2: 1.00,   # bull  — fully deployed
}


def construct_weights(
    scores:     pd.Series,
    vol:        pd.Series,
    regime:     int,
    n_long:     int   = 5,
    n_short:    int   = 5,     # kept in signature for API compat, not used
    max_pos:    float = 0.20,
    bear_scale: float = 0.35,  # kept for API compat; use REGIME_EXPOSURE instead
) -> pd.Series:
    """
    Build a long-only weight vector from cross-sectional alpha scores.

    Args:
        scores   : alpha score per stock (higher = more attractive long)
        vol      : realized vol per stock (used for inverse-vol sizing)
        regime   : HMM regime label (0=bear, 1=chop, 2=bull)
        n_long   : number of long positions to hold
        max_pos  : hard cap on any single position weight
    Returns:
        weights  : Series of position weights summing to <= gross_exposure.
                   Positive values only; no shorts.
    """
    gross_target = REGIME_EXPOSURE.get(regime, 0.70)

    ranked = scores.rank(ascending=False)
    longs  = ranked[ranked <= n_long].index

    if len(longs) == 0:
        logger.debug("No valid scores — holding cash")
        return pd.Series(0.0, index=scores.index)

    # Inverse-vol sizing within the long leg
    v      = vol.reindex(longs).fillna(vol.mean())
    inv_v  = 1.0 / (v + 1e-8)
    raw_w  = inv_v / inv_v.sum()         # sum to 1 within long leg
    weights = pd.Series(0.0, index=scores.index)
    weights[longs] = raw_w * gross_target

    # Hard cap per position, then rescale to preserve gross_target
    weights = weights.clip(upper=max_pos)
    if weights.sum() > 0:
        weights = weights / weights.sum() * gross_target

    regime_name = {0: "bear", 1: "chop", 2: "bull"}.get(regime, "?")
    logger.debug(
        f"Portfolio: longs={longs.tolist()} | regime={regime_name} "
        f"| gross={weights.sum():.2f} | cash={1 - weights.sum():.2f}"
    )
    return weights
