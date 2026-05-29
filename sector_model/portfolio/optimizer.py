"""
Portfolio construction: translate alpha scores into long-only position weights.

Exposure management — volatility targeting (Moreira & Muir 2017, JF):
  Scale gross exposure inversely with realised sector volatility to maintain
  a roughly constant risk budget.

      exposure = clip( target_vol / sector_vol_ann, min_exposure, max_exposure )

  Intuition: when the market is calm (low vol), deploy fully — that's when
  carrying risk is cheapest.  When vol spikes (crashes, dislocations), reduce
  exposure automatically — that's when unexpected moves are largest.

  This is principled rather than regime-fitted: you're not predicting bear vs
  bull, you're responding to a directly observable risk signal (realised vol)
  that is known to be short-term persistent (vol clustering).

  Reference: Moreira & Muir (2017) "Volatility-Managed Portfolios", JF 72(4).
"""

import numpy as np
import pandas as pd
from loguru import logger


def construct_weights(
    scores:          pd.Series,
    stock_vol:       pd.Series,
    sector_vol_ann:  float,
    n_long:          int   = 8,
    max_pos:         float = 0.20,
    target_vol:      float = 0.15,
    min_exposure:    float = 0.50,
    max_exposure:    float = 1.00,
) -> pd.Series:
    """
    Build a long-only weight vector from cross-sectional alpha scores.

    Args:
        scores          : alpha score per stock (higher = more attractive long)
        stock_vol       : per-stock 21-day realised vol (for inverse-vol sizing)
        sector_vol_ann  : sector (SMH) annualised realised vol — drives exposure
        n_long          : number of long positions to hold
        max_pos         : hard cap on any single position weight
        target_vol      : target annualised portfolio volatility
        min_exposure    : floor gross exposure (always at least this deployed)
        max_exposure    : ceiling gross exposure
    Returns:
        weights         : Series of non-negative weights. Sums to gross_target.
    """
    # Vol-targeted gross exposure
    if sector_vol_ann > 1e-6:
        gross_target = np.clip(target_vol / sector_vol_ann, min_exposure, max_exposure)
    else:
        gross_target = max_exposure

    ranked = scores.rank(ascending=False)
    longs  = ranked[ranked <= n_long].index

    if len(longs) == 0:
        logger.debug("No valid scores — holding cash")
        return pd.Series(0.0, index=scores.index)

    v     = stock_vol.reindex(longs).fillna(stock_vol.mean())
    inv_v = 1.0 / (v + 1e-8)
    raw_w = inv_v / inv_v.sum()

    weights = pd.Series(0.0, index=scores.index)
    weights[longs] = raw_w * gross_target

    weights = weights.clip(upper=max_pos)
    if weights.sum() > 0:
        weights = weights / weights.sum() * gross_target

    logger.debug(
        f"Portfolio: longs={longs.tolist()} | sector_vol={sector_vol_ann:.1%} "
        f"| gross={weights.sum():.2f} | cash={1 - weights.sum():.2f}"
    )
    return weights
