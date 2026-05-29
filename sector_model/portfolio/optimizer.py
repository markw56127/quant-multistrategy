"""
Portfolio construction: translate alpha scores into long-only position weights.

Exposure management — downside semivariance targeting:
  Scale gross exposure inversely with realised DOWNSIDE sector volatility.

      downside_vol = std(clip(daily_returns, upper=0)) × √252 × √2
      exposure     = clip( target_vol / downside_vol, min_exposure, max_exposure )

  Key difference from total-vol targeting: positive volatility (fast recoveries,
  bull surges) does not reduce exposure. Only days where the sector actually falls
  contribute to downside_vol, so the model stays deployed during V-shaped rallies.

  The √2 correction ensures that for a symmetric return distribution, downside_vol
  equals total_vol — keeping target_vol calibration consistent.

  Reference: Ang, Chen & Xing (2006) "Downside Risk", RFS 19(4).
             Moreira & Muir (2017) "Volatility-Managed Portfolios", JF 72(4).

Position sizing — score × inverse-vol (Kelly-proportional):
  Within the selected longs, weight each position by:
      w[i] ∝ score_rank[i] × (1 / stock_vol[i])

  The highest-alpha, lowest-vol positions receive the most capital. This is
  a Kelly-inspired heuristic: bet more where you have more edge and lower
  uncertainty, less where the signal is weaker.
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
        stock_vol       : per-stock 21-day realised vol (for sizing)
        sector_vol_ann  : sector downside vol (annualised) — drives exposure
        n_long          : number of long positions to hold
        max_pos         : hard cap on any single position weight
        target_vol      : target annualised portfolio volatility
        min_exposure    : floor gross exposure
        max_exposure    : ceiling gross exposure
    Returns:
        weights         : Series of non-negative weights summing to gross_target.
    """
    gross_target = np.clip(target_vol / sector_vol_ann, min_exposure, max_exposure) \
                   if sector_vol_ann > 1e-6 else max_exposure

    ranked = scores.rank(ascending=False)
    longs  = ranked[ranked <= n_long].index

    if len(longs) == 0:
        logger.debug("No valid scores — holding cash")
        return pd.Series(0.0, index=scores.index)

    v     = stock_vol.reindex(longs).fillna(stock_vol.mean())
    inv_v = 1.0 / (v + 1e-8)

    # Score-weighted sizing: rank within selected longs × inverse-vol
    # Higher-conviction AND lower-vol positions receive more capital
    score_rank = scores.reindex(longs).rank()   # 1=weakest long, n=strongest
    sizing     = score_rank * inv_v
    raw_w      = sizing / sizing.sum()

    weights = pd.Series(0.0, index=scores.index)
    weights[longs] = raw_w * gross_target

    weights = weights.clip(upper=max_pos)
    if weights.sum() > 0:
        weights = weights / weights.sum() * gross_target

    logger.debug(
        f"Portfolio: longs={longs.tolist()} | downvol={sector_vol_ann:.1%} "
        f"| gross={weights.sum():.2f} | cash={1 - weights.sum():.2f}"
    )
    return weights
