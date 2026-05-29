"""
Sector regime detection via a Gaussian Hidden Markov Model.

Why HMM over a threshold rule or rolling z-score:
  The semiconductor cycle has genuinely latent structure — the same sector
  return level means different things depending on whether you're in an
  inventory build or a demand collapse. HMM infers this hidden state from
  the joint distribution of returns AND realized volatility, so it can
  distinguish "high-vol-up" (recovery) from "low-vol-up" (steady expansion).

Three states after re-sorting by mean return:
  0 = bear    (negative returns, high vol)
  1 = chop    (near-zero returns, moderate vol)
  2 = bull    (positive returns, low-moderate vol)

The state probabilities (not just the argmax) are passed into the
cross-sectional model as features, letting it express partial-regime beliefs
(e.g., 60% bull / 40% chop) rather than hard-switching.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from loguru import logger
from sklearn.preprocessing import StandardScaler

N_STATES = 3


class SectorRegimeModel:
    def __init__(self, n_states: int = N_STATES, n_iter: int = 300, random_state: int = 42):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self._state_order: np.ndarray = np.arange(n_states)  # bear→chop→bull sorted indices

    def _build_features(self, sector_returns: pd.Series, vol_window: int = 21) -> np.ndarray:
        vol = sector_returns.rolling(vol_window).std().bfill()
        return np.column_stack([sector_returns.values, vol.values])

    def fit(self, sector_returns: pd.Series, vol_window: int = 21) -> "SectorRegimeModel":
        X_raw = self._build_features(sector_returns, vol_window)
        X = self.scaler.fit_transform(X_raw)
        self.model.fit(X)

        # Re-order states by mean return so state 0 = bear, state 2 = bull.
        # HMM initialization is random, so state numbering is arbitrary without this.
        means = self.model.means_[:, 0]  # first feature is sector return
        self._state_order = np.argsort(means)  # ascending: bear=0, bull=2

        logger.info(
            f"HMM fitted | state mean returns (sorted): "
            f"{means[self._state_order].round(4).tolist()}"
        )
        return self

    def predict_proba(
        self, sector_returns: pd.Series, vol_window: int = 21
    ) -> pd.DataFrame:
        """
        Returns (T, n_states) posterior state probabilities.
        Columns are sorted bear→bull (0=bear, n-1=bull).
        """
        X = self.scaler.transform(self._build_features(sector_returns, vol_window))
        raw_proba = self.model.predict_proba(X)  # (T, n_states), unsorted

        reordered = np.zeros_like(raw_proba)
        for new_idx, old_idx in enumerate(self._state_order):
            reordered[:, new_idx] = raw_proba[:, old_idx]

        return pd.DataFrame(
            reordered,
            index=sector_returns.index,
            columns=[f"regime_{i}" for i in range(self.n_states)],
        )

    def predict_state(
        self, sector_returns: pd.Series, vol_window: int = 21
    ) -> pd.Series:
        """Most probable regime state (integer 0=bear, 1=chop, 2=bull)."""
        proba = self.predict_proba(sector_returns, vol_window)
        return proba.idxmax(axis=1).str.extract(r"(\d+)")[0].astype(int).rename("regime")
