from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler


class FactorDecomposition:
    """
    Reduces the high-dimensional ticker return space into primary Eigen-factors
    using a combination of PCA, Factor Analysis, and CCA.

    Architecture:
      1. Per-sector PCA: compress each sector's tickers → sector eigen-factors
      2. Factor Analysis: extract rotation-invariant latent factors from eigen-factors
      3. Global CCA: discover shared structure between sectors and macro (ETF) returns
    """

    def __init__(
        self,
        n_pca_components: int = 12,
        n_global_factors: int = 8,
        n_fa_components: Optional[int] = None,
    ):
        self.n_pca_components = n_pca_components
        self.n_global_factors = n_global_factors
        self.n_fa_components = n_fa_components or n_pca_components

        self._sector_pca: Dict[str, PCA] = {}
        self._sector_scalers: Dict[str, StandardScaler] = {}
        self._fa: Optional[FactorAnalysis] = None
        self._fa_scaler: Optional[StandardScaler] = None
        self._global_pca: Optional[PCA] = None
        self._global_scaler: Optional[StandardScaler] = None
        self._cca: Optional[CCA] = None

        self.sector_eigen_factors_: Optional[pd.DataFrame] = None
        self.global_factors_: Optional[pd.DataFrame] = None
        self.explained_variance_: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        sector_returns: Dict[str, pd.DataFrame],
        macro_returns: Optional[pd.DataFrame] = None,
    ) -> "FactorDecomposition":
        """
        Fit PCA per sector, then Factor Analysis on the stacked eigen-factors,
        then optional CCA against macro ETF returns.
        """
        # Step 1: per-sector PCA
        sector_eigen_frames = []
        common_idx = None
        for sector, ret_df in sector_returns.items():
            ret_df = ret_df.dropna(how="all").fillna(0.0)
            n_comp = min(self.n_pca_components, ret_df.shape[1], ret_df.shape[0] - 1)
            scaler = StandardScaler()
            X = scaler.fit_transform(ret_df.values)
            pca = PCA(n_components=n_comp, svd_solver="full")
            pca.fit(X)
            self._sector_pca[sector] = pca
            self._sector_scalers[sector] = scaler
            self.explained_variance_[sector] = pca.explained_variance_ratio_

            cum_var = pca.explained_variance_ratio_.cumsum()[-1]
            logger.info(
                f"  {sector}: {n_comp} components, cumulative explained variance = {cum_var:.1%}"
            )
            if cum_var < 0.80:
                logger.warning(
                    f"  {sector} PCA explains only {cum_var:.1%} variance with {n_comp} components — "
                    f"consider increasing n_pca_components."
                )

            proj = pca.transform(X)
            cols = [f"{sector}_pc{i+1}" for i in range(n_comp)]
            df_proj = pd.DataFrame(proj, index=ret_df.index, columns=cols)
            sector_eigen_frames.append(df_proj)
            common_idx = df_proj.index if common_idx is None else common_idx.intersection(df_proj.index)

        # Step 2: stack eigen-factors and fit global Factor Analysis
        stacked = pd.concat(
            [df.reindex(common_idx) for df in sector_eigen_frames], axis=1
        ).fillna(0.0)
        self.sector_eigen_factors_ = stacked

        fa_scaler = StandardScaler()
        X_stack = fa_scaler.fit_transform(stacked.values)
        self._fa_scaler = fa_scaler

        n_fa = min(self.n_fa_components, X_stack.shape[1], X_stack.shape[0] - 1)
        fa = FactorAnalysis(n_components=n_fa, random_state=42)
        fa.fit(X_stack)
        self._fa = fa

        # Step 3: global PCA on FA loadings to get final latent factors
        fa_proj = fa.transform(X_stack)
        g_scaler = StandardScaler()
        X_fa = g_scaler.fit_transform(fa_proj)
        self._global_scaler = g_scaler

        n_global = min(self.n_global_factors, X_fa.shape[1], X_fa.shape[0] - 1)
        g_pca = PCA(n_components=n_global, svd_solver="full")
        g_pca.fit(X_fa)
        self._global_pca = g_pca

        global_proj = g_pca.transform(X_fa)
        cols = [f"gf_{i+1}" for i in range(n_global)]
        self.global_factors_ = pd.DataFrame(global_proj, index=common_idx, columns=cols)
        global_cum_var = g_pca.explained_variance_ratio_.cumsum()[-1]
        logger.info(f"  Global PCA: {n_global} factors, cumulative explained variance = {global_cum_var:.1%}")

        # Step 4: optional CCA with macro ETF returns
        if macro_returns is not None and not macro_returns.empty:
            self._fit_cca(self.global_factors_, macro_returns)

        logger.info(
            f"FactorDecomposition fit: "
            f"{len(sector_returns)} sectors → "
            f"{stacked.shape[1]} eigen-factors → "
            f"{n_global} global factors"
        )
        return self

    def transform(
        self,
        sector_returns: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Project new sector returns into the fitted global factor space."""
        sector_eigen_frames = []
        common_idx = None
        for sector, ret_df in sector_returns.items():
            if sector not in self._sector_pca:
                continue
            ret_df = ret_df.fillna(0.0)
            X = self._sector_scalers[sector].transform(ret_df.values)
            proj = self._sector_pca[sector].transform(X)
            n_comp = proj.shape[1]
            cols = [f"{sector}_pc{i+1}" for i in range(n_comp)]
            df_proj = pd.DataFrame(proj, index=ret_df.index, columns=cols)
            sector_eigen_frames.append(df_proj)
            common_idx = df_proj.index if common_idx is None else common_idx.intersection(df_proj.index)

        stacked = pd.concat(
            [df.reindex(common_idx) for df in sector_eigen_frames], axis=1
        ).fillna(0.0)

        # Align columns with fitted scaler
        stacked = stacked.reindex(columns=self.sector_eigen_factors_.columns, fill_value=0.0)
        X_stack = self._fa_scaler.transform(stacked.values)
        fa_proj = self._fa.transform(X_stack)
        X_fa = self._global_scaler.transform(fa_proj)
        global_proj = self._global_pca.transform(X_fa)
        cols = [f"gf_{i+1}" for i in range(global_proj.shape[1])]
        return pd.DataFrame(global_proj, index=common_idx, columns=cols)

    def fit_transform(
        self,
        sector_returns: Dict[str, pd.DataFrame],
        macro_returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        self.fit(sector_returns, macro_returns)
        return self.global_factors_

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------

    def sector_loadings(self) -> pd.DataFrame:
        """Returns the per-sector explained variance ratio as a DataFrame."""
        rows = {}
        for sector, evr in self.explained_variance_.items():
            for i, v in enumerate(evr):
                rows[f"{sector}_pc{i+1}"] = v
        return pd.Series(rows).sort_values(ascending=False).to_frame("explained_var")

    def global_factor_loadings(self) -> pd.DataFrame:
        """Returns the global PCA component matrix (global_factors → sector eigen-factors)."""
        if self._global_pca is None:
            raise RuntimeError("Model not fitted yet.")
        # Composite loadings: global_pca.components_ @ fa.components_ @ stacked_features
        return pd.DataFrame(
            self._global_pca.components_,
            columns=[f"fa_{i+1}" for i in range(self._global_pca.components_.shape[1])],
            index=[f"gf_{i+1}" for i in range(self._global_pca.n_components_)],
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def save(self, path: str):
        joblib.dump({
            "sector_pca":           self._sector_pca,
            "sector_scalers":       self._sector_scalers,
            "fa":                   self._fa,
            "fa_scaler":            self._fa_scaler,
            "global_pca":           self._global_pca,
            "global_scaler":        self._global_scaler,
            "cca":                  self._cca,
            "sector_eigen_factors": self.sector_eigen_factors_,
            "global_factors":       self.global_factors_,
            "explained_variance":   self.explained_variance_,
        }, path)
        logger.info(f"FactorDecomposition saved to {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self._sector_pca          = data["sector_pca"]
        self._sector_scalers      = data["sector_scalers"]
        self._fa                  = data["fa"]
        self._fa_scaler           = data["fa_scaler"]
        self._global_pca          = data["global_pca"]
        self._global_scaler       = data["global_scaler"]
        self._cca                 = data["cca"]
        self.sector_eigen_factors_ = data["sector_eigen_factors"]
        self.global_factors_       = data["global_factors"]
        self.explained_variance_   = data["explained_variance"]
        logger.info(f"FactorDecomposition loaded from {path}")

    def _fit_cca(self, factors: pd.DataFrame, macro: pd.DataFrame):
        common = factors.index.intersection(macro.index)
        X = factors.loc[common].values
        Y = macro.loc[common].fillna(0.0).values
        n_comp = min(self.n_global_factors, X.shape[1], Y.shape[1])
        cca = CCA(n_components=n_comp)
        try:
            cca.fit(X, Y)
            self._cca = cca
            logger.info(f"CCA fitted with {n_comp} components")
        except Exception as e:
            logger.warning(f"CCA fitting failed: {e}")
