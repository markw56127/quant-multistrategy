from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger


class TechnicalIndicators:
    """
    Computes a comprehensive set of technical indicators for each ticker.
    Indicators are grouped into families for sensitivity analysis (zeroing out groups).
    """

    # Indicator families for sensitivity analysis
    FAMILIES = {
        "trend":     ["ema_12", "ema_26", "macd", "macd_signal", "macd_hist", "adx", "cci"],
        "momentum":  ["rsi_14", "rsi_28", "stoch_k", "stoch_d", "roc_5", "roc_21", "mom_14"],
        "volatility":["atr_14", "bb_upper", "bb_lower", "bb_width", "kc_upper", "kc_lower", "natr"],
        "volume":    ["obv", "vwap", "mfi_14", "cmf_20", "adosc"],
        "structure": ["support", "resistance", "pivot", "donch_upper", "donch_lower"],
    }

    def __init__(self, zero_families: Optional[List[str]] = None):
        # zero_families: list of family names to zero out (sensitivity analysis)
        self.zero_families = set(zero_families or [])

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Given a single ticker OHLCV DataFrame, returns a DataFrame of indicator values
        aligned to the same date index.
        """
        df = ohlcv.copy()
        # Normalize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            # Try "adj_close" as close
            if "adj_close" in df.columns and "close" not in df.columns:
                df = df.rename(columns={"adj_close": "close"})
        if "volume" not in df.columns:
            df["volume"] = 0.0
        # Price columns: ensure numpy float64 so pandas-ta 0.3.14b receives
        # plain numpy arrays (not pandas nullable types).
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
        # Volume: auto_adjust=True divides by a price ratio, producing fractional
        # floats (e.g. 3821394879.9999998).  pandas-ta casts volume to int64
        # internally; pandas 3.x now raises ValueError if the float isn't an exact
        # integer, so round first, then store as float64 for the arithmetic.
        df["volume"] = df["volume"].round().astype(np.float64)

        features = pd.DataFrame(index=df.index)

        # --- Trend ---
        features["ema_12"]      = ta.ema(df["close"], length=12)
        features["ema_26"]      = ta.ema(df["close"], length=26)
        macd_df                  = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            features["macd"]         = macd_df.iloc[:, 0]
            features["macd_signal"]  = macd_df.iloc[:, 2]
            features["macd_hist"]    = macd_df.iloc[:, 1]
        adx_df                   = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None and not adx_df.empty:
            features["adx"]          = adx_df.iloc[:, 0]
        features["cci"]         = ta.cci(df["high"], df["low"], df["close"], length=20)

        # Normalize trend features relative to price
        for col in ["ema_12", "ema_26"]:
            if col in features.columns:
                features[col] = (features[col] - df["close"]) / df["close"]

        # --- Momentum ---
        features["rsi_14"]     = ta.rsi(df["close"], length=14) / 100.0
        features["rsi_28"]     = ta.rsi(df["close"], length=28) / 100.0
        stoch_df               = ta.stoch(df["high"], df["low"], df["close"])
        if stoch_df is not None and not stoch_df.empty:
            features["stoch_k"]    = stoch_df.iloc[:, 0] / 100.0
            features["stoch_d"]    = stoch_df.iloc[:, 1] / 100.0
        features["roc_5"]      = ta.roc(df["close"], length=5) / 100.0
        features["roc_21"]     = ta.roc(df["close"], length=21) / 100.0
        features["mom_14"]     = ta.mom(df["close"], length=14) / df["close"]

        # --- Volatility ---
        features["atr_14"]     = ta.atr(df["high"], df["low"], df["close"], length=14) / df["close"]
        bb_df                  = ta.bbands(df["close"], length=20)
        if bb_df is not None and not bb_df.empty:
            features["bb_upper"]   = (bb_df.iloc[:, 2] - df["close"]) / df["close"]
            features["bb_lower"]   = (bb_df.iloc[:, 0] - df["close"]) / df["close"]
            features["bb_width"]   = (bb_df.iloc[:, 2] - bb_df.iloc[:, 0]) / df["close"]
        kc_df                  = ta.kc(df["high"], df["low"], df["close"])
        if kc_df is not None and not kc_df.empty:
            features["kc_upper"]   = (kc_df.iloc[:, 2] - df["close"]) / df["close"]
            features["kc_lower"]   = (kc_df.iloc[:, 0] - df["close"]) / df["close"]
        features["natr"]       = ta.natr(df["high"], df["low"], df["close"], length=14) / 100.0

        # --- Volume ---
        if df["volume"].sum() > 0:
            features["obv"]        = ta.obv(df["close"], df["volume"])
            features["obv"]        = features["obv"].pct_change().fillna(0)
            vwap_val               = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            if vwap_val is not None:
                features["vwap"]       = (vwap_val - df["close"]) / df["close"]
            # ta.mfi is broken on pandas 3.x: it initialises internal columns
            # with integer 0, producing int64 dtype, then assigns float money-flow
            # values which pandas 3.x rejects.  Inline equivalent using float 0.0.
            tp  = (df["high"] + df["low"] + df["close"]) / 3.0
            rmf = tp * df["volume"]
            pos_mf = rmf.where(tp.diff(1) > 0, 0.0)
            neg_mf = rmf.where(tp.diff(1) < 0, 0.0)
            psum   = pos_mf.rolling(14).sum()
            nsum   = neg_mf.rolling(14).sum()
            denom  = psum + nsum
            features["mfi_14"] = (100.0 * psum / denom.replace(0, np.nan)).fillna(50.0) / 100.0
            cmf_val                = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
            if cmf_val is not None:
                features["cmf_20"]     = cmf_val
            adosc_val              = ta.adosc(df["high"], df["low"], df["close"], df["volume"])
            if adosc_val is not None:
                features["adosc"]      = adosc_val.pct_change().fillna(0)

        # --- Structure ---
        pivot_pts              = self._pivot_points(df)
        features["pivot"]      = (pivot_pts["pivot"] - df["close"]) / df["close"]
        features["support"]    = (pivot_pts["s1"] - df["close"]) / df["close"]
        features["resistance"] = (pivot_pts["r1"] - df["close"]) / df["close"]
        donch_df               = ta.donchian(df["high"], df["low"], lower_length=20, upper_length=20)
        if donch_df is not None and not donch_df.empty:
            features["donch_upper"] = (donch_df.iloc[:, 2] - df["close"]) / df["close"]
            features["donch_lower"] = (donch_df.iloc[:, 0] - df["close"]) / df["close"]

        # Apply sensitivity zeroing
        for family in self.zero_families:
            cols = self.FAMILIES.get(family, [])
            for col in cols:
                if col in features.columns:
                    features[col] = 0.0

        return features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    def compute_universe(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """Compute indicators for every ticker in the universe."""
        result = {}
        for ticker, df in ohlcv_data.items():
            try:
                result[ticker] = self.compute(df)
            except Exception as e:
                logger.warning(f"Indicator computation failed for {ticker}: {e}")
        logger.info(f"Computed indicators for {len(result)} tickers")
        return result

    def build_feature_matrix(
        self,
        indicators: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Stacks all ticker indicators into a single wide DataFrame per date.
        Columns are <ticker>_<indicator_name>.
        """
        frames = []
        for ticker, ind_df in indicators.items():
            aligned = ind_df.reindex(returns.index).ffill().fillna(0.0)
            aligned.columns = [f"{ticker}_{c}" for c in aligned.columns]
            frames.append(aligned)
        if not frames:
            return pd.DataFrame(index=returns.index)
        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pivot_points(df: pd.DataFrame) -> pd.DataFrame:
        """Classic daily pivot points using prior-day H/L/C."""
        prev_high  = df["high"].shift(1)
        prev_low   = df["low"].shift(1)
        prev_close = df["close"].shift(1)
        pivot      = (prev_high + prev_low + prev_close) / 3.0
        return pd.DataFrame({
            "pivot": pivot,
            "r1": 2 * pivot - prev_low,
            "r2": pivot + (prev_high - prev_low),
            "s1": 2 * pivot - prev_high,
            "s2": pivot - (prev_high - prev_low),
        }, index=df.index)
