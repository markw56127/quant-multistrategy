"""
Fundamental data module: EPS, earnings surprises, analyst estimates, PEG.

All features are aligned to the day AFTER the earnings announcement date to
avoid lookahead bias (earnings are released after market close; price impact
is first observable on the next trading day). Features are then forward-filled
daily until the next announcement.

Features produced (per ticker, daily frequency):
  eps_actual         — most recent quarterly EPS (in dollars)
  eps_estimate       — analyst consensus EPS estimate at announcement time
  eps_surprise_pct   — (actual - estimate) / |estimate| * 100
  eps_growth_qoq     — quarter-over-quarter EPS growth
  eps_growth_yoy     — year-over-year EPS growth (same quarter)
  post_earn_ret_5d   — realized 5-day return starting from announcement+1
  analyst_fwd_eps    — forward (next-twelve-month) EPS estimate from analyst consensus
  peg_ratio          — P/E / (forward EPS growth rate); NaN if unavailable
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


class FundamentalsEngine:
    """Fetches and aligns fundamental data for a universe of tickers."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_fundamental_matrix(
        self,
        tickers: List[str],
        price_returns: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Build a (T, n_features * n_tickers) fundamental feature matrix aligned
        to the daily trading calendar in `price_returns`.

        Each ticker contributes ~8 features; all are forward-filled quarterly →
        daily so they slot into the existing feature matrix without frequency issues.
        """
        trading_days = price_returns.index
        all_frames: List[pd.DataFrame] = []

        for ticker in tickers:
            try:
                df = self._ticker_fundamentals(ticker, trading_days, price_returns)
                if df is not None and not df.empty:
                    all_frames.append(df)
                    logger.debug(f"  {ticker}: {df.shape[1]} fundamental features")
            except Exception as e:
                logger.warning(f"  {ticker} fundamentals failed: {e}")

        if not all_frames:
            logger.warning("No fundamental data retrieved — returning empty DataFrame")
            return pd.DataFrame(index=trading_days)

        result = pd.concat(all_frames, axis=1)
        result = result.reindex(trading_days).ffill().fillna(0.0)
        logger.info(f"Fundamentals matrix: {result.shape[1]} features × {len(result)} days")
        return result

    # ------------------------------------------------------------------
    # Per-ticker fundamentals
    # ------------------------------------------------------------------

    def _ticker_fundamentals(
        self,
        ticker: str,
        trading_days: pd.DatetimeIndex,
        price_returns: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        t = yf.Ticker(ticker)

        earnings_df = self._earnings_features(t, ticker, trading_days, price_returns)
        # LOOKAHEAD FIX (2026-06): _static_ratios broadcast TODAY's forwardEps /
        # pegRatio (a current yfinance snapshot) across the entire historical
        # window — e.g. 2026 analyst expectations visible to the model in 2015.
        # yfinance has no point-in-time history for these, so they are dropped.
        static_df = None

        frames = [f for f in [earnings_df, static_df] if f is not None]
        if not frames:
            return None
        return pd.concat(frames, axis=1).reindex(trading_days)

    def _earnings_features(
        self,
        t: yf.Ticker,
        ticker: str,
        trading_days: pd.DatetimeIndex,
        price_returns: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """
        Build time-series of EPS surprise, growth rates, and post-earnings drift.
        All values are placed on announcement_date + 1 business day to avoid
        lookahead bias, then forward-filled.
        """
        try:
            # earnings_dates: DataFrame with columns ['EPS Estimate', 'Reported EPS']
            earn = t.earnings_dates
            if earn is None or earn.empty:
                return None

            earn = earn.copy()
            earn.index = pd.DatetimeIndex(earn.index).tz_localize(None)
            earn = earn.sort_index()

            # Clip to training window + some buffer
            earn = earn[earn.index >= pd.Timestamp("2010-01-01")]

            actual_col   = "Reported EPS"
            estimate_col = "EPS Estimate"
            if actual_col not in earn.columns or estimate_col not in earn.columns:
                return None

            earn = earn[[actual_col, estimate_col]].dropna(subset=[actual_col])
            if len(earn) < 4:
                return None

            earn.columns = ["eps_actual", "eps_estimate"]
            earn["eps_estimate"] = earn["eps_estimate"].fillna(earn["eps_actual"])

            # Surprise %
            earn["eps_surprise_pct"] = (
                (earn["eps_actual"] - earn["eps_estimate"])
                / (earn["eps_estimate"].abs().clip(lower=1e-4))
                * 100.0
            )

            # Quarter-over-quarter growth
            earn["eps_growth_qoq"] = earn["eps_actual"].pct_change(1).clip(-5, 5)

            # Year-over-year growth (4 quarters back)
            earn["eps_growth_yoy"] = earn["eps_actual"].pct_change(4).clip(-5, 5)

            # Post-earnings 5-day return. LOOKAHEAD FIX (2026-06): this is the
            # realized return over announcement+1 .. announcement+5, so it is
            # only KNOWN at announcement+5. It used to be placed at
            # announcement+1, leaking 5 days of future returns into the feature
            # matrix. We now lag it by `horizon` extra business days so the
            # feature represents "drift after the PREVIOUS announcement" by the
            # time it becomes visible.
            if ticker in price_returns.columns:
                pead = self._post_earnings_return(
                    earn.index, price_returns[ticker], horizon=5
                )
                earn["post_earn_ret_5d"] = pead
                self._pead_extra_lag = 5  # consumed below when shifting
            else:
                earn["post_earn_ret_5d"] = 0.0
                self._pead_extra_lag = 0

            # Shift to announcement + 1 business day (avoid lookahead)
            shifted_idx = self._shift_to_next_business_day(earn.index, trading_days)
            earn_shifted = earn.copy()
            earn_shifted.index = shifted_idx
            earn_shifted = earn_shifted[~earn_shifted.index.isna()]

            # Map onto daily trading calendar via forward-fill
            daily_df = earn_shifted.reindex(trading_days).ffill()

            # Apply the extra availability lag to the realized-drift column so
            # it never contains returns from the future relative to its row.
            extra = getattr(self, "_pead_extra_lag", 0)
            if extra > 0 and "post_earn_ret_5d" in daily_df.columns:
                daily_df["post_earn_ret_5d"] = daily_df["post_earn_ret_5d"].shift(extra)
            daily_df.columns = [f"{ticker}_fund_{c}" for c in daily_df.columns]
            return daily_df

        except Exception as e:
            logger.debug(f"  {ticker} earnings features error: {e}")
            return None

    def _static_ratios(
        self,
        t: yf.Ticker,
        ticker: str,
        trading_days: pd.DatetimeIndex,
    ) -> Optional[pd.DataFrame]:
        """
        Forward EPS estimate and PEG ratio from yfinance .info.
        These are point-in-time snapshots (current values), so they're used
        as a slowly-varying feature. They will only be current as of last fetch.
        """
        try:
            info = t.info
            if not info:
                return None

            fwd_eps = info.get("forwardEps", np.nan)
            peg     = info.get("pegRatio",   np.nan)

            if np.isnan(fwd_eps) and np.isnan(peg):
                return None

            # Broadcast as constants across all trading days
            df = pd.DataFrame(
                {
                    f"{ticker}_fund_analyst_fwd_eps": fwd_eps,
                    f"{ticker}_fund_peg_ratio":       np.clip(peg, -10, 10) if not np.isnan(peg) else 0.0,
                },
                index=trading_days,
            )
            return df

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _post_earnings_return(
        announcement_dates: pd.DatetimeIndex,
        price_ret: pd.Series,
        horizon: int = 5,
    ) -> pd.Series:
        """
        Realized cumulative return over `horizon` days starting from the
        first trading day after each announcement.
        """
        results = []
        ret_index = price_ret.index
        for ann_date in announcement_dates:
            # Find position of announcement date or the next trading day
            pos = ret_index.searchsorted(ann_date)
            start = min(pos + 1, len(ret_index) - 1)
            end   = min(start + horizon, len(ret_index))
            if start >= len(ret_index):
                results.append(np.nan)
            else:
                cum_ret = (1 + price_ret.iloc[start:end]).prod() - 1
                results.append(float(cum_ret))
        return pd.Series(results, index=announcement_dates)

    @staticmethod
    def _shift_to_next_business_day(
        dates: pd.DatetimeIndex,
        trading_days: pd.DatetimeIndex,
    ) -> pd.DatetimeIndex:
        """
        For each date, find the next date in `trading_days` that is strictly
        after the announcement (handles after-hours releases and weekends).
        """
        shifted = []
        for d in dates:
            candidates = trading_days[trading_days > d]
            if len(candidates) == 0:
                shifted.append(pd.NaT)
            else:
                shifted.append(candidates[0])
        return pd.DatetimeIndex(shifted)