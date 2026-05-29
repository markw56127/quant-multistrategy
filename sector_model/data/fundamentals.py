"""
Fundamental feature builder.

Sources from yfinance:
  - earnings_dates : quarterly reported EPS, analyst EPS estimate, surprise %
                     goes back ~5-6 years (2020-present)

Features computed per ticker, aligned to trading dates with no look-ahead:
  eps_surprise      -- % surprise vs analyst estimate at most recent earnings
  eps_ttm           -- trailing 12-month EPS (sum of last 4 reported quarters)
  eps_growth_yoy    -- TTM EPS growth vs same TTM one year prior
  eps_acceleration  -- change in YoY growth rate vs the prior year (is growth speeding up?)
  eps_revision      -- analyst EPS estimate change vs the prior quarter's estimate
  trailing_pe       -- daily price / eps_ttm  (clamped to [2, 150], NaN when eps_ttm <= 0)
  peg_ratio         -- trailing_pe / (eps_growth_yoy * 100), NaN when growth <= 0

All features are lagged by one trading day past the earnings announcement date so
that no rebalance date can see an announcement made on the same day.

Coverage note: yfinance earnings_dates starts ~2020 for most tickers.  Rows
before that window will be NaN and LightGBM will route those samples through
non-fundamental splits automatically.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


def _fetch_ticker_earnings(ticker: str) -> pd.DataFrame:
    """
    Pull earnings_dates for one ticker.
    Returns DataFrame with tz-naive date index and columns:
      reported_eps, eps_estimate, eps_surprise_pct
    """
    try:
        raw = yf.Ticker(ticker).earnings_dates
    except Exception as e:
        logger.warning(f"{ticker}: earnings_dates failed ({e})")
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw.sort_index()

    # Drop future rows (no reported EPS yet)
    raw = raw[raw["Reported EPS"].notna()].copy()

    raw = raw.rename(columns={
        "Reported EPS":  "reported_eps",
        "EPS Estimate":  "eps_estimate",
        "Surprise(%)":   "eps_surprise_pct",
    })
    return raw[["reported_eps", "eps_estimate", "eps_surprise_pct"]]


def _build_ticker_features(
    ticker: str,
    earnings: pd.DataFrame,
    daily_prices: pd.Series,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute fundamental features for one ticker on all trading_dates.
    """
    out = pd.DataFrame(index=trading_dates, dtype=np.float64)
    out.index.name = "date"

    if earnings.empty:
        for col in ["eps_surprise", "eps_ttm", "eps_growth_yoy",
                    "eps_acceleration", "eps_revision", "trailing_pe", "peg_ratio"]:
            out[col] = np.nan
        return out

    # Deduplicate earnings by date (keep first if multiple on same date)
    earnings = earnings[~earnings.index.duplicated(keep="first")]

    # ── EPS series on announcement dates ──────────────────────────────────
    # Shift 1 calendar day so same-day announcements aren't visible yet.
    # (Earnings after market close on date D → available from D+1 onwards.)
    eps_ann = earnings["reported_eps"].copy()
    eps_ann.index = eps_ann.index + pd.Timedelta(days=1)

    est_ann = earnings["eps_estimate"].copy()
    est_ann.index = est_ann.index + pd.Timedelta(days=1)

    surp_ann = earnings["eps_surprise_pct"].copy()
    surp_ann.index = surp_ann.index + pd.Timedelta(days=1)

    # ── Reindex to trading dates and forward-fill ──────────────────────────
    eps_daily  = eps_ann.reindex(trading_dates).ffill()
    est_daily  = est_ann.reindex(trading_dates).ffill()
    surp_daily = surp_ann.reindex(trading_dates).ffill()

    out["eps_surprise"] = surp_daily

    # ── TTM EPS: rolling sum of last 4 quarterly reported values ──────────
    # Build a quarterly series on announcement dates (shifted), then ffill.
    # Summing 4 most recent quarters gives trailing 12-month EPS.
    eps_q = eps_ann.reindex(trading_dates).ffill()

    # Count how many unique earnings dates have been published up to each date
    # by checking the shifted announcement dates
    shifted_dates = sorted(eps_ann.index)
    ttm_by_date: dict = {}
    for d in trading_dates:
        past = [dt for dt in shifted_dates if dt <= d]
        if len(past) >= 4:
            ttm_by_date[d] = float(eps_ann.reindex(past).iloc[-4:].sum())
        elif len(past) >= 1:
            ttm_by_date[d] = float(eps_ann.reindex(past).sum())  # partial TTM
        # else: NaN

    eps_ttm = pd.Series(ttm_by_date, name="eps_ttm").reindex(trading_dates)
    out["eps_ttm"] = eps_ttm

    # ── YoY EPS growth ─────────────────────────────────────────────────────
    # Compare TTM now vs TTM 252 trading days ago (~1 year)
    ttm_1yr = eps_ttm.shift(252).astype(float)  # convert object dtype to float, None → NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = (eps_ttm - ttm_1yr) / ttm_1yr.abs().replace(0, np.nan)
    out["eps_growth_yoy"] = growth.clip(-2, 10)  # cap extreme values

    # ── EPS acceleration ───────────────────────────────────────────────────
    # Change in YoY growth vs the same growth one year ago
    out["eps_acceleration"] = (out["eps_growth_yoy"] - out["eps_growth_yoy"].shift(252)).clip(-5, 5)

    # ── Analyst estimate revision ──────────────────────────────────────────
    # How much did the analyst EPS estimate change vs the previous quarter's estimate?
    prev_est = est_daily.shift(63).astype(float)   # convert object dtype to float, None → NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        revision = (est_daily - prev_est) / prev_est.abs().replace(0, np.nan)
    out["eps_revision"] = revision.clip(-1, 1)

    # ── Estimate revision trend (direction consistency) ────────────────────
    # Sign of revision at each quarterly announcement date (+1 up, -1 down).
    # Rolling 4-quarter sum → ranges from -4 (all down) to +4 (all up).
    # Backed by earnings estimate revision literature (Hawkins et al. 1984,
    # Stickel 1991): stocks with persistent upward revisions outperform.
    rev_sign = pd.Series(
        np.sign(est_ann.values - est_ann.shift(1).values),
        index=est_ann.index,
    )
    rev_trend_q = rev_sign.rolling(4, min_periods=2).sum()
    out["eps_revision_trend"] = rev_trend_q.reindex(trading_dates).ffill()

    # ── Trailing P/E ───────────────────────────────────────────────────────
    price = daily_prices.reindex(trading_dates).ffill()
    valid_ttm = eps_ttm.where(eps_ttm > 0)          # P/E undefined for negative EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = price / valid_ttm
    out["trailing_pe"] = pe.clip(2, 150)

    # ── PEG ratio ──────────────────────────────────────────────────────────
    pos_growth = out["eps_growth_yoy"].where(out["eps_growth_yoy"] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        peg = out["trailing_pe"] / (pos_growth * 100)
    out["peg_ratio"] = peg.clip(0, 10)

    return out


def fetch_fundamental_features(
    tickers: List[str],
    prices: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a (date × ticker) stacked fundamental feature DataFrame.

    Args:
        tickers       : list of stock tickers
        prices        : daily adjusted close DataFrame (index=dates, columns=tickers)
        trading_dates : the index from stock_returns — defines output grid
        cache_dir     : if provided, cache/load from parquet

    Returns:
        DataFrame with MultiIndex (date, ticker) and fundamental feature columns.
        NaN where earnings history is unavailable (pre-2020 for most tickers).
    """
    if cache_dir:
        cache_path = Path(cache_dir) / "fundamentals.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached fundamentals from {cache_path}")
            return pd.read_parquet(cache_path)

    logger.info(f"Fetching fundamental data for {len(tickers)} tickers")

    records = []
    for ticker in tickers:
        earnings = _fetch_ticker_earnings(ticker)
        n_quarters = len(earnings)
        # logger.debug(f"  {ticker}: {n_quarters} earnings quarters available "
        #              f"({'N/A' if earnings.empty else str(earnings.index.min().date()) + ' → ' + str(earnings.index.max().date())})")

        price_series = prices[ticker] if ticker in prices.columns else pd.Series(dtype=float)
        feats = _build_ticker_features(ticker, earnings, price_series, trading_dates)
        feats["ticker"] = ticker
        records.append(feats)

    panel = pd.concat(records)
    panel.index = pd.MultiIndex.from_arrays(
        [panel.index, panel["ticker"]], names=["date", "ticker"]
    )
    panel = panel.drop(columns=["ticker"]).sort_index()

    coverage = panel["eps_ttm"].notna().mean()
    logger.info(
        f"Fundamental panel built | shape={panel.shape} | "
        f"coverage={coverage:.1%} (NaN outside ~2020+ window is expected)"
    )

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache_path)
        logger.info(f"Cached fundamentals to {cache_path}")

    return panel
