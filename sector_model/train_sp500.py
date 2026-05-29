"""
Two-layer S&P 500 portfolio.

Layer 1 — Sector Rotation (signals/sector_rotation.py):
  Monthly sector weights from 12-1 momentum + macro tilt (yield curve, VIX).
  Applied across 8 active GICS sectors.

Layer 2 — Within-Sector Stock Picker (existing pipeline):
  For each sector, a LightGBM model trained walk-forward on the sector's
  stocks selects the top-N positions sized by score × inverse-vol.

Final weight:
  w[stock] = sector_weight[sector(stock)] × within_sector_weight[stock]

Run from sector_model/ directory:
  python train_sp500.py
  python train_sp500.py --out results/sp500/backtest.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from data.universe import load_sector_data, fetch_volumes
from data.fundamentals import fetch_fundamental_features
from data.sp500 import (
    fetch_sp500_universe, get_sector_tickers,
    build_sector_cfg, SECTOR_ETFS, ACTIVE_SECTORS,
)
from signals.residuals import rolling_ols_decompose, forward_cross_sectional_excess
from signals.sector import SectorRegimeModel
from signals.cross_section import CrossSectionalModel, build_features
from signals.sector_rotation import SectorRotationModel, fetch_macro_data
from portfolio.optimizer import construct_weights
from backtest.engine import SectorBacktest


def load_base_config(path: str = "config/sectors/financials.yaml") -> dict:
    """Load a base config — only the modelling params are used; data section is overridden."""
    with open(path) as f:
        return yaml.safe_load(f)


def _run_sector_pipeline(sector_name: str, sector_cfg: dict):
    """
    Run the full within-sector pipeline for one GICS sector.

    Returns:
        stock_ret    : (T, N) daily log-returns
        sector_ret   : (T,)  sector ETF log-returns
        features     : (date, ticker) MultiIndex feature panel
        targets      : (T, N) forward cross-sectional excess returns
        alpha_model  : fitted (or unfitted) CrossSectionalModel
        regime_model : fitted SectorRegimeModel
        residuals    : (T, N) OLS residuals (for vol estimation)
    """
    cache_dir = sector_cfg["data"]["cache_dir"]
    sig_cfg   = sector_cfg["signals"]

    stock_ret, sector_ret, prices = load_sector_data(sector_cfg["data"], cache_dir)
    if stock_ret.shape[1] < 5:
        logger.warning(f"{sector_name}: only {stock_ret.shape[1]} stocks after filter — skipping")
        return None

    betas, _, residuals = rolling_ols_decompose(
        stock_ret, sector_ret,
        window=sig_cfg["ols_window"],
        min_periods=sig_cfg["ols_min_periods"],
    )
    targets = forward_cross_sectional_excess(stock_ret, horizon=sig_cfg["forward_horizon"])

    regime_model = SectorRegimeModel(
        n_states=sector_cfg["regime"]["n_states"],
        n_iter=sector_cfg["regime"]["n_iter"],
    )
    regime_model.fit(sector_ret, vol_window=sector_cfg["regime"]["vol_window"])
    regime_proba = regime_model.predict_proba(sector_ret, vol_window=sector_cfg["regime"]["vol_window"])

    tickers = sector_cfg["data"]["universe"]
    volumes = fetch_volumes(tickers, sector_cfg["data"]["start_date"],
                            sector_cfg["data"]["end_date"], cache_dir)
    fundamentals = fetch_fundamental_features(tickers, prices, stock_ret.index, cache_dir)

    features = build_features(
        stock_ret, residuals, betas, regime_proba, sector_ret,
        fundamentals=fundamentals, volumes=volumes,
    )
    alpha_model = CrossSectionalModel(sector_cfg["alpha_model"])

    logger.info(
        f"  {sector_name}: {stock_ret.shape[1]} stocks × {len(stock_ret)} days "
        f"| features {features.shape}"
    )
    return {
        "stock_ret":    stock_ret,
        "sector_ret":   sector_ret,
        "residuals":    residuals,
        "features":     features,
        "targets":      targets,
        "alpha_model":  alpha_model,
        "regime_model": regime_model,
    }


def run_sp500(base_cfg: dict, out_path: str = "results/sp500/backtest.csv") -> pd.DataFrame:
    start_date = base_cfg["data"]["start_date"]
    end_date   = base_cfg["data"]["end_date"]
    sp500_cache = "cache/sp500"

    # ── Universe ──────────────────────────────────────────────────────────
    logger.info("═══ Stage 1: Loading S&P 500 universe ═══")
    sp500 = fetch_sp500_universe(cache_dir=sp500_cache)

    # ── Macro data for sector rotation ────────────────────────────────────
    logger.info("═══ Stage 2: Fetching macro data (VIX, yields) ═══")
    macro = fetch_macro_data(start_date, end_date, cache_dir=sp500_cache)

    # ── Sector ETF returns for rotation model ─────────────────────────────
    logger.info("═══ Stage 3: Fetching sector ETF prices ═══")
    import yfinance as yf
    etf_tickers = [SECTOR_ETFS[s] for s in ACTIVE_SECTORS]
    raw = yf.download(etf_tickers, start=start_date, end=end_date,
                      auto_adjust=True, progress=False)
    etf_prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    etf_prices = etf_prices.ffill(limit=5).dropna()
    etf_ret    = np.log(etf_prices / etf_prices.shift(1)).dropna()
    # Rename columns from ETF tickers back to sector names
    etf_to_sector = {v: k for k, v in SECTOR_ETFS.items() if k in ACTIVE_SECTORS}
    etf_ret = etf_ret.rename(columns=etf_to_sector)

    # ── Per-sector within-sector models ───────────────────────────────────
    logger.info("═══ Stage 4: Building per-sector models ═══")
    pipelines: Dict[str, dict] = {}
    for sector in ACTIVE_SECTORS:
        etf     = SECTOR_ETFS[sector]
        tickers = get_sector_tickers(sp500, sector)
        if not tickers:
            continue
        cfg = build_sector_cfg(
            tickers=tickers,
            sector_etf=etf,
            base_cfg=base_cfg,
            cache_subdir=f"cache/sp500/{sector.lower().replace(' ', '_')}",
        )
        logger.info(f"  ── {sector} ({etf}) — {len(tickers)} candidates ──")
        result = _run_sector_pipeline(sector, cfg)
        if result:
            result["cfg"] = cfg
            pipelines[sector] = result

    if not pipelines:
        raise RuntimeError("No sectors loaded — check universe and config.")

    # ── Sector rotation model ─────────────────────────────────────────────
    logger.info("═══ Stage 5: Initialising sector rotation model ═══")
    active_loaded = [s for s in ACTIVE_SECTORS if s in pipelines]
    rotation_model = SectorRotationModel(
        sector_etf_returns=etf_ret[[s for s in active_loaded if s in etf_ret.columns]],
        macro=macro,
        sectors=active_loaded,
    )

    # ── Walk-forward two-layer backtest ───────────────────────────────────
    logger.info("═══ Stage 6: Walk-forward two-layer backtest ═══")
    bt_cfg     = base_cfg["backtest"]
    port_cfg   = base_cfg["portfolio"]
    rebal_freq = bt_cfg.get("rebalance_freq", 20)
    train_win  = bt_cfg.get("train_window", 504)
    retrain_n  = bt_cfg.get("retrain_every_n", 3)
    trans_cost = bt_cfg.get("transaction_cost", 0.001)
    init_cap   = bt_cfg.get("initial_capital", 1_000_000)
    target_vol   = port_cfg.get("target_vol", 0.15)
    min_exposure = port_cfg.get("min_exposure", 0.50)
    max_exposure = port_cfg.get("max_exposure", 1.00)
    max_pos      = port_cfg.get("max_position_size", 0.05)  # tighter: 500 stocks

    # Use the largest-overlap trading dates across sectors
    all_dates = sorted(set.intersection(*[
        set(p["stock_ret"].index) for p in pipelines.values()
    ]))
    dates = pd.DatetimeIndex(all_dates)

    # Sector-level downside vol for exposure scaling (use SPY as market vol proxy)
    spy_raw  = yf.download("SPY", start=start_date, end=end_date,
                            auto_adjust=True, progress=False)
    spy_ret  = np.log(spy_raw["Close"] / spy_raw["Close"].shift(1)).dropna()
    spy_dvol = spy_ret.clip(upper=0).rolling(21).std() * np.sqrt(252) * np.sqrt(2)

    capital         = float(init_cap)
    current_weights = {}          # stock → current weight
    last_fit_rebals = {s: -retrain_n for s in pipelines}
    records         = []

    rebal_dates = dates[train_win::rebal_freq]

    for rebal_num, rebal_date in enumerate(rebal_dates):
        date_idx = dates.get_loc(rebal_date)

        # ── Layer 1: sector weights ────────────────────────────────────────
        sector_weights = rotation_model.predict(rebal_date)
        SectorRotationModel.log_weights(sector_weights, rebal_date)

        # ── Layer 2: within-sector scores + refit ─────────────────────────
        new_weights: Dict[str, float] = {}
        spy_vol = float(spy_dvol.reindex([rebal_date]).iloc[0]) \
                  if rebal_date in spy_dvol.index else target_vol

        for sector, pipe in pipelines.items():
            sec_wt = float(sector_weights.get(sector, 0.0))
            if sec_wt < 0.01:
                continue

            p_dates = pipe["stock_ret"].index
            if rebal_date not in p_dates:
                continue
            d_idx = p_dates.get_loc(rebal_date)
            if d_idx < train_win:
                continue

            # Refit LightGBM quarterly
            last_fit = last_fit_rebals[sector]
            if rebal_num - last_fit >= retrain_n:
                tr_start = p_dates[max(0, d_idx - train_win)]
                tr_end   = p_dates[d_idx - 1]
                pipe["alpha_model"].fit(
                    pipe["features"], pipe["targets"], tr_start, tr_end
                )
                last_fit_rebals[sector] = rebal_num

            try:
                scores = pipe["alpha_model"].predict(pipe["features"], rebal_date)
            except Exception:
                continue

            vol_now  = pipe["residuals"].rolling(21).std().loc[rebal_date] \
                       .reindex(scores.index).fillna(0.02)
            sec_port_cfg = pipe["cfg"].get("portfolio", port_cfg)
            n_long   = sec_port_cfg.get("n_long", 8)

            within_wt = construct_weights(
                scores, vol_now, spy_vol,
                n_long=n_long, max_pos=1.0,  # no single-stock cap here
                target_vol=target_vol,
                min_exposure=min_exposure,
                max_exposure=max_exposure,
            )
            # Apply sector allocation: final stock weight = sector_wt × within_wt
            for stock, w in within_wt[within_wt > 0].items():
                new_weights[stock] = sec_wt * float(w)

        # Normalise so total gross = sector-weighted exposure
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total * sum(sector_weights.values())
                           for k, v in new_weights.items()}

        # ── Portfolio P&L ─────────────────────────────────────────────────
        # Turnover cost
        all_stocks = set(current_weights) | set(new_weights)
        turnover   = sum(abs(new_weights.get(s, 0) - current_weights.get(s, 0))
                        for s in all_stocks)
        capital   *= (1.0 - turnover * trans_cost)

        # Hold period
        end_idx = min(date_idx + rebal_freq, len(dates) - 1)
        period_ret = 0.0
        for stock, w in new_weights.items():
            for sec_pipe in pipelines.values():
                if stock in sec_pipe["stock_ret"].columns:
                    hold = sec_pipe["stock_ret"].iloc[date_idx:end_idx][stock]
                    period_ret += w * float(hold.sum())
                    break
        capital *= (1.0 + period_ret)

        # SPY benchmark return for the same window
        bench_ret = float(spy_ret.iloc[date_idx:end_idx].sum()) \
                    if date_idx < len(spy_ret) else 0.0

        records.append({
            "date":              rebal_date,
            "capital":           capital,
            "period_return":     period_ret,
            "benchmark_return":  bench_ret,
            "turnover":          turnover,
            "cost":              turnover * trans_cost,
            "n_stocks":          len(new_weights),
            "gross_exposure":    sum(new_weights.values()),
            "top_sector":        sector_weights.idxmax(),
        })
        current_weights = new_weights

    results = pd.DataFrame(records).set_index("date")
    total_ret = capital / init_cap - 1
    bench_total = (1 + results["benchmark_return"]).prod() - 1
    logger.info(
        f"S&P 500 backtest complete | ${capital:,.0f} | "
        f"total={total_ret:+.1%} | bench(SPY)={bench_total:+.1%} | "
        f"excess={total_ret - bench_total:+.1%}"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path)
    logger.info(f"Results saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-layer S&P 500 portfolio")
    parser.add_argument("--config", default="config/sectors/financials.yaml",
                        help="Base config (modelling params only; data section overridden)")
    parser.add_argument("--out", default="results/sp500/backtest.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        import yaml
        base_cfg = yaml.safe_load(f)

    run_sp500(base_cfg, out_path=args.out)
