"""
Pre-registered reformulation experiments for the sector-rotation sleeve.

v0 (blend of 12-1 momentum + macro tilt, long-short top3/bottom3) was shelved
at net Sharpe ~0 on 2015-2024 (README). These experiments are NOT parameter
iteration on v0 — they are (a) v0's first true out-of-sample evaluation on
2025-2026 data it never saw, and (b) three theory-motivated reformulations,
each run exactly once against a pass bar fixed before running:

  v0_blend  original composite (0.6 z(mom) + 0.4 z(macro)), top3/bottom3.
            Bar: OOS (2025+) Sharpe > 0.
  v1_volmom rank on vol-adjusted momentum (12-1 log return / trailing 60d
            ann. vol), macro tilt dropped (its IC t=0.38; it was the flagged
            overfit risk). Same top3/bottom3 L/S construction.
            Bar: dev net Sharpe >= 0.3 AND OOS > 0.
  v2_tsmom  time-series momentum: each sector long/short by the sign of its
            OWN 12-1 return, inverse-vol weights normalized to gross 1.0.
            No cross-sectional rank. Bar: dev >= 0.3 AND OOS > 0 AND monthly
            corr to trend_model < 0.6 (else it duplicates the surviving sleeve).
  v3_dual   dual momentum, long-only: top-3 by 12-1 momentum at 1/3 each, but
            a slice goes to cash if that sector's own 12-1 return is negative.
            No shorts, no borrow. Bar: beats SPY Sharpe in dev AND in OOS
            (long-only, so the benchmark is SPY, not zero).

Timing follows v0 / LOOKAHEAD_FINDING.md: signals use closes through the
rebalance date rd; entry at rd close; P&L close(rd) -> close(rd+freq).
Costs as v0: 5 bps on turnover, 30 bps/yr borrow on short gross.

Run from sector_rotation/:
    python experiments_v1.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sector_model"))
from signals.sector_rotation import _MACRO_BETAS, fetch_macro_data  # noqa: E402

from run import fetch_etf_prices  # noqa: E402

START, END = "2014-01-01", "2026-06-30"
OOS_START = pd.Timestamp("2025-01-01")
ETF_MAP = {
    "Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
    "Industrials": "XLI", "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Energy": "XLE", "Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}
LB, SKIP, ZWIN, VOLWIN = 252, 21, 252, 60
MOM_W, MAC_W = 0.6, 0.4
N_SIDE, MIN_SECS = 3, 8
FREQ, WARMUP = 21, 252
TCOST, BORROW = 0.0005, 0.003
VOL_FLOOR = 0.05

VARIANTS = ["v0_blend", "v1_volmom", "v2_tsmom", "v3_dual"]


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd > 1e-8 else s * 0.0


def main() -> None:
    tickers = list(ETF_MAP.values())
    sector_of = {v: k for k, v in ETF_MAP.items()}

    prices = fetch_etf_prices(tickers + ["SPY"], START, END, "cache")
    spy_px = prices["SPY"]
    px = prices[[t for t in tickers if t in prices.columns]]
    logret = np.log(px / px.shift(1))
    macro = fetch_macro_data(START, END, cache_dir="cache")

    dates = px.index
    rebal_dates = dates[WARMUP::FREQ]

    prev_w = {v: pd.Series(dtype=float) for v in VARIANTS}
    records = {v: [] for v in VARIANTS}

    for rd in rebal_dates:
        t = dates.get_loc(rd)
        if t < LB:
            continue

        window = logret.iloc[t - LB: t - SKIP]
        ok = window.notna().mean() > 0.95
        mom = window.sum()[ok[ok].index]
        if len(mom) < MIN_SECS:
            continue

        vol = (logret.iloc[t - VOLWIN: t].std() * np.sqrt(252)).reindex(mom.index)
        vol = vol.fillna(vol.mean()).clip(lower=VOL_FLOOR)

        hist = macro.loc[:rd].tail(ZWIN)
        if len(hist) < ZWIN // 2:
            continue
        row = hist.iloc[-1]
        yc_z = (row["yield_curve"] - hist["yield_curve"].mean()) / (hist["yield_curve"].std() + 1e-8)
        vix_z = (row["vix"] - hist["vix"].mean()) / (hist["vix"].std() + 1e-8)
        mac = pd.Series({tk: _MACRO_BETAS.get(sector_of[tk], (0.0, 0.0))[0] * yc_z
                             + _MACRO_BETAS.get(sector_of[tk], (0.0, 0.0))[1] * vix_z
                         for tk in mom.index})

        weights = {}

        # v0: original blended composite, top3/bottom3
        comp = MOM_W * zscore(mom) + MAC_W * zscore(mac)
        ranked = comp.sort_values(ascending=False)
        w0 = pd.Series(0.0, index=mom.index)
        w0[ranked.index[:N_SIDE]] = +1.0 / N_SIDE
        w0[ranked.index[-N_SIDE:]] = -1.0 / N_SIDE
        weights["v0_blend"] = w0

        # v1: vol-adjusted momentum rank, macro dropped
        volmom = mom / vol
        ranked = volmom.sort_values(ascending=False)
        w1 = pd.Series(0.0, index=mom.index)
        w1[ranked.index[:N_SIDE]] = +1.0 / N_SIDE
        w1[ranked.index[-N_SIDE:]] = -1.0 / N_SIDE
        weights["v1_volmom"] = w1

        # v2: per-sector time-series momentum, inverse-vol, gross 1.0
        sign = np.sign(mom)
        w2 = sign / vol
        w2 = w2 / w2.abs().sum() if w2.abs().sum() > 0 else w2 * 0.0
        weights["v2_tsmom"] = w2

        # v3: long-only top-3 with absolute-momentum gate (slice to cash if
        # the sector's own 12-1 return is negative)
        top = mom.sort_values(ascending=False).index[:N_SIDE]
        w3 = pd.Series(0.0, index=mom.index)
        for tk in top:
            if mom[tk] > 0:
                w3[tk] = 1.0 / N_SIDE
        weights["v3_dual"] = w3

        ei = min(t + FREQ, len(dates) - 1)
        fwd = (px.iloc[ei] / px.iloc[t] - 1).reindex(mom.index).fillna(0.0)
        spy_ret = float(spy_px.iloc[ei] / spy_px.iloc[t] - 1)

        for v in VARIANTS:
            w = weights[v]
            all_idx = prev_w[v].index.union(w.index)
            to = float((w.reindex(all_idx).fillna(0) - prev_w[v].reindex(all_idx).fillna(0)).abs().sum())
            borrow_cost = BORROW * (ei - t) / 252.0 * float(w[w < 0].abs().sum())
            net = float((w * fwd).sum()) - to * TCOST - borrow_cost
            records[v].append({
                "date": rd, "period_return": net, "benchmark_return": spy_ret,
                "turnover": to, "gross": float(w.abs().sum()),
            })
            prev_w[v] = w

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ppy = 252 / FREQ

    def stats(r: pd.Series) -> tuple:
        if len(r) < 2 or r.std() == 0:
            return 0.0, 0.0, 0.0
        eq = (1 + r).cumprod()
        return eq.iloc[-1] - 1, r.mean() / r.std() * np.sqrt(ppy), (eq / eq.cummax() - 1).min()

    # trend_model monthly returns for the v2 redundancy check
    trend_r = None
    trend_csv = Path("../trend_model/results/backtest.csv")
    if trend_csv.exists():
        tr = pd.read_csv(trend_csv, parse_dates=["date"]).set_index("date")["period_return"]
        trend_r = tr.groupby(tr.index.to_period("M")).apply(lambda g: (1 + g).prod() - 1)

    print(f"\n{'variant':10s} {'window':4s} {'total':>8s} {'Sharpe':>7s} {'maxDD':>7s} "
          f"{'corr(SPY)':>9s} {'corr(trend)':>11s}  bar")
    for v in VARIANTS:
        df = pd.DataFrame(records[v]).set_index("date")
        df.to_csv(out_dir / f"experiment_{v}.csv")
        for label, part in [("dev", df[df.index < OOS_START]), ("OOS", df[df.index >= OOS_START])]:
            tot, sh, dd = stats(part["period_return"])
            spy_tot, spy_sh, _ = stats(part["benchmark_return"])
            c_spy = part["period_return"].corr(part["benchmark_return"])
            c_tr = np.nan
            if trend_r is not None:
                m = part["period_return"].copy()
                m.index = m.index.to_period("M")
                c_tr = m.corr(trend_r)
            bench = f" | SPY {spy_tot:+.1%}/{spy_sh:.2f}" if v == "v3_dual" else ""
            print(f"{v:10s} {label:4s} {tot:+8.1%} {sh:7.2f} {dd:+7.1%} "
                  f"{c_spy:9.2f} {c_tr:11.2f}{bench}")
    logger.info("done -> results/experiment_<variant>.csv")


if __name__ == "__main__":
    main()
