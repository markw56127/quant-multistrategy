"""
Roll-noise validation of the trend sleeve via total-return ETF proxies.

The sleeve's known data caveat (README): yfinance front-month "=F" series are
not back-adjusted, so roll gaps are noise, defended by clipping daily returns
at ±15%. Free, truly back-adjusted continuous futures do not exist (Quandl
CHRIS is discontinued; Norgate/CSI are paid). This script is the free
alternative test: run the IDENTICAL TSMOM construction on dividend-adjusted
ETF equivalents in EXCESS-of-cash terms — a roll-free replication of the same
underlying exposures. If it reproduces the futures book's dev/OOS profile,
the trend result is not a roll-gap artifact.

Proxy map (duration/beta-matched as closely as free ETFs allow):
  ES=F->SPY   NQ=F->QQQ   YM=F->DIA   RTY=F->IWM
  ZT=F->SHY (1-3y)  ZF=F->IEI (3-7y)  ZN=F->IEF (7-10y)  ZB=F->TLT (20y+)

Excess return = ETF total return - 3m T-bill (FRED DGS3MO), matching what a
fully-collateralized futures position earns. Signal is computed on the
compounded excess-return index; all parameters are trend_model's, unchanged.

Run from trend_model/:
    python validate_etf_proxy.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

PROXY = {"ES=F": "SPY", "NQ=F": "QQQ", "YM=F": "DIA", "RTY=F": "IWM",
         "ZT=F": "SHY", "ZF=F": "IEI", "ZN=F": "IEF", "ZB=F": "TLT"}
START, END = "2015-01-01", "2026-06-30"
OOS = pd.Timestamp("2025-01-01")

# trend_model/config.yaml parameters, verbatim
LOOKBACKS = [63, 126, 252]
REBAL_FREQ, WARMUP, VOL_WINDOW = 21, 252, 60
CLIP, VOL_FLOOR, MAX_W = 0.15, 0.06, 0.25
INST_VT, PORT_VT, MAX_LEV, TCOST = 0.10, 0.12, 3.0, 0.0005


def fetch_etfs() -> pd.DataFrame:
    p = Path("cache") / f"etf_proxy_{START}_{END}.parquet"
    if p.exists():
        logger.info(f"Loading cached ETF proxies from {p}")
        return pd.read_parquet(p)
    raw = yf.download(list(PROXY.values()), start=START, end=END,
                      auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.ffill(limit=5)
    close.to_parquet(p)
    return close


def fetch_cash(dates: pd.DatetimeIndex) -> pd.Series:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
    df = pd.read_csv(url, na_values=".")
    df.columns = ["date", "y"]
    y = df.set_index(pd.to_datetime(df["date"]))["y"]
    return (y / 100.0 / 252.0).reindex(dates, method="ffill").fillna(0.0)


def main() -> None:
    px = fetch_etfs()
    cash = fetch_cash(px.index)
    xret = (px.pct_change().sub(cash, axis=0)).clip(-CLIP, CLIP)
    xidx = (1.0 + xret.fillna(0.0)).cumprod()          # excess-return index for the signal
    dates = px.index
    rebal_dates = dates[WARMUP::REBAL_FREQ]
    max_lb = max(LOOKBACKS)

    capital, prev_w = 1.0, pd.Series(0.0, index=px.columns)
    unlev_hist, records = [], []
    for rd in rebal_dates:
        di = dates.get_loc(rd)
        if di < max_lb:
            continue
        now = xidx.iloc[di]
        sig = pd.Series(0.0, index=px.columns)
        for L in LOOKBACKS:
            sig = sig.add(np.sign(now / xidx.iloc[di - L] - 1.0), fill_value=0.0)
        sig /= len(LOOKBACKS)

        win = xret.iloc[di - VOL_WINDOW:di]
        inst_vol = (win.std() * np.sqrt(252)).clip(lower=VOL_FLOOR)
        valid = px.iloc[di].notna() & inst_vol.notna() & (inst_vol > 0)
        raw_w = pd.Series(0.0, index=px.columns)
        raw_w[valid] = (sig[valid] * INST_VT / inst_vol[valid]).clip(-MAX_W, MAX_W)

        end_idx = min(di + REBAL_FREQ, len(dates) - 1)
        hold = xret.iloc[di + 1:end_idx + 1]
        fwd = ((1.0 + hold).prod() - 1.0).reindex(raw_w.index).fillna(0.0)
        unlev_ret = float((raw_w * fwd).sum())

        if len(unlev_hist) >= 6:
            book_vol = np.std(unlev_hist, ddof=1) * np.sqrt(12)
            lev = min(PORT_VT / book_vol, MAX_LEV) if book_vol > 0 else 1.0
        else:
            lev = 1.0
        w = raw_w * lev
        to = float((w - prev_w).abs().sum())
        net = lev * unlev_ret - to * TCOST
        capital *= 1.0 + net
        records.append({"date": rd, "period_return": net})
        unlev_hist.append(unlev_ret)
        prev_w = w

    res = pd.DataFrame(records).set_index("date")["period_return"]
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/etf_proxy_validation.csv")

    fut = pd.read_csv("results/backtest.csv", parse_dates=["date"]).set_index("date")["period_return"]

    def stats(r):
        eq = (1 + r).cumprod()
        return (f"total={eq.iloc[-1]-1:+7.1%}  Sharpe={r.mean()/r.std()*np.sqrt(12):+5.2f}  "
                f"maxDD={(eq/eq.cummax()-1).min():+.1%}")

    print(f"\n{'book':22s} {'window':6s} stats")
    for name, r in [("futures (=F, clipped)", fut), ("ETF proxy (roll-free)", res)]:
        print(f"{name:22s} dev    {stats(r[r.index < OOS])}")
        print(f"{name:22s} OOS    {stats(r[r.index >= OOS])}")
    both = pd.concat([fut.rename("fut"), res.rename("etf")], axis=1)
    both.index = both.index.to_period("M")
    both = both.groupby(both.index).sum(min_count=1).dropna()
    print(f"\nmonthly corr(futures book, ETF-proxy book) = {both['fut'].corr(both['etf']):+.2f} (n={len(both)})")


if __name__ == "__main__":
    main()
