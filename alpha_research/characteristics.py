"""
C1–C5 at the panel's monthly rebalance dates, point-in-time. Definitions and predicted
signs are fixed in README.md.

Every EDGAR input comes from edgar_extra (earliest filing per period, available the day
after filing; TTM flows from per-tag YTD differencing). Share counts are the cover-page
counts with their real filing dates, put on a common split basis. Market cap comes from
shared/market_cap. Raw characteristics are winsorized, sector-neutralized and z-scored
with factor_model's own functions, so they are directly comparable with the existing
factors.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
for p in ["shared", "factor_model", "sector_model"]:
    sys.path.insert(0, str(REPO / p))
sys.path.insert(0, str(ROOT))

from edgar_pit import fetch, ttm_flow, instant, _first_filed  # noqa: E402
from market_cap import add_market_cap, _log_split_cum, _cum_at  # noqa: E402
from factors import _winsorize, _sector_neutralize  # noqa: E402
from universe_pit import fetch_sectors  # noqa: E402
from data_fundamentals import fetch_factor_fundamentals, fetch_cik_map  # noqa: E402

CACHE = REPO / "factor_model" / "cache"
PANEL = REPO / "signal_combiner" / "cache" / "panel.parquet"
PRICES = CACHE / "prices_sf_2015-01-01_2026-06-30.parquet"
OUT = ROOT / "cache" / "characteristics.parquet"
CANDS = ["asset_growth", "net_issuance", "accruals", "rd_intensity", "st_reversal"]
OCF = ["NetCashProvidedByUsedInOperatingActivities",
       "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
RD = ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"]
NI = ["NetIncomeLoss", "ProfitLoss"]
YEAR = pd.Timedelta(days=365)


def split_basis_shares(df, logC, total, dates):
    """Cover-page share count as of each date, restated to the latest split basis:
    count x prod(splits after the count's as-of date)."""
    d = _first_filed(df, ["dei:SharesOutstanding"])
    if d.empty:
        return pd.Series(np.nan, index=dates)
    d = d.assign(avail=d.filed + pd.Timedelta(days=1)).sort_values(["avail", "end"])
    d = d.drop_duplicates("avail", keep="last")
    d = d[d.end >= d.end.cummax()]
    basis = d.val.values * np.exp(total - _cum_at(logC, pd.DatetimeIndex(d.end.values)))
    s = pd.Series(basis, index=d.avail.values)
    out = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    age = pd.Series(s.index, index=s.index).reindex(s.index.union(dates)).ffill().reindex(dates)
    return out.where((pd.Series(dates, index=dates) - age).dt.days <= 400)


def raw_characteristics(panel, prices, facts, mcap, raw_splits):
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    lag = dates - YEAR
    rows = []
    for t, df in facts.items():
        if df.empty:
            continue
        A_now, A_lag = instant(df, ["Assets"], dates), instant(df, ["Assets"], lag)
        A_lag.index = dates
        ocf, ni = ttm_flow(df, OCF, dates), ttm_flow(df, NI, dates)
        rd = ttm_flow(df, RD, dates).fillna(0.0)                     # no R&D line = no R&D spend
        logC = _log_split_cum(raw_splits[t]) if t in raw_splits.columns else None
        total = logC.iloc[-1] if logC is not None else 0.0
        sh_now, sh_lag = split_basis_shares(df, logC, total, dates), split_basis_shares(df, logC, total, lag)
        sh_lag.index = dates
        mc = mcap[t].reindex(dates) if t in mcap.columns else pd.Series(np.nan, index=dates)
        rows.append(pd.DataFrame({
            "date": dates, "ticker": t,
            "asset_growth": (A_now / A_lag - 1).values,
            "net_issuance": np.log(sh_now / sh_lag).values,
            "accruals": ((ni - ocf) / ((A_now + A_lag) / 2)).values,
            "rd_intensity": (rd / mc).values,
        }))
    ch = pd.concat(rows, ignore_index=True)
    # C5: return over the 21 trading days ending at the rebalance date
    pos = prices.index.get_indexer(dates)
    rev = (prices.iloc[pos] / prices.iloc[pos - 21].values - 1).stack().rename("st_reversal").reset_index()
    rev.columns = ["date", "ticker", "st_reversal"]
    ch = ch.merge(rev, on=["date", "ticker"], how="outer")
    return panel[["date", "ticker"]].merge(ch, on=["date", "ticker"], how="left")


def neutralize(raw, sectors):
    out = []
    for d, g in raw.groupby("date"):
        g = g.set_index("ticker")
        z = pd.DataFrame(index=g.index)
        for c in CANDS:
            x = g[c].replace([np.inf, -np.inf], np.nan)
            ok = x.notna()
            z[c] = np.nan
            if ok.sum() >= 30:
                z.loc[ok, c] = _sector_neutralize(_winsorize(x[ok]), sectors)
        z["date"] = d
        out.append(z.reset_index())
    return pd.concat(out, ignore_index=True)


def build():
    panel = pd.read_parquet(PANEL)
    prices = pd.read_parquet(PRICES)
    tickers = sorted(panel.ticker.unique())
    facts = fetch(tickers, str(CACHE))                                # shared cache: factor_model/cache/edgar_pit
    fund = fetch_factor_fundamentals(list(prices.columns), prices.index,
                                     cache_dir=str(CACHE / "factor_fund"), cik_map=fetch_cik_map(cache_dir=str(CACHE)))
    fund = add_market_cap(fund, prices, "2015-01-01", cache_dir=str(CACHE))
    mcap = fund["market_cap"].unstack("ticker")
    raw_splits = pd.read_parquet(CACHE / "raw_close_splits.parquet")["splits"]
    sys.path.insert(0, str(REPO / "sector_model"))
    from data.sp500 import fetch_sp500_universe
    seed = fetch_sp500_universe(cache_dir=str(CACHE)).set_index("Symbol")["GICS Sector"].to_dict()
    sectors = fetch_sectors(list(prices.columns), seed=seed, cache_dir=str(CACHE))

    raw = raw_characteristics(panel, prices, facts, mcap, raw_splits)
    z = neutralize(raw, sectors)
    out = raw.rename(columns={c: f"{c}_raw" for c in CANDS}).merge(z, on=["date", "ticker"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    cov = out.groupby("date")[CANDS].apply(lambda g: g.notna().mean()).mean()
    print("mean monthly coverage of the universe:\n" + cov.round(3).to_string())
    return out


if __name__ == "__main__":
    build()
