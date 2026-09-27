"""
Split/dividend lookahead audit of the factor panel's market cap.

`factors.compute_factor_scores` computes market_cap = adjusted_price x EDGAR shares.
yfinance's adjusted prices divide every past price by all LATER splits (and by later
dividends), while EDGAR's `EntityCommonStockSharesOutstanding` is as-reported. So a
stock that WILL split k:1 shows a past market cap k times too small. Its B/P, E/P and
S/P come out k times too large, and its size score is too small-cap. Stocks split
after they rise, so the value factor was partly sorting on future returns.

True market cap at t:
    raw_close(t)   = Close_splitadj(t) * prod(splits after t)      (Close: not dividend-adjusted)
    shares_now(t)  = shares_reported(filing f) * prod(splits in (f, t])
    mcap(t)        = raw_close(t) * shares_now(t)

The corrected panel is built by feeding compute_factor_scores an equivalent share count
shares_equiv = mcap / adjusted_price, so everything else in the pipeline is unchanged.
Then value/size/composite premia, IC and the quintile book are compared, original vs
corrected. Non-destructive: writes only to factor_research/cache and results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "shared"))
sys.path.insert(0, str(REPO / "factor_model"))
sys.path.insert(0, str(REPO / "sector_model"))
sys.path.insert(0, str(ROOT))

from universe_pit import historical_universe, membership_matrix, fetch_sectors, fetch_prices_survivorship_free  # noqa: E402
from data_fundamentals import fetch_factor_fundamentals, fetch_cik_map  # noqa: E402
from factors import compute_factor_scores, FACTORS  # noqa: E402
from fama_macbeth import fama_macbeth, _nw_tstat  # noqa: E402

CACHE = REPO / "factor_model" / "cache"
START, END = "2015-01-01", "2026-06-30"
REBAL, WARMUP = 21, 252
OOS = pd.Timestamp("2025-01-01")
RAW_CACHE = ROOT / "cache" / "split_audit_raw.parquet"
PANEL_ORIG = REPO / "signal_combiner" / "cache" / "panel.parquet"
PANEL_FIX = ROOT / "cache" / "panel_splitfix.parquet"


def fetch_raw(tickers):
    """Split-adjusted (not dividend-adjusted) Close and split events, downloaded to today so
    every split the adjustment basis includes is also in the event list."""
    if RAW_CACHE.exists():
        return pd.read_parquet(RAW_CACHE)
    logger.info(f"Downloading Close + splits for {len(tickers)} tickers ...")
    raw = yf.download(list(tickers), start=START, auto_adjust=False, actions=True, progress=False)
    out = pd.concat({"close": raw["Close"], "splits": raw["Stock Splits"]}, axis=1)
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RAW_CACHE)
    return out


def equivalent_shares(fund, adj_px, raw):
    """Replace shares_outstanding with mcap_true / adjusted_price, per (date, ticker)."""
    close, splits = raw["close"], raw["splits"].replace(0.0, np.nan)
    fund = fund.copy()
    sh = fund["shares_outstanding"].unstack("ticker")
    dates = sh.index
    fixed, stats = {}, {"corrected": 0, "no_raw": 0}
    for t in sh.columns:
        s = sh[t]
        if t not in close.columns or close[t].notna().sum() < 60 or t not in adj_px.columns:
            fixed[t] = s                                       # cannot correct: leave as-is
            stats["no_raw"] += 1
            continue
        sp = splits[t].dropna()
        logC = np.log(sp).cumsum() if len(sp) else pd.Series(dtype=float)
        def C(x):                                              # log prod of splits on/before x
            if logC.empty:
                return np.zeros(len(x))
            i = np.searchsorted(logC.index.values, np.asarray(x, dtype="datetime64[ns]"), side="right")
            return np.where(i > 0, logC.values[np.maximum(i - 1, 0)], 0.0)
        total = logC.iloc[-1] if len(logC) else 0.0
        # filing dates: change points of the ffilled reported share series
        chg = s.ne(s.shift()) & s.notna()
        f = pd.Series(np.where(chg, s.index, pd.NaT), index=s.index).ffill()
        raw_close = close[t].reindex(dates).ffill(limit=5) * np.exp(total - C(dates))
        shares_now = s * np.exp(C(dates) - C(pd.DatetimeIndex(f.values)))
        mcap = raw_close * shares_now
        fixed[t] = mcap / adj_px[t].reindex(dates)
        stats["corrected"] += 1
    fund["shares_outstanding"] = pd.DataFrame(fixed).stack(future_stack=True).reindex(fund.index)
    return fund, stats


def build_panel_fixed(prices, fund_fixed, sectors, members_mat):
    dates = prices.index
    rows = []
    for rd in dates[WARMUP::REBAL]:
        members = None
        if rd in members_mat.index:
            row = members_mat.loc[rd]
            members = set(row.index[row.values])
        sc = compute_factor_scores(rd, prices, fund_fixed, sectors, members=members, composite_factors=FACTORS)
        if sc.empty:
            continue
        di = dates.get_loc(rd)
        end = min(di + REBAL, len(dates) - 1)
        blk = sc[FACTORS].copy()
        blk["fwd_ret"] = (prices.iloc[end] / prices.iloc[di] - 1.0).reindex(sc.index)
        blk["date"], blk["ticker"] = rd, blk.index
        rows.append(blk.dropna(subset=["fwd_ret"]))
    p = pd.concat(rows, ignore_index=True).dropna(subset=FACTORS, how="all")
    p.to_parquet(PANEL_FIX)
    return p


def ls_sharpe(panel, cols, q=0.2):
    r = []
    for d, g in panel.groupby("date"):
        s = g[cols].mean(axis=1)
        g = g[s.notna()]
        s = s.dropna().rank()
        k = max(int(len(s) * q), 1)
        r.append((d, g.loc[s.nlargest(k).index, "fwd_ret"].mean() - g.loc[s.nsmallest(k).index, "fwd_ret"].mean()))
    r = pd.Series(dict(r)) / 2                          # 0.5 long / 0.5 short, as the sleeve
    return {seg: x.mean() / x.std() * np.sqrt(12) for seg, x in
            [("full", r), ("dev", r[r.index < OOS]), ("oos", r[r.index >= OOS])]}


def main():
    tickers = historical_universe(START, END, cache_dir=str(CACHE))
    prices = fetch_prices_survivorship_free(tickers, START, END, cache_dir=str(CACHE))
    from data.sp500 import fetch_sp500_universe
    seed = fetch_sp500_universe(cache_dir=str(CACHE)).set_index("Symbol")["GICS Sector"].to_dict()
    sectors = fetch_sectors(list(prices.columns), seed=seed, cache_dir=str(CACHE))
    members_mat = membership_matrix(prices.index, cache_dir=str(CACHE))
    fund = fetch_factor_fundamentals(list(prices.columns), prices.index,
                                     cache_dir=str(CACHE / "factor_fund"), cik_map=fetch_cik_map(cache_dir=str(CACHE)))
    raw = fetch_raw(prices.columns)

    fund_fixed, stats = equivalent_shares(fund, prices, raw)
    print(f"share correction: {stats}")
    # sanity: implied mcap for known splitters, original vs corrected
    for t in ["NVDA", "TSLA", "AMZN", "AAPL", "GOOGL"]:
        for d in ["2017-06-01", "2023-06-01"]:
            dd = prices.index[prices.index.searchsorted(pd.Timestamp(d))]
            try:
                o = prices.at[dd, t] * fund.at[(dd, t), "shares_outstanding"] / 1e9
                c = prices.at[dd, t] * fund_fixed.at[(dd, t), "shares_outstanding"] / 1e9
                print(f"  {t:<6}{dd.date()}  mcap original {o:>9.1f}B   corrected {c:>9.1f}B")
            except KeyError:
                pass

    fixed = pd.read_parquet(PANEL_FIX) if PANEL_FIX.exists() else build_panel_fixed(prices, fund_fixed, sectors, members_mat)
    orig = pd.read_parquet(PANEL_ORIG)

    m = orig.merge(fixed, on=["date", "ticker"], suffixes=("_o", "_c"))
    print(f"\nmatched stock-months: {len(m):,} of {len(orig):,} original")
    print("cross-sectional corr, original vs corrected score (mean over months):")
    for f in FACTORS:
        c = m.groupby("date").apply(lambda g: g[f"{f}_o"].corr(g[f"{f}_c"])).mean()
        print(f"  {f:<9} {c:+.3f}")

    print(f"\n{'='*78}\nFama-MacBeth premia (ann.), Newey-West t — ORIGINAL vs SPLIT-CORRECTED\n{'='*78}")
    print(f"{'factor':<10}" + "".join(f"{s:>22}" for s in ["full", "dev (<2025)", "oos (2025+)"]))
    res = {}
    for name, p in [("original", orig), ("corrected", fixed)]:
        lam = fama_macbeth(p)
        for f in FACTORS:
            cells = []
            for seg, x in [("full", lam), ("dev", lam[lam.index < OOS]), ("oos", lam[lam.index >= OOS])]:
                prem, t = _nw_tstat(x[f])
                res[(name, f, seg)] = (prem * 12, t)
                cells.append(f"{prem*12:>+9.2%} t={t:>+5.2f}")
            print(f"{f:<10}" + "".join(f"{c:>22}" for c in cells) + f"   [{name}]")

    print(f"\n{'='*78}\nQuintile L/S gross Sharpe (value+quality, and value alone)\n{'='*78}")
    for name, p in [("original", orig), ("corrected", fixed)]:
        for cols in [["value", "quality"], ["value"], ["size"]]:
            s = ls_sharpe(p, cols)
            print(f"{name:<10}{'+'.join(cols):<15} full {s['full']:+.2f}   dev {s['dev']:+.2f}   oos {s['oos']:+.2f}")

    pd.Series({f"{a}|{b}|{c}": v for (a, b, c), v in res.items()}).to_csv(ROOT / "results" / "split_audit.csv")


if __name__ == "__main__":
    main()
