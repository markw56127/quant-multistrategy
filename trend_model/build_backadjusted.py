"""
Build back-adjusted continuous futures series from individual contract bars.

Input:  cache/contracts_daily.parquet (from fetch_databento.py), long format
        [date, root, symbol, close, volume] — or any file matching that schema
        from another vendor.
Output: cache/backadjusted.parquet — one back-adjusted close series per root,
        columns named like the yfinance tickers ("ES=F", ...) so trend_model
        can consume them directly (config: data.backadjusted_file), plus
        cache/roll_calendar.csv for audit.

Method (standard Panama / difference back-adjustment):
  - Contract order comes from the symbol's month-year code (expiry order).
  - The active contract rolls to the next-in-order contract after that
    contract's volume exceeds the active one's for 2 consecutive sessions
    (volume-crossover rule; robust to short data gaps).
  - On a roll date the price gap (new close − old close) is recorded, and the
    entire history BEFORE the roll is shifted by the cumulative gap, so daily
    differences across the roll equal the tradeable contract-to-contract P&L
    and no fake roll return ever enters the series.

Difference-adjusted series can go negative in long histories; percent returns
must therefore be computed with care. trend_model consumes the series through
pct_change on a level-shifted variant — see load_backadjusted() in run.py.

Run from trend_model/:
    python build_backadjusted.py
Self-test (synthetic contracts, no data purchase needed):
    python build_backadjusted.py --self-test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

MONTH_CODES = {c: i + 1 for i, c in enumerate("FGHJKMNQUVXZ")}
ROOT_TO_TICKER = {"ES": "ES=F", "NQ": "NQ=F", "YM": "YM=F", "RTY": "RTY=F",
                  "ZT": "ZT=F", "ZF": "ZF=F", "ZN": "ZN=F", "ZB": "ZB=F"}
CROSSOVER_DAYS = 2


def expiry_key(symbol: str, root: str, sample_year: int) -> tuple:
    """Sortable (year, month) from a contract symbol like ESZ5 / ESZ25."""
    tail = symbol[len(root):]
    month = MONTH_CODES[tail[0]]
    yr = int(tail[1:])
    if yr < 100:  # 1- or 2-digit year → resolve to the decade around the data
        base = sample_year - sample_year % 10
        yr = base + yr if yr < 10 else 2000 + yr
        while yr < sample_year - 1:
            yr += 10
    return (yr, month)


def stitch_root(df: pd.DataFrame, root: str) -> tuple:
    """Back-adjusted close series + roll log for one root."""
    df = df.sort_values("date")
    sample_year = int(df["date"].dt.year.min())
    order = {s: expiry_key(s, root, sample_year) for s in df["symbol"].unique()}
    dates = sorted(df["date"].unique())
    px = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    vol = df.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last")
    contracts = sorted(order, key=order.get)

    # start on the highest-volume live contract of the first date
    first_live = vol.loc[dates[0]].dropna()
    if first_live.empty:
        return pd.Series(dtype=float), []
    active = max(first_live.index, key=lambda s: (first_live[s], ))
    active_i = contracts.index(active)

    raw, rolls, streak = {}, [], 0
    for dt in dates:
        nxt = contracts[active_i + 1] if active_i + 1 < len(contracts) else None
        v_a = vol.at[dt, active] if active in vol.columns else np.nan
        v_n = vol.at[dt, nxt] if nxt and nxt in vol.columns else np.nan
        streak = streak + 1 if (pd.notna(v_n) and (pd.isna(v_a) or v_n > v_a)) else 0

        if streak >= CROSSOVER_DAYS and nxt is not None and pd.notna(px.at[dt, nxt]):
            old_close = px.at[dt, active] if pd.notna(px.at[dt, active]) else raw.get(dt, np.nan)
            gap = px.at[dt, nxt] - old_close if pd.notna(old_close) else 0.0
            rolls.append({"root": root, "date": dt, "from": active, "to": nxt, "gap": gap})
            active, active_i, streak = nxt, active_i + 1, 0

        p = px.at[dt, active] if active in px.columns else np.nan
        if pd.notna(p):
            raw[dt] = (p, len(rolls))          # close of the then-active contract

    if not raw:
        return pd.Series(dtype=float), pd.Series(dtype=float), rolls
    ser = pd.Series({d: v[0] for d, v in raw.items()}).sort_index()
    n_rolls_at = pd.Series({d: v[1] for d, v in raw.items()}).sort_index()
    # Panama: shift all history BEFORE each roll by that roll's gap so the
    # series is continuous through every roll (latest prices unchanged)
    gaps = np.array([r["gap"] for r in rolls])
    cum_after = np.concatenate([np.cumsum(gaps[::-1])[::-1], [0.0]])  # gaps still ahead
    adj = ser + pd.Series(cum_after[n_rolls_at.values], index=ser.index)
    # Tradeable daily percent return: point change of the held contract over
    # the prior day's ACTUAL contract close (adj alone would mis-scale returns
    # once cumulative gaps push the adjusted level away from traded prices)
    rets = adj.diff() / ser.shift(1)
    return adj, rets, rolls


def self_test() -> None:
    """Two synthetic contracts with a known gap: the adjusted series must have
    zero return on the roll date if both contracts' prices are flat."""
    rows = []
    d = pd.bdate_range("2020-01-01", periods=20)
    for i, dt in enumerate(d):
        rows.append({"date": dt, "root": "ES", "symbol": "ESH0",
                     "close": 100.0, "volume": 1000 if i < 10 else 10})
        rows.append({"date": dt, "root": "ES", "symbol": "ESM0",
                     "close": 95.0, "volume": 10 if i < 10 else 1000})
    adj, rets, rolls = stitch_root(pd.DataFrame(rows), "ES")
    assert len(rolls) == 1, f"expected 1 roll, got {len(rolls)}"
    assert abs(rolls[0]["gap"] - (-5.0)) < 1e-9, f"gap wrong: {rolls[0]['gap']}"
    assert (rets.dropna().abs() < 1e-9).all(), "flat contracts must give zero returns"
    assert abs(adj.iloc[-1] - 95.0) < 1e-9, "latest price must be unadjusted"
    assert abs(adj.iloc[0] - 95.0) < 1e-9, "pre-roll history must be shifted by the gap"
    # trending variant: daily +1 point in both contracts must survive the roll
    rows2 = []
    for i, dt in enumerate(d):
        rows2.append({"date": dt, "root": "ES", "symbol": "ESH0",
                      "close": 100.0 + i, "volume": 1000 if i < 10 else 10})
        rows2.append({"date": dt, "root": "ES", "symbol": "ESM0",
                      "close": 95.0 + i, "volume": 10 if i < 10 else 1000})
    adj2, rets2, rolls2 = stitch_root(pd.DataFrame(rows2), "ES")
    assert len(rolls2) == 1
    assert np.allclose(adj2.diff().dropna().values, 1.0), "trend must be +1/day through roll"
    r2 = rets2.dropna()
    assert (r2 > 0).all(), "uptrend must give positive returns every day"
    # +1 point/day on traded closes ranging 95..119 → returns within (1/120, 1/94)
    assert r2.max() < 1 / 94 and r2.min() > 1 / 120, "returns mis-scaled vs traded closes"
    print("self-test OK: gaps removed, latest prices unadjusted, trends preserved, returns scaled")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--contracts", default="cache/contracts_daily.parquet")
    ap.add_argument("--out", default="cache/backadjusted.parquet")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    src = Path(args.contracts)
    if not src.exists():
        raise SystemExit(f"{src} not found — run fetch_databento.py first")
    df = pd.read_parquet(src)
    df["date"] = pd.to_datetime(df["date"])

    levels, returns, all_rolls = {}, {}, []
    for root, g in df.groupby("root"):
        adj, rets, rolls = stitch_root(g, root)
        tk = ROOT_TO_TICKER.get(root, root)
        levels[tk], returns[tk] = adj, rets
        all_rolls.extend(rolls)
        logger.info(f"{root}: {len(adj)} days, {len(rolls)} rolls, "
                    f"cum gap {sum(r['gap'] for r in rolls):+.2f}")

    pd.DataFrame(levels).sort_index().to_parquet(args.out)
    rets_path = args.out.replace(".parquet", "_returns.parquet")
    pd.DataFrame(returns).sort_index().to_parquet(rets_path)
    pd.DataFrame(all_rolls).to_csv("cache/roll_calendar.csv", index=False)
    logger.info(f"→ {args.out}, {rets_path}, cache/roll_calendar.csv")
    logger.info("Point trend_model at it:  data.backadjusted_returns: "
                "cache/backadjusted_returns.parquet")


if __name__ == "__main__":
    main()
