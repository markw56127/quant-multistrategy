"""The F2/F3 universe must stay the pre-registered 50, whatever the surface files contain."""

import pandas as pd

import signals


def _fake_download(tickers, **kw):
    """Constant prices: realized vol is zero, which is fine, since only the universe is tested."""
    idx = pd.date_range("2098-12-01", periods=30)
    return pd.concat({"Close": pd.DataFrame(1.0, index=idx, columns=list(tickers))}, axis=1)


def test_f2_f3_universe_is_frozen():
    day = "2099-01-01"
    rows = [{"underlying": t, "kind": "stock", "capture": "regular", "skew30": 0.05, "atm30": 0.3}
            for t in signals.CORE + ["ZZZ1", "ZZZ2", "ZZZ3"]]           # S&P 500 expansion names
    rows.append({"underlying": "AAPL", "kind": "stock", "capture": "after_hours", "skew30": 0.9, "atm30": 0.9})
    path = signals.PARAMS / f"{day}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    orig = signals.yf.download
    try:
        signals.yf.download = _fake_download
        out = signals.f2_f3_options(day)
    finally:
        signals.yf.download = orig
        path.unlink()
    assert len(signals.CORE) == 50
    assert set(out.index) == set(signals.CORE)                       # expansion names excluded
    assert out.loc["AAPL", "F2_skew"] == 0.05                         # the after-hours row is ignored
