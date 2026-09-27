"""
edgar_pit against published 10-K figures. Each date falls after the fiscal-year 10-K
was filed and before the next 10-Q, so the TTM must equal the reported full year.
Needs the cached facts (factor_model/cache/edgar_pit). Run:
    python -c "import test_edgar_pit as t; [getattr(t, f)() for f in dir(t) if f.startswith('test_')]"
"""

import pandas as pd

from edgar_pit import fetch, ttm_flow, instant

OCF = ["NetCashProvidedByUsedInOperatingActivities",
       "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
RD = ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"]
NI = ["NetIncomeLoss", "ProfitLoss"]

# (ticker, as-of date, extractor, tags, published value in $B)
CASES = [
    ("AAPL", "2023-12-01", ttm_flow, NI, 96.995),        # FY2023 (Sep), 10-K filed 2023-11-03
    ("AAPL", "2023-12-01", ttm_flow, OCF, 110.543),
    ("AAPL", "2023-12-01", ttm_flow, RD, 29.915),
    ("AAPL", "2023-12-01", instant, ["Assets"], 352.583),
    ("MSFT", "2023-09-01", ttm_flow, NI, 72.361),        # FY2023 (Jun), 10-K filed 2023-07-27
    ("MSFT", "2023-09-01", ttm_flow, OCF, 87.582),
    ("AMZN", "2024-03-01", ttm_flow, NI, 30.425),        # FY2023, 10-K filed 2024-02-02 (old extractor: ~0)
    ("JNJ", "2024-03-01", ttm_flow, RD, 15.085),         # two-tag trap; restated FY (Kenvue)
    # 10-Q path: FY + YTD - prior-year YTD
    ("AAPL", "2023-08-10", ttm_flow, NI, 99.803 + 74.039 - 79.521),   # TTM to Jun-2023 = 94.321
    ("MSFT", "2023-12-01", ttm_flow, NI, 72.361 + 22.291 - 17.556),   # TTM to Sep-2023 = 77.096
]


def _val(t, d, fn, tags):
    facts = fetch([t], None)[t]
    return float(fn(facts, tags, pd.DatetimeIndex([d])).iloc[0]) / 1e9


def test_matches_published_10k():
    for t, d, fn, tags, pub in CASES:
        got = _val(t, d, fn, tags)
        assert abs(got / pub - 1) < 0.005, (t, d, tags[0], got, pub)


def test_not_available_before_filing():
    """The day before AAPL's FY2023 10-K was filed, the TTM must still be the Q3 version."""
    before = _val("AAPL", "2023-11-02", ttm_flow, NI)
    after = _val("AAPL", "2023-11-06", ttm_flow, NI)
    assert abs(after / 96.995 - 1) < 0.005
    assert abs(before / 96.995 - 1) > 0.005, before
