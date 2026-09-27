# alpha_research — mechanism-first alpha screening

**Status:** pre-registration FIXED 2026-09-26, before any candidate was computed.
**Result (2026-09-26): NO candidate passes.** 0 of 6 survive Holm at 5% FWER. See Results.

## Why this exists

After the split-lookahead fix ([../SPLIT_FINDING.md](../SPLIT_FINDING.md)), no factor in
the repo has a development-period edge. `dynamic_trading` showed that better
optimization cannot create one. This directory is the alpha layer. Its method is the
point:

1. **Mechanism first.** A candidate enters only with a stated economic reason: who is
   on the other side, and why they accept a worse price. That turns open-ended data
   mining into a small, pre-declared family of hypotheses.
2. **Predicted sign, declared in advance.** Tests are one-sided in that direction. A
   "significant" result with the wrong sign counts as a failure, not a discovery.
3. **One family, one multiple-testing correction.** All six candidates are tested
   together under Holm–Bonferroni at a 5% familywise error rate. A candidate cannot
   be dropped from the family after its result is seen.
4. **Dev only.** Out-of-sample data (2025+) is not touched here. Any survivors get
   **one** OOS look, as a family, after sign-off.

## The family (six candidates)

**Data correction made before any candidate was computed (2026-09-26).** Building C3 exposed
a second pipeline defect: factor_model's TTM flows dropped fourth quarters (AMZN's TTM net
income read $0.2B at 2023-09 against ~$20B). This affects E/P, S/P, ROE and GP/A, which are
the *controls* in the incrementality test. The extraction was rebuilt (`shared/edgar_pit.py`,
tested against published 10-K/10-Q figures) and the panel regenerated **before the screen
ran**. No test statistic had been seen. The candidate family, signs and tests above are
unchanged.

All characteristics are computed at the panel's monthly rebalance dates for its
point-in-time S&P 500 universe, from data available at that date. EDGAR items are
indexed by filing date + 1 day. Market cap comes from `shared/market_cap.py`. Each
characteristic is winsorized at 2/98%, sector-neutralized and z-scored, exactly as in
`factor_model/factors.py`.

| # | candidate | construction | sign | mechanism | reference |
|---|---|---|---|---|---|
| C1 | **asset growth** | total assets_t / total assets_{t−12m} − 1 | − | Firms that grow assets fast overinvest (empire building), and investors extrapolate past growth; in the q-theory reading, high investment signals a low discount rate. | Cooper, Gulen & Schill 2008; FF5 CMA |
| C2 | **net share issuance** | log(split-adj. shares_t / split-adj. shares_{t−12m}) | − | Managers know more than the market. They issue equity when it is overvalued and buy back when it is undervalued, and the market underreacts to the signal. | Pontiff & Woodgate 2008; Daniel & Titman 2006 |
| C3 | **accruals** | (net income_TTM − operating cash flow_TTM) / average total assets over 12m | − | Earnings = cash flow + accruals, and accruals are the less persistent part. Investors fixate on headline earnings, so high-accrual earnings are overpriced. | Sloan 1996 |
| C4 | **R&D intensity** | R&D expense_TTM / market cap (missing R&D = 0) | + | GAAP expenses R&D immediately, so earnings understate firms that invest in intangibles. Investors underreact to future payoffs that are not yet in earnings. | Chan, Lakonishok & Sougiannis 2001 |
| C5 | **short-term reversal** | return over the last 21 trading days | − | Liquidity provision. Uninformed order imbalances push prices away from value, and whoever absorbs them is paid when the prices revert. | Jegadeesh 1990; Nagel 2012 |
| C6 | **index-deletion rebound** (event) | mean CAR[+1, +21] trading days after the effective date of S&P 500 removal, vs the equal-weight universe | + | Index funds must sell at the effective-date close whatever the price. The forced selling depresses the price, and it reverts once that selling is done. | Chen, Noronha & Singal 2004 |

C6 excludes deletions caused by delisting or acquisition, meaning names whose price
series ends within 5 trading days of the effective date. They cannot be traded, and
their removal reflects a corporate event, not index mechanics.

## Tests

- **Sample:** dev only. For C1–C5, rebalance dates whose forward-return window ends
  before 2025-01-01 (107 months). For C6, events whose [+1, +21] window ends before
  2025-01-01.
- **Primary statistic, C1–C5:** the univariate Fama–MacBeth slope of the next-month
  return on the characteristic z-score, with a Newey–West (3 lags) t and a
  one-sided p in the predicted direction.
- **Primary statistic, C6:** the mean CAR, with the t computed over calendar-month
  averages of event CARs (clustering by month), and a one-sided p.
- **Pass (all of these):**
  1. The candidate survives **Holm–Bonferroni across all six at FWER 5%**.
  2. For C1–C5, it is **incremental**: its slope in a joint Fama–MacBeth regression
     with the five existing (corrected) factors has the predicted sign, one-sided
     p < 0.05. For C6: CAR[+1, +63] also has the predicted sign, one-sided p < 0.05,
     so the effect is not a 1–2 day bounce.
- **Reported, no bar:**
  - the Harvey–Liu–Zhu hurdle (|t| > 3)
  - quintile L/S gross and net (10 bps) Sharpe
  - mean IC
  - subperiod t-stats for 2016–19 and 2020–24
  - correlations among candidates and with the existing factors
- **OOS:** untouched. If anything passes, the survivors, and only they, get one joint
  OOS look after sign-off.

## Files

| file | role |
|---|---|
| `../shared/edgar_pit.py` | point-in-time EDGAR extraction, shared with factor_model: TTM from each filing's own numbers, with its tests in `../shared/test_edgar_pit.py` |
| `characteristics.py` | C1–C5 at panel dates, point-in-time |
| `screen.py` | the pre-registered tests for C1–C5, plus the Holm correction over the whole family |
| `events.py` | C6 index-deletion event study |

## Results (dev, 107 rebalances 2016-01 → 2024-11; `python screen.py`)

| candidate | predicted sign | premium/yr per 1σ | t | one-sided p | Holm threshold | joint t | L/S gross / net SR | t 2016–19 / 2020–24 |
|---|---|---|---|---|---|---|---|---|
| accruals | − | −0.85% | **−1.72** | 0.045 | 0.0083 | −0.62 | +0.65 / +0.54 | −1.86 / −0.68 |
| short-term reversal | − | −1.84% | −1.51 | 0.068 | 0.0100 | −1.18 | +0.43 / +0.13 | −1.50 / −0.87 |
| index-deletion rebound | + | CAR21 +0.97% (n = 83) | +0.74 | 0.233 | 0.0125 | CAR63 +0.77 | — | — |
| R&D intensity | + | −0.11% | −0.16 | 0.564 | 0.0167 | −0.40 | −0.39 / −0.43 | +1.03 / −1.02 |
| net share issuance | − | **+0.31%** | +0.51 | 0.695 | 0.0250 | +0.32 | −0.08 / −0.15 | +0.88 / −0.03 |
| asset growth | − | **+0.65%** | +0.75 | 0.772 | 0.0500 | +1.15 | −0.49 / −0.57 | +0.51 / +0.62 |

**Verdict: nothing passes.** No candidate clears even the most lenient Holm step
(p ≤ 0.05 for the last-ranked one). None reaches the Harvey–Liu–Zhu |t| > 3 hurdle.

**Reading it:**

- **Accruals is the only near-miss.** It has the right sign in both subperiods,
  nominal p = 0.045, and the best long-short book (net SR 0.54). But it fails the
  family correction, and its joint t (−0.62) says most of it overlaps with the
  existing factors: it correlates 0.26 with value. Under the pre-registration it is a
  fail. It is **not** to be re-tested alone on dev. Singling it out now would be
  exactly the selection the family correction exists to prevent.
- **Asset growth and net issuance have the wrong sign.** These are two of the
  best-documented anomalies of the 1990s–2000s, and in 2016–2024 S&P 500 large caps
  they went the other way. That fits post-publication decay (McLean & Pontiff 2016)
  and the buyback era: heavy repurchasers were not rewarded.
- **Short-term reversal has the right sign but is weak, and costs eat it**
  (gross 0.43 → net 0.13 at 10 bps).
- **Index-deletion rebound has the right sign but is weak**, and its sample has
  a survivorship limit. Only 125 of 261 removals have free price data; names that
  later delisted entirely are missing. 83 events remain after excluding M&A and
  delisting removals.
- **The candidates are nearly independent** (|ρ| ≤ 0.1, except asset growth with
  issuance at 0.50). The family was genuinely six tests, not one repeated.

**What this establishes.** Among published, mechanism-backed anomalies computable
from free EDGAR and price data, none has a detectable edge in 2016–24 S&P 500 large
caps. This is the result the literature on anomaly decay predicts. The OOS window is
**untouched**, because nothing earned a look.

**Where the evidence points next:**

1. **Different information, not different transforms of the same statements.** Every
   candidate here was a transform of public financial statements or prices, and the
   market prices those well. The options snapshots (live since 2026-07) are the one
   dataset in this repo that almost no one has in point-in-time form for free.
2. **A different pond.** Anomalies are documented to be stronger in small and mid
   caps, but that needs survivorship-free data (see `../smallcap_factor`).
3. **Horizon or mechanism, not characteristic.** Event-driven, forced-trading
   mechanisms, like C6 but with complete data.

