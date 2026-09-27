# flow_events — forced-trading mechanisms

Event studies of **price-insensitive, predictable flows**: someone must trade on a known
schedule, whatever the price. This is the "mechanism" source of edge. The counterparty
isn't out-guessed; they are constrained. Each study is pre-registered in the same style
as `../alpha_research`.

## Study 1: month-end balanced-fund rebalancing

**Status:** pre-registration FIXED 2026-09-26, before any price was loaded.
**Result: DEV PASS, OOS NOT CONFIRMED → not adopted.** It looks like post-publication decay (see Results).

### Mechanism

Pension and balanced funds hold fixed stock/bond targets (e.g. 60/40). When stocks beat
bonds during a month, the stock weight drifts above target, and calendar rebalancing
forces them to **sell stocks and buy bonds near month-end**, whatever the price. The flow
is large (trillions in fixed-weight mandates), scheduled, and signed by the month's
stock-minus-bond performance, which is observable before the rebalancing window. Once
the forced flow is absorbed, the pressure should unwind in the following days.

Reference: Harvey, Mazzoleni & Melone, "The Unintended Consequences of Rebalancing"
(2023–25). **This is a published effect**, so decay or crowding is the main risk. That is
why the bar requires it to hold in both halves of the sample.

### Definitions (fixed)

- **Assets:** SPY (equity) and IEF (7–10y Treasuries, close to the duration of a typical
  balanced fund's bond sleeve). Adjusted closes, so total return. TLT is reported as a
  diagnostic with no bar.
- **Month m.** Let N be its last trading day. Days are counted back from N: N−1, N−2, …
- **Signal** S_m = r_SPY − r_IEF from the close of the previous month's last trading day
  to the close of **N−3**. It is fully known before the window opens.
- **H1, rebalancing window:** the spread return R_m = r_SPY − r_IEF from the close of N−3
  to the close of N (the last 3 trading days). **Predicted: slope of R_m on S_m < 0**
  (outperforming stocks get sold).
- **H2, unwind:** the spread return over the first 3 trading days of month m+1.
  **Predicted: slope > 0**, as the pressure reverses.
- **Strategy (for economics):** hold −sign(S_m) units of (SPY − IEF) over the H1 window,
  i.e. short the leg that outperformed. Cost is **2 bps per leg per transaction**, so
  8 bps per month for entering and exiting both legs.

### Tests

- **Samples:** dev = months with the H1 window ending 2003-01 → 2024-12. IEF starts
  2002-07, so 2002 is skipped to let the first signal month be complete. OOS = 2025-01
  onward, **untouched**.
- **Statistic:** time-series OLS of the window spread on S_m, one observation per month,
  Newey–West (3 lags) t, one-sided in the predicted direction.
- **Family:** {H1, H2}, Holm at 5% FWER.
- **Pass:**
  1. H1 survives Holm, **and**
  2. the H1 sign-strategy net Sharpe (monthly P&L, annualized) is > 0 in **both halves**
     of dev (2003–2013 and 2014–2024).

  H2 is reported, and can pass Holm on its own, but it is not required for H1.
- **Reported, no bar:** window lengths k ∈ {1…5} (varying N−k) as a sensitivity map.
  **The primary result is k = 3 only.** Also reported: TLT instead of IEF; gross and net
  Sharpe; hit rate; the slope in the top vs bottom tercile of |S|.
- **OOS and paper:** if study 1 passes, it gets one OOS look (2025–2026-09). If that
  holds, it is paper-traded forward.

### Results (2026-09-26, `python month_end.py`)

**Dev (2003–2024, 263 months, SPY−IEF, k = 3): PASS.**

| | value |
|---|---|
| H1 window slope (predicted < 0) | −0.097, **t = −3.28**, one-sided p = 0.0006, passes Holm |
| H2 unwind slope (predicted > 0) | +0.075, t = +2.15, p = 0.016, passes Holm |
| strategy (short the leg that outperformed) | gross SR 0.48, **net SR 0.34**, hit 59%, +18 bps/month net |
| net SR by half | 2003–13: 0.42 (H1 t −4.38); **2014–24: 0.24 (H1 t −0.95)** |
| sensitivity k = 1–5, IEF and TLT | right sign in all 10 (H1 t from −2.06 to −4.16) |

**Post-hoc diagnostics (no bar):**

- Robustness: excluding 2008–09 gives t = −2.54 (net SR 0.27); winsorizing S at 5/95%
  gives t = −2.89; Spearman ρ(R, S) = −0.24.
- **Placebo (same test with the window moved earlier in the month):** N−6: +1.22,
  N−8: −0.24, N−10: −0.75, N−12: −1.60, N−14: −2.02. The month-end window is the
  strongest, and the adjacent windows are flat, which supports a month-end mechanism.
  But the early-month windows show some reversal, so **part of the effect may be a
  generic short-horizon reversal of the stock–bond spread**, not only rebalancing flow.

**OOS (2025-01 → 2026-08, 20 months), the single pre-registered look: NOT CONFIRMED.**

| | value |
|---|---|
| H1 slope | +0.009, t = +0.12 (no effect) |
| H2 slope | +0.068, t = +0.65 |
| strategy | gross SR +0.10, **net SR −0.14**, hit 45% |

The Sharpe SE over 20 months is about ±0.77, so the OOS result cannot rule out the dev
Sharpe of 0.34. It also gives no support. Together with the fading second half (H1 t
−4.38 → −0.95) and the effect's publication around 2023, the most likely reading is
**post-publication decay**, the same fate as PEAD in this repo.

**Verdict:** under the pre-registration, not adopted and not paper-traded. It is the first
candidate in this repo to pass a pre-registered dev bar on clean data, and the decay
pattern is itself informative. Mechanisms that are *published* get arbitraged. The flow
still exists, but others now trade ahead of it.


## Study 2: Treasury month-end index extension

**Status:** pre-registration FIXED 2026-09-26, before any study-2 statistic was computed.
**Result: DEV PASS, OOS NOT CONFIRMED → not adopted.** Same decay pattern as study 1.
Study 1's price cache (SPY/IEF/TLT) exists, but no TLT/SHY month-end spread statistic had
been looked at.

### Mechanism

Treasury indices (e.g. Bloomberg US Treasury) rebalance at **month-end**. Newly issued
bonds enter, bonds under one year drop out, and the index's duration jumps. Passive and
benchmarked bond funds must extend duration **at the month-end close** to match, buying
long Treasuries whatever the price. The extension is largest in **quarterly refunding
months (Feb, May, Aug, Nov)**, when new 10y and 30y bonds are issued. That gives a
dose-response test: if the mechanism is real, the effect should be bigger then. Capacity
is enormous and the trade is well known to dealers, so it may be fully arbitraged. This
is a falsification test.

### Definitions (fixed)

- **Spread:** TLT − SHY daily total return (long duration minus short duration; SHY
  starts 2002-07). Raw, not duration-matched, because the flow is *into* duration.
- **Window:** the last **2** trading days of month m (close of N−2 to close of N).
- **H1:** mean window spread > 0.
- **H2 (dose-response):** mean window spread in refunding months (Feb, May, Aug, Nov) >
  mean in the other months (Welch t, one-sided).
- **Inference:** H1 is a t-test over months (one observation per month; windows don't
  overlap). The family is {H1, H2}, Holm at 5% FWER.
- **Pass:** H1 survives Holm **and** its mean is > 0 in both halves (2003–13 and 2014–24).
  H2 is reported, and passing it strengthens the mechanism reading.
- **Dev:** 2003-01 → 2024-12. **OOS:** 2025-01 onward, with one look only if dev passes.
- **Cost:** 2 bps per leg per transaction, so 8 bps per monthly round trip.
- **Reported, no bar:** a placebo window (the 2 days ending N−10); window lengths 1–5; IEF
  in place of TLT.

### Results (2026-09-26, `python treasury_extension.py`)

**Dev (2003–2024, 264 months): PASS.**

| | value |
|---|---|
| H1: TLT−SHY over the last 2 days | **+26.2 bps, t = +3.98**, passes Holm |
| H2: refunding vs other months | +27.7 vs +25.4 bps, t = +0.16, **fails** |
| halves | 2003–13 +41.6 bps; **2014–24 +10.7 bps** |
| strategy | gross SR 0.85, **net SR 0.59**, hit 62% |
| sensitivity | significant for all windows of 1–5 days, with both TLT and IEF (t from 3.1 to 4.4) |

**Diagnostics (post-hoc, no bar):**

- **Flaw in the pre-registration:** H1 tested "mean > 0", which bull-market drift
  in long bonds could partly satisfy. The right comparison is against other days.
  Month-end days average **+13.1 bps/day vs −0.7 bps/day** on all other days
  (Welch t = +3.93). By half: 2003–13 t = +4.20, **2014–24 t = +1.25**. Drift does not
  explain it.
- The day profile is concentrated at month-end: last day +16.2 bps, N−1 +9.9, then noise
  (−2.6 to +9.1). The placebo (2 days ending N−10) is +12.2 bps, t = 1.76.
- H2's failure is less damaging than planned. Treasury *notes* (2y, 5y, 7y) are auctioned
  every month, so the index extends every month, and the refunding-month contrast was a
  weak dose-response prediction to begin with.

**OOS (2025-01 → 2026-08, 20 complete months), the single look: NOT CONFIRMED.** Mean
−2.4 bps, t = −0.13, hit 40%, net SR −0.44 (SE ≈ ±0.77). An earlier run wrongly counted
the incomplete September 2026 as a month-end. That was fixed before this was recorded.

**Verdict:** not adopted. **Both forced-flow studies show the same arc:** strong in
2003–13, fading in 2014–24, absent in 2025–26. Well-known flows with huge capacity are
still real flows, but other traders now pre-position for them, and the price impact has
moved earlier or vanished.

