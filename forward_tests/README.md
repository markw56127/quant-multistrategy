# forward_tests — pre-registered, strictly forward signal tests

**Pre-registration FIXED 2026-10-08, before any forward return in the evaluation window
exists.** Only data collected *after* this date is evaluated. There is no historical
backtest to overfit: the out-of-sample period is created by waiting. This is the cleanest
test available to this repo, and it uses data almost nobody has for free (the banked
estimates and option chains in `../options_data`).

## The family (three signals, one Holm correction)

| id | signal | universe | predicted | mechanism | reference |
|---|---|---|---|---|---|
| **F1** | **EPS revision**: for consensus EPS periods `0y` (current fiscal year) and `+1y`, rev = (current − 30 days ago) / \|30 days ago\| from the `eps_trend` snapshot; signal = mean of the two cross-sectionally winsorized (2/98%) z-scores | S&P 500 members in that day's estimates file | **+** | Analysts revise gradually and investors underreact to revisions, so news diffuses slowly into prices | Chan, Jegadeesh & Lakonishok 1996; Stickel 1991 |
| **F2** | **Option skew**: 30-day SVI implied vol at k = −σ_ATM√T minus at k = +σ_ATM√T (±1 standard deviation of the expected move, ≈ 16-delta put vs call), from `../vol_surface`; regular-session captures only *(amended 2026-10-08, see below)* | the 50 banked single stocks | **−** | Informed traders buy downside protection before bad news; steep put skew carries negative private information | Xing, Zhang & Zhao 2010 |
| **F3** | **Implied minus realized vol**: 30-day ATM SVI vol minus the annualized realized vol of the prior 21 daily close-to-close returns | the 50 stocks | **−** | Option prices that run rich to realized volatility signal demand ahead of bad news, and the volatility risk premium is larger for riskier names | Bali & Hovakimian 2009 |

## Amendment 1 (2026-10-08, before the first signal date and before any forward data exists)

- **F2's moneyness changed** from fixed k = ±0.10 to volatility-scaled k = ±σ_ATM√T.
  *Reason:* a measurement flaw found while validating on 2026-10-08. Fixed ±10% often lies
  outside the quoted strike range (30-day index calls at +10% are worth less than $0.05).
  SVI then extrapolates, which gave AT&T a "skew" of 95 vol points. Vol-scaled moneyness
  is comparable across names and stays inside the quotes. Only signal *values* were
  inspected, never forward returns, and no evaluation-window data existed.
- **Surface rules made explicit:** summary values come only from inside each expiry's
  quoted k-range (otherwise NaN), and **only regular-session captures are used**.
  After-hours chains (2026-09-30, 2026-10-06) have stale or one-sided quotes.
- **Known coverage:** about 80% of stock-days have a valid F2 (coarse strikes leave the
  rest NaN), so F2 has roughly 40 names per week.

## Protocol

- **Signal dates:** each **Wednesday**, using that day's banked snapshot (estimates or
  chains). If Wednesday is missing, use Thursday; if both are missing, the week is
  skipped and logged. The first signal date is **2026-10-14**. Nothing earlier is
  evaluated; earlier days serve only as computation sanity checks.
- **Forward return:** from the signal day's close to the close 5 trading days later,
  using split- and dividend-adjusted closes (yfinance; re-downloadable at evaluation
  time). Weeks therefore do not overlap.
- **Primary statistic:** the weekly cross-sectional Spearman IC between the signal and
  the forward return. Test: mean IC, Newey–West (2 lags) t, one-sided in the predicted
  direction.
- **Family correction:** Holm–Bonferroni over {F1, F2, F3} at 5% familywise error.
- **Pass (each signal):**
  1. it survives Holm, **and**
  2. its weekly long-short book (top vs bottom quintile, equal-weight, dollar-neutral,
     signed by prediction) has net Sharpe > 0 after 10 bps per unit of one-way turnover.
- **Evaluation dates:**
  - **Interim at 26 weeks (~2027-04-14):** descriptive only, no decision. An interim
    look must not change any definition.
  - **Primary at 52 weeks (~2027-10-13).** The verdict is made there, once.
- **A pass leads to paper trading** (`../paper_trading`) as a further forward test before
  any real capital. It is not an adoption on its own.

## Power, stated up front

52 weekly ICs is not much. With a true mean IC of 0.03 and a weekly IC standard deviation
of about 0.10 (typical for ~500 names), the expected t is about 2.2, so F1 is roughly a
coin flip to detect a real effect. With 50 names, F2 and F3 have IC noise about 3× larger,
and **can only detect large effects** in a year. A null at 52 weeks is therefore "not
detected," not "absent." The protocol may be extended to 104 weeks by a decision written
down **before** the 52-week look. Not after.

## Files

| file | role |
|---|---|
| `signals.py` | computes F1–F3 for a date from the banked data (point-in-time: only that day's files) |
| `evaluate.py` | *(written at the interim date)* forward returns, ICs, Holm, long-short books |
| `signals/{date}.csv` | signal values per Wednesday, written by `signals.py` |
