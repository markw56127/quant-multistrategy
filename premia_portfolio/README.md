# premia_portfolio — harvesting risk premia efficiently (Track A)

**Status:** pre-registration FIXED 2026-09-26, before any portfolio was computed.
**Dev results (2026-09-26):** A1 FAIL, A2 both claims FAIL, A3 Itô validated (claim failed on a
grid I set too narrow), A3b mixed. OOS untouched. See Results.

## Why this exists

The alpha hunt (`../alpha_research`, `../flow_events`) asks whether we can beat other
traders. This asks a different question with a much more reliable answer. **Risk premia
exist**: equity holders are paid for bearing equity risk, and bond holders for bearing
duration. **How efficiently can they be collected?** Here the course material
(`~/Desktop/portfolio-model`: efficient frontier, single-index and constant-correlation
models, no-short-sales optimization) and Itô calculus are the main tools, not a side
topic. Success is measured in volatility, drawdown and compounded growth.

## Universe and data

Nine liquid ETFs (total-return adjusted closes, yfinance), monthly rebalancing:
**SPY** (US equity), **EFA** (developed ex-US), **EEM** (emerging), **IEF** (7–10y
Treasury), **TLT** (20y+ Treasury), **LQD** (IG credit), **GLD** (gold), **DBC**
(commodities), **VNQ** (REITs). The risk-free rate is the 13-week T-bill (^IRX). The
common history starts 2006-03. Estimation uses trailing data only. **Dev:** portfolios
formed 2009-03 → 2024-12. **OOS:** 2025-01 onward, untouched until the dev results are
written up. Costs are 5 bps per unit of one-way turnover.

## A1: covariance estimators (course topics 4–6)

This is the course's central claim, tested at scale: structured covariance models beat
the sample covariance out of sample, because a sample covariance fit on few observations
(Project 1: T/N ≈ 2) is mostly noise.

Estimators, all on the trailing 36 monthly returns (T/N = 4), plus one daily reference:

| id | estimator |
|---|---|
| `sample` | sample covariance |
| `single_index` | single-index model, index = SPY (course Project 4) |
| `const_corr` | constant-correlation model (course Project 6) |
| `ledoit_wolf` | Ledoit–Wolf shrinkage |
| `daily_1y` | *reference:* sample covariance of the trailing 252 daily returns, × 21 |

- **Test portfolio:** the **long-only global minimum-variance (GMV)** portfolio, the no
  short-sales frontier from course Project 6. Its weights are the purest function of Σ.
- **Metric:** realized OOS volatility of each estimator's GMV (lower = better risk model),
  plus the bias statistic std(r / σ̂) (ideal 1).
- **A1 pass:** at least 2 of the 3 structured estimators (`single_index`, `const_corr`,
  `ledoit_wolf`) give lower realized GMV volatility than `sample`, each with the 90%
  block-bootstrap interval of the variance ratio below 1.

## A2: weighting schemes

Every scheme uses the **pre-set** `daily_1y` covariance, so that A2 does not depend on
A1's winner (that would be selection).

| id | scheme |
|---|---|
| `sixty_forty` | 60% SPY / 40% IEF, the conventional benchmark |
| `equal` | 1/N |
| `inv_vol` | weights ∝ 1/σᵢ |
| `risk_parity` | equal risk contribution (ERC) |
| `gmv` | long-only minimum variance |
| `max_sharpe` | long-only tangency with trailing 36-month **sample means**. This is the course's optimal portfolio, and the textbook prediction is that estimation error in the means makes it the worst performer OOS |

- **A2 claims:**
  - **(a)** `risk_parity` has higher dev Sharpe than `sixty_forty`.
  - **(b)** `max_sharpe` has the **lowest** dev Sharpe of the six.
- Both are reported with 90% block-bootstrap intervals on the Sharpe differences. They
  are claims about the theory, not adoption bars.

## A3: sizing, via Itô and Kelly

For a portfolio with excess return μ and volatility σ at leverage L, Itô's lemma gives
long-run log growth

    g(L) = r + L·μ − ½·L²·σ²,

maximized at the Kelly leverage L* = μ/σ². Growth turns **negative** beyond ≈ 2L*.

- **A3a (in-sample check of Itô):** for `risk_parity`, compute realized dev log growth
  across L ∈ {0.5, 1, 1.5, …, 6}. Borrowing costs the T-bill + 50 bps. Compare with
  g(L) from the realized μ and σ. **Claim:** the Itô curve tracks the realized growth
  curve, and the realized optimum sits within ±1 of L*.
- **A3b (causal sizing):** the leverage each month uses trailing 36-month estimates only.
  - full Kelly (L = μ̂/σ̂², capped at 8)
  - half Kelly
  - fixed L = 1
  - **vol targeting** (L = 10% / σ̂, which needs no estimate of μ)

  **Claim:** half Kelly beats full Kelly on realized growth *and* drawdown, and vol
  targeting beats both on Sharpe, because μ̂ is the noisy input. That is the
  estimation-error lesson again.

## Honesty notes

- This track **will not produce "alpha."** Its output is a well-built premia portfolio,
  plus measured evidence of which course methods matter out of sample.
- Most of these ETFs are 2004–06 vintages, so dev covers one bond bull market and one
  bond crash (2022). Risk parity's bond-heavy tilt is exposed to exactly that.

## Results (dev: 190 holding months, 2009-03 → 2024-12; `python run.py`, ~9 s)

### A1: covariance estimators. FAIL (0 of 3 structured beat sample)

| estimator | realized GMV vol | bias std(r/σ̂) | variance ratio vs sample, 90% CI |
|---|---|---|---|
| **sample** | **5.15%** | 1.10 | 1 |
| single_index | 6.36% | **1.67** | 1.53 [1.43, 1.73] |
| const_corr | 6.35% | 1.17 | 1.52 [1.37, 1.76] |
| ledoit_wolf | 5.86% | 1.00 | 1.30 [1.22, 1.47] |
| daily_1y | 5.40% | 1.06 | 1.10 [0.98, 1.22] |

**Why the course's result reversed here: structure must match the data.** The single-index
and constant-correlation models assume one kind of co-movement, which suits a homogeneous
30-stock universe (the course projects). This universe is **block-structured**: equities
correlate with each other at 0.75–0.88, Treasuries at 0.92, and the two blocks at ≈ −0.05.
In the last window, the IEF–TLT correlation is 0.95 in the sample, but **0.46 under single
index and 0.55 under constant correlation**. The GMV then treats two near-duplicate bond
funds as diversifying, overweights both, and its realized risk comes out 50% higher than
forecast (bias 1.67). Ledoit–Wolf shrinks toward a scaled identity, which ignores the 5×
spread in asset volatilities. At T/N = 4 with nine assets, the sample covariance is not yet
noise-dominated. **The course's multigroup model (topic 6) is designed for exactly this
block structure.** It was not in the pre-registered set. It is the natural pre-registered
follow-up.

### A2: weighting schemes. Both claims FAIL

| scheme | excess ret | vol | Sharpe | max DD | turnover/yr |
|---|---|---|---|---|---|
| **sixty_forty** | +9.5% | 9.1% | **+1.04** | −20.5% | 0.27 |
| equal | +6.5% | 9.6% | +0.68 | −19.4% | 0.37 |
| inv_vol | +4.7% | 7.8% | +0.60 | −19.8% | 0.44 |
| risk_parity | +4.0% | 7.3% | +0.55 | −19.0% | 0.52 |
| max_sharpe | +4.0% | 7.5% | +0.54 | −21.6% | 3.01 |
| gmv | +2.5% | 5.4% | **+0.46** | −17.2% | 0.97 |

- (a) Risk parity − 60/40 Sharpe = **−0.50**, 90% CI [−0.75, −0.20]. The claim fails, clearly.
- (b) The worst scheme is GMV, not max_sharpe. max_sharpe paid 3.0×/yr turnover and still
  matched risk parity.

**Why:** 2009–24 was a single-winner period. US equities (SPY) beat developed ex-US,
emerging markets, commodities and REITs by wide margins, and 60/40 is concentrated in
that winner. Every diversifying scheme spread weight into assets that lagged. This is one
16-year path. It shows that diversification **cost** return in this sample. It does not
show that 60/40 is structurally superior (compare 2000–09, when US equities went nowhere).

### A3a: Itô's growth formula. VALIDATED (the pre-registered check failed on its grid)

Risk parity, realized dev log growth vs Itô's g(L) = r + Lμ − ½L²σ² − borrow cost:

| L | realized g | Itô g(L) | max DD |
|---|---|---|---|
| 1 | +4.84% | +4.84% | −19% |
| 2 | +7.51% | +7.53% | −36% |
| 4 | +11.19% | +11.30% | −66% |
| 6 | +12.49% | +12.92% | **−84%** |

**Itô's formula predicts realized compounded growth to within 0.43%/yr at every leverage
up to 6×.** The pre-registered check ("realized optimum within ±1 of L\*") failed only
because I capped the grid at L = 6 while L\* = μ/σ² = 7.44. The realized curve is still
rising at 6, consistent with L\* ≈ 7.4. That was a design error in the pre-registration,
not a failure of the theory.

**The practical lesson:** the "growth-optimal" leverage in hindsight is about 7×, and at
6× the path already has an **84% drawdown**. Maximizing long-run growth and surviving
the path are different objectives.

### A3b: causal sizing (trailing estimates only). Mixed

| sizing | mean L | log growth | vol | Sharpe | max DD |
|---|---|---|---|---|---|
| fixed L = 1 | 1.00 | +4.84% | 7.3% | **+0.55** | −19.0% |
| full Kelly | 5.77 (48% of months at the 8× cap) | **+10.19%** | 42.5% | +0.44 | **−73.2%** |
| half Kelly | 4.38 | +6.24% | 34.1% | +0.33 | −61.2% |
| vol target 10% | 1.59 | +5.71% | 11.0% | +0.48 | −24.5% |

- Half Kelly beats full Kelly on drawdown (✓) but **not** on growth (✗). In this sample,
  μ̂ was not overestimated enough for full Kelly to lose.
- Vol targeting has the best Sharpe of the three levered rules (✓), but it is still below
  unlevered (0.55), because leverage costs the borrow spread.
- Full Kelly compounded fastest *and* suffered a −73% drawdown, which nobody could hold
  through in practice.

### What Track A establishes

1. **Structured covariance helps only when the structure is right.** The course methods
   matter, but the choice of model matters more than the choice to shrink.
2. **On this path, simple beat optimized** (60/40 > risk parity > max Sharpe ≈ GMV). The
   out-of-sample damage from estimation error is real (max_sharpe: 3× turnover, no gain).
   But this period's dominant fact was one asset class winning.
3. **Itô's growth formula is empirically exact enough to size with**, and it says the
   growth-optimal leverage is psychologically and practically untenable. That is the
   case for fractional Kelly and vol targeting, stated quantitatively.


## A4: multigroup covariance (follow-up, pre-registered 2026-09-26)

**Why this is a follow-up and how it stays honest.** A1 showed the single-index and
constant-correlation models fail on this universe because it is block-structured. The
course's **multigroup model** (topic 6) is designed for exactly that. It was chosen
*after* seeing the dev results, so **a dev-period test is not blind**, and dev is reported
but carries no bar. The bar sits on two samples no one in this repo has examined:

1. **Backcast (primary):** 1996–2008, using Vanguard index funds as proxies for the same
   blocks. Nobody in this repo had looked at this period's returns when this was written.
2. **OOS:** 2025-01 → 2026-08 on the ETFs (20 months, low power, secondary).

**Estimator.** Groups are fixed by asset class. The within-group correlation is the mean
pairwise correlation inside each group; the between-group correlation is the mean
correlation across each pair of groups. Σᵢⱼ = ρ̄_{g(i)g(j)} σᵢ σⱼ, estimated on the trailing
36 monthly returns like the other A1 estimators.

| group | ETFs (dev, OOS) | backcast funds |
|---|---|---|
| equity | SPY, EFA, EEM, VNQ | VFINX, VGTSX, VEIEX, VGSIX |
| treasury | IEF, TLT | VFITX, VUSTX |
| credit | LQD | VFICX |
| real | GLD, DBC | VGPMX (precious-metals equity; no commodity fund exists that early) |

**Test.** Same as A1: the long-only GMV built from each estimator; the variance ratio of
realized GMV returns, multigroup vs sample, with a 90% block-bootstrap CI.
- **PASS:** the variance ratio is < 1 in the backcast **and** < 1 in OOS.
- **STRONG PASS:** additionally, the backcast CI upper bound is < 1.
- **Also reported (no bar):** single_index and const_corr in the backcast. That checks
  whether A1's failure replicates on independent data. Also dev numbers (not blind), and
  bias statistics.

### A4 result: FAIL (`python multigroup.py`)

Variance ratio of the realized long-only GMV vs the sample covariance (90% block-bootstrap CI):

| sample | multigroup | single_index | const_corr |
|---|---|---|---|
| **backcast 1999-06 → 2008-12** (115 months, primary) | **1.127** [0.987, 1.203] | 1.149 [0.896, 1.321] | 1.248 [1.014, 1.739] |
| **OOS 2025-01 → 2026-08** (20 months) | **1.109** [0.797, 1.380] | 1.066 [0.790, 1.806] | 0.586 [0.425, 1.058] |
| dev 2009–24 (not blind) | 1.177 [1.116, 1.276] | 1.528 | 1.524 |

- **Multigroup fails its bar.** It is worse than sample in both clean samples. It *is* much
  better than single-index and constant-correlation on dev (1.18 vs 1.53), so modeling
  the blocks fixes most of A1's damage. It just does not beat doing nothing.
- **A1's failure replicates on independent data.** In the backcast decade, every structured
  estimator is worse than sample. The OOS const_corr figure (0.59) rests on 20 months with
  a CI reaching 1.06, and is not interpretable.
- **The likely explanation is known theory.** Jagannathan & Ma (2003): **imposing no short
  sales is equivalent to shrinking the covariance matrix.** The long-only GMV is already
  regularized, so a structured model adds bias without removing much noise. Structured
  models pay when N is large relative to T and short sales are allowed. That was the
  course setting (30 stocks, 59 months), and it is the 450-stock risk model in
  `../dynamic_trading`, where a PCA factor model is unavoidable.

**Track A conclusion:** for a small multi-asset universe held long-only, use the sample
(or daily 1-year) covariance. Choose the structured-versus-sample question by N/T and the
short-sale constraint, not by default. The course methods remain the right tools for
large cross-sections.
