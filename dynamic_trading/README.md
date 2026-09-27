# dynamic_trading — model-based trading control (Gârleanu–Pedersen → APG)

> **SPLIT-LOOKAHEAD CORRECTION (2026-09-26, later the same day):** everything here was
> built on `panel_presplitfix.parquet`, the panel whose market caps leaked future splits
> ([../SPLIT_FINDING.md](../SPLIT_FINDING.md)). The *method* findings stand as method
> findings: the surrogate's alpha is overstated, the trust gap tracks turnover, and
> gradients through a wrong value surface fail. The value premium they were built
> around, however, is not there in dev once corrected. **M5's dev pass is void**, and its
> OOS look should not be taken. The scripts now read the corrected panel, so rerunning
> them will not reproduce the numbers below. Point `identify.PANEL` at
> `panel_presplitfix.parquet` to reproduce them.

**Status (2026-09-26):** M1–M6 implemented. **Every surrogate-optimized
controller FAILED its pre-registered bar** at $100M: M2 LQ 0.29, M4 APG 0.32,
M6 neural MBRL 0.36, against the quintile book's 0.54. **M5, walk-forward
pessimistic selection scored on history, PASSED dev** with net 1.23 vs 0.75 on
2020-01 → 2024-11. That pass is contaminated: it selected the grid corner M3 had
already shown wins. It earns the single OOS look, **which has NOT been taken**. M3 located the cause. The surrogate overstates the alpha of
optimized portfolios by 1.2–2.5× and understates their risk by 1.3–1.6×, and
the APG policy's simulated Sharpe was 2.89 against 0.32 realized. The binding
defect is the surrogate's alpha model, not its cost model.

## Why this exists

This sleeve takes the method from `../../meq-mpo` and `../../vlasov-poisson`:
identify a cheap model that can be solved exactly, certify the optimum on it,
then measure where it stops being true. The dynamical system here is **not
prices**. A differential-equation model of prices would put all its weight on a
drift term that can't be identified, and any gradient taken through it would
follow estimation error. The controlled state is **our own holdings**. The market
enters only through signals whose dynamics *can* be estimated:

```
signals   f_{t+1} = A f_t + eps        A = I − Phi   (discrete-time OU)
returns   r_{t+1} = B f_t + u,   Cov(u) = Sigma
holdings  x_t     = x_{t−1} + dx_t
cost      TC_t    = ½ dx_t' Lambda dx_t
```

With quadratic cost this is Gârleanu & Pedersen (2013, *J. Finance*). Its optimal
policy has a closed form: trade part of the way toward an aim portfolio that
blends current and future Markowitz portfolios, weighted by how slowly each
signal decays. This is the trading counterpart of MEQ's identified linear model
plus LQR.

| MEQ | here |
|---|---|
| M1 identify A, B, C by finite differences | M1 identify Phi, B, Sigma, Lambda (this file) |
| M2 LQR scored on MEQ | GP closed form, replayed on historical returns with the repo's real costs |
| M3 trust gap vs controller aggressiveness | predicted vs realised net P&L across the cost-aversion setting |
| M4/APG for the non-quadratic Degrave reward | APG through a differentiable simulator for the costs LQ can't represent: linear 10 bps, √-impact, long-only and leverage limits |

## M1 — identification (dev only, `python identify.py`, ~30 s)

**Leakage rule:** a row counts as dev only if its forward-return window **ends**
before 2025-01-01. This excludes the 2024-12-06 rebalance, whose window ends
2025-01-08. The result is 107 rebalances, 2016-01 → 2024-11. No price dated 2025
or later is read. Output: `results/m1_identification.json`.

### (a) Signal dynamics — well identified, stable

Cross-sectional slope of z_{t+h} on z_t, Fama–MacBeth-averaged:

| | AR(1) | half-life | fits one OU? | 2016-19 / 2020-24 AR(1) |
|---|---|---|---|---|
| **value** | 0.987 | **51 mo** | **yes**: max\|ACF − a₁ʰ\| = 0.004 over 12 lags | 0.990 / 0.984 |
| **quality** | 0.939 | 11 mo (misleading) | **no**: misfit 0.163 at h=12 | 0.940 / 0.938 |

Quality is a **two-component** process. About 30% of its variance decays fast
(a=0.81, half-life ≈ 3 mo) and about 70% is as slow as value (a=0.988). In GP
terms these are two signals with different Phi. The VAR(1) cross-terms are zero
(|coef| ≤ 0.001), so Phi is diagonal.

### (b) Return decay — NOT identified; take decay from (a)

These are Fama–MacBeth slopes of the month-(t+h) return on f_t, with value and
quality regressed jointly, in % per month per z:

```
h         0      1      2      3      4      5      6      7      8      9     10     11
value  +0.194 +0.190 +0.148 +0.143 +0.135 +0.132 +0.143 +0.146 +0.133 +0.135 +0.128 +0.096
quality+0.067 +0.059 +0.052 +0.031 +0.023 +0.031 +0.014 +0.019 +0.004 +0.006 +0.031 +0.027
```

GP implies β_h = β₀·aʰ with the same *a* as in (a). That is the model's internal
consistency test. For value, the fitted return decay is b = 0.957, but the 90%
block-bootstrap interval is **[0.77, 1.13]**, and P(b ≥ 0.987) = 0.32. The data
is consistent with the model *and* with nearly anything else. Quality is wider
still: [0.49, 1.30].

(An earlier pass reported b = 0.957 ± 0.008 and read it as a 4σ inconsistency.
That was curve_fit rescaling its covariance by a weighted SSE of 0.26 while
treating 12 correlated horizons as independent. Corrected before any write-up.)

**Decision for M2:** take the **decay from signal persistence**, which is
identified to ±0.002. Take only the **level β₀ from returns**:

| | β₀ (%/mo per z) | t | 12-mo cumulative | 2016-19 / 2020-24 β₀ |
|---|---|---|---|---|
| value | +0.18 | ≈2.1 | +1.72% ± 0.69 | +0.15 / +0.25 |
| quality | +0.07 | ≈1.3 | +0.37% ± 0.34 | +0.08 / +0.05 (12-mo sum −0.16%) |

This is the VP lesson applied to trading. Identify the model from the part of
the system that is measured precisely (how signals evolve), not from the noisy
outcome (returns). GP will mostly trade value. That follows from β, not from a
choice made here.

### (c) Risk model — calibrated on the book, flattering on the optimized portfolio

PCA-K factors plus a diagonal residual, fit on trailing 252 daily returns. The
test is next-month realized P&L over predicted σ; bias = std(z), ideal 1, 95%
band ±0.13 at n=107.

| K | L/S quintile book | **mean-variance aim** | random dollar-neutral |
|---|---|---|---|
| 1 | 1.36 | 1.36 | 1.00 |
| 5 | 1.05 | 1.22 | 1.01 |
| 10 | 1.01 | **1.17** | 1.01 |
| 20 | 1.00 | **1.23** | 1.01 |

The rule, set in the code before it ran, picks K as the argmin of mean QLIKE over
the two held portfolios. It selects **K\* = 20**, a near-tie with K = 10 (QLIKE
−9.855 vs −9.842 on the aim portfolio). **Every K underestimates the optimized
portfolio's risk by 17–26%, outside the band.** The GP aim portfolio *is* an
optimized portfolio, so this is the first measured trust gap. M2 must either
inflate the aim portfolio's σ̂ by the measured factor or shrink toward the
quintile book.

### (d) Costs — the LQ model can't represent most of the cost at this repo's scale

Measured on the 6,592 trades the canonical value+quality book actually made. The
costs are 10 bps linear (the sleeve's convention) plus square-root impact
(`factor_research/capacity.py`, c=1):

| AUM | linear bps/yr | √-impact bps/yr | **linear share** |
|---|---|---|---|
| $10M | 18.8 | 7.4 | **72%** |
| $100M | 18.8 | 23.4 | **45%** |
| $1B | 18.8 | 74.0 | 20% |

- **Linear cost is out of reach for LQ.** Under linear cost the optimal policy
  is a no-trade band, not partial adjustment. Below about $100M it is the larger
  part of the cost. GP is misspecified there by construction, and that is where
  M4/APG has a reason to exist.
- **GP's canonical Λ ∝ Σ is not supported.** The per-name linear-impact
  coefficient σ/ADV has a Spearman correlation of only +0.26 with σ², and it
  spreads 10.7× from p10 to p90 across names. M2 should use a diagonal Kyle-type
  Λ_ii ∝ σ_i/ADV_i.
- **The quadratic fits the *current* book's trades** once matched on total
  cost: ratio 1.00, 0% of cost off by more than 2×. The reason is that the
  quintile book's cost comes almost entirely from full entries and exits of
  similar size. That says nothing about GP's own trades. Per trade,
  quad/√ = √(q/q_ref), so the quadratic **understates the cost of small partial
  trades**, which is exactly the kind of trade GP makes. This is a known
  direction of optimism, and M3 should measure it.

## Decision log

**2026-09-26: the scoring AUM is $100M; $10M is reported but not scored.**
The choice decides what M2 measures, because the cost mix flips between the two
(M1 (d)):

- **$10M.** Linear 10 bps is 72% of cost (18.8 bps/yr linear vs 7.4 √-impact).
  LQ has no linear term, so here GP optimizes against a cost model that has
  most of the real cost missing. A loss at $10M would mostly measure GP's
  misspecification, not the idea of trading on signal decay. This is the regime
  where a no-trade band (M4/APG, or the analytic band) is the right tool.
- **$100M.** Linear 45%, √-impact 55% (18.8 vs 23.4 bps/yr). The quadratic
  cost GP optimizes is the larger part of the true cost, so this is the fair
  test of the model on its own terms. The 45% linear part is still unmodeled,
  and that works against GP, so a pass here is conservative.
- At $1B, impact is 80% of cost. That would flatter GP, and the repo has never
  claimed that scale.

$10M is printed alongside every M2 result as a diagnostic. It carries no bar.

## M2 — pre-registration (FIXED 2026-09-26, before any M2 code was run)

- **Controller:** discrete-time GP with general Λ, solved exactly per rebalance.
  Substitute y = Λ^½x; the Riccati equation is then diagonal in the eigenbasis of
  Λ^−½ΣΛ^−½ and has a scalar closed-form root. Three alpha streams:
  - value: decay a_v, α = β_v·v
  - quality-fast: a_f, α = w·β_q·q
  - quality-slow: a_s, α = (1−w)·β_q·q

  (w, a_f, a_s) is the two-OU fit of quality's autocorrelation. The w/(1−w)
  split is E[component | q_t], the certainty-equivalent split given only the
  current score. Discount δ = 0.99/month.
- **Parameters, all causal:** decays and w come from the expanding-window
  autocorrelation. β₀ comes from the expanding-window Fama–MacBeth h=0 slope,
  using only return windows that ended by the rebalance date. Burn-in is 36
  months, so the dev scoring window is about 2019-01 → 2024-11. The same dates
  apply to both books.
- **Σ:** PCA-20 plus diagonal residual, trailing 252 days, ×21 for monthly.
  *Dropped from the draft:* "inflate σ̂ by the aim bias". A uniform Σ scale is
  identical to rescaling γ, which the vol-match sets, so it was a no-op. Realized
  vol is reported for both books instead.
- **Λ:** diagonal. Λ_ii = 2cσ_i√A / √(ADV_i·0.00614), which is the M1
  per-name quadratic matched to √-impact on total cost. q_ref/A = 0.00614 is a
  structural constant of the quintile book's trade sizes. It does not come from
  returns.
- **γ:** one constant, set by bisection so that GP's mean ex-ante vol equals the
  quintile book's mean ex-ante vol over the scoring window. This uses the risk
  model only, never returns.
- **Comparator:** the canonical value+quality top/bottom-20% book. Both books
  are held from rebalance to rebalance with no drift (the `capacity.py`
  convention), both start from zero, and names that leave the universe are
  liquidated at cost.
- **Costs used for scoring (both books):** 10 bps × |Δx|, plus
  cσ_i|Δx_i|^1.5·√(A/ADV_i), plus 1%/yr borrow on shorts. AUM = **$100M**.
- **Bar:** adopt only if (i) dev net Sharpe ≥ canonical + 0.10 **and** (ii) dev
  gross Sharpe ≥ canonical − 0.05. Condition (ii) catches the signal-dilution
  failure that sank the turnover buffer.
- **OOS:** only if dev passes. One look, same code with the window continued
  into 2025-26, which must beat canonical net. If dev fails, the OOS window is
  not touched.

## M2 — result: FAIL (2026-09-26, `python m2_controller.py`, ~30 s)

The controller is verified before it is scored. `test_controller.py` checks the
closed form against a stacked 600-step finite-horizon QP that shares none of its
derivation. It matches to 1e-6 across 3 seeds × 3 values of γ, from zero and
from arbitrary holdings, and reduces to Markowitz as Λ→0.

Causal parameters drift only mildly over the window: a_v 0.991→0.987,
w 0.34→0.25, β_v 0.21→0.20%, β_q 0.12→0.06%. The scored window is 2019-01 →
2024-11, 71 months, same dates for both books, both vol-matched ex-ante to
0.94%/mo.

| $100M (scored) | gross SR | **net SR** | real vol | turn/mo | GMV | linear | impact | borrow (bps/yr) |
|---|---|---|---|---|---|---|---|---|
| GP | +0.81 | **+0.29** | 5.06% | 0.55 | 2.96 | 66 | 49 | **148** |
| quintile | +0.82 | **+0.54** | 3.50% | 0.18 | 1.00 | 22 | 27 | 50 |

Net SR difference −0.25, 90% block-bootstrap [−0.84, +0.43]. **Bar (i) FAIL,
(ii) pass.** At $10M (diagnostic) it is −0.33: GP 0.26 vs quintile 0.59.

**Why it failed, as measured, not as an excuse to re-tune:**

1. **Gross Sharpe is equal (0.81 vs 0.82).** GP's signal weighting and decay
   awareness cost nothing and gained nothing before costs.
2. **Leverage is the mechanism.** At equal ex-ante vol, GP holds 3× the gross,
   because Σ says hedged, optimized portfolios are low-risk. Two things follow:
   - **Borrow** (148 bps/yr) and **linear cost** (66 bps/yr), neither of which
     exists in the LQ objective, scale with that gross. Together they are
     about 2.1% of return per year. Those are the two cost types M1 (d)
     flagged as out of LQ's reach.
   - **Risk is underestimated.** GP's realized vol is 5.06% against a
     3.26%/yr ex-ante, a ratio of 1.55. The quintile book's ratio is 1.07. This
     is M1 (c)'s optimized-portfolio bias again, larger here because GP is
     more optimized than the pure Markowitz test portfolio.
3. **Relative turnover did not fall:** turnover/GMV is 0.19 vs 0.18. At the
   identified Λ, the quadratic cost is too small relative to γΣ to slow
   trading, so the "trade patiently on slow signals" mechanism barely engaged.

Everything that hurt GP is either a cost missing from the model's objective or
the model's risk estimate being wrong in exactly the directions it optimizes
into. That is the case for M4, which optimizes against the true cost
function, and it is what M3 measures.

## M3 — trust gap (2026-09-26, `python m3_trust_gap.py`, ~20 s). Descriptive, nothing adopted

The sweep covers 30 GP controllers: risk aversion γ = g·γ\* for g ∈ {¼ … 8}, and
cost aversion Λ → κΛ for κ ∈ {¼ … 64}. All run at $100M on the M2 scoring
window. `results/m3_trust_gap.csv` has the full table.

| model-vs-truth ratio | quintile book | GP range | moves with |
|---|---|---|---|
| alpha: realized gross / model x'μ | 0.96 | **0.39 – 0.86** | rises with κ (ρ = +0.86) |
| risk: realized vol / ex-ante | 1.07 | **1.30 – 1.56** | falls with κ (ρ = −0.86) |
| cost: true (10 bps + √) / quadratic ½Δx'ΛΔx | 1.8 | **0.9 – 23** | worst for slow, small-trade controllers |

- **The VP finding transfers, more strongly.** The model's shortfall
  (predicted net − realized net) has rank correlation **+0.99 with turnover**
  and **+0.97 with GMV** across controllers; VP's figure for max|H| was +0.67.
  Aggressiveness predicts transfer failure.
- **The optimizer overestimates its own alpha by 1.2–2.5×.** The quintile book
  realizes 96% of model alpha. GP realizes 39–86%. This is mean-variance error
  maximization, measured, and it shrinks as the controller slows.
- **Inertia is the dominant knob.** Net SR has rank correlation +0.93 with κ.
  Gross SR depends only on γ/κ, rising 0.71 → 1.45 as trading slows, so
  inertia helps *before* costs too. The likely reason is that it averages
  month-to-month signal noise, the opposite of the buffer's dilution.
- 12 of 30 controllers beat the quintile book's net SR on dev (best g=¼, κ=64:
  1.10). **That is a maximum over a grid on the scored window and is not
  adoptable.**

**Contamination notice.** M3 has shown that dev net SR rises with κ. Any later
dev test of a policy that can move κ is no longer blind. M4's decisive test is
therefore its one OOS look.

## M4 — pre-registration (FIXED 2026-09-26, before any M4 code was run)

The question: M2 lost to costs the LQ objective can't see (linear 10 bps,
borrow, √-impact). Does optimizing against the *true* cost function by
**analytic policy gradients through a differentiable simulator** of the M1 model
fix that? This is the method from `meq-mpo/meq_apg.py`.

- **Simulator (surrogate), in torch.** A fixed cross-section: the first scored
  rebalance, 2019-01-04, with its universe, PCA-20 Σ, σ_i, ADV_i and causal M1
  parameters (data up to that date only). Signals are simulated as the M1 model
  says:
  - value is a single OU process
  - quality is **two hidden OU components** (fast and slow), with only their
    sum observed
  - all are unit-variance stationary
  - returns are r = β_v v + β_q q + Σ^½ξ

  Costs are the true scoring costs, with |·| smoothed at 1e-6: 10 bps linear,
  c σ|Δx|^1.5 √(A/ADV), and 1%/yr borrow. $100M. 64 paths × 60 months, starting
  from zero holdings.
- **Policy (4 parameters), structured on the GP closed form:**
  - Σ_η = (1−η)Σ + η·diag Σ. This shrinks hedging toward diagonal, which lets
    the policy trade leverage for borrow.
  - x^GP = GP(x_prev; γ = g·γ\*, κΛ, Σ_η).
  - A no-trade band: x = x_prev + softshrink(x^GP − x_prev, b·c/(γΣ_ii)), where
    c = 10 bps. This is the myopic linear-cost band.
  - Parameters θ = (log g, log κ, b ≥ 0, η ∈ [0,1]), initialized at M2
    (1, 1, 0, 0). The policy sees only the observed quality, split w/(1−w) as in
    M2.
- **Objective in sim:** mean net monthly return − 100·(mean ex-ante vol / target
  − 1)², with target = the quintile book's ex-ante vol on 2019-01-04. Adam,
  lr 0.05, 300 steps, fixed seeds. The final θ is **frozen** and replayed on
  history with per-date causal inputs exactly as in M2.
- **Bar ($100M):** dev net SR ≥ quintile + 0.10 **and** dev gross SR ≥
  quintile − 0.05. Given the M3 contamination, a dev pass only permits the
  **one OOS look**. OOS net SR must beat the quintile book's OOS net SR for
  adoption.
- **Reported regardless:** the sim-predicted net SR vs the realized dev net SR
  for the trained policy. This is the trust gap of the model-based-RL policy
  itself, the measurement the VP chapter argues most MBRL work leaves out.

## M4 — result: FAIL (2026-09-26, `python m4_apg.py`, ~2 min)

Verified before scoring (`test_controller.py`):
- The torch policy with b→0, η→0 reproduces the numpy closed form to 1e-8.
- Autograd gradients through the full simulator (hidden OU signals, smoothed
  true costs, borrow, vol penalty) match central finite differences to 1e-4 on
  all 4 parameters.

The APG converged by iteration ~100. The trained θ: **g = 1.73, κ = 1.73**,
b = 0.015, η = 0.004.

| $100M dev, 71 mo | sim SR | gross SR | **net SR** | real vol | turn/mo | GMV | lin | imp | borrow |
|---|---|---|---|---|---|---|---|---|---|
| APG (M4) | 2.89 | +0.81 | **+0.32** | 2.89% | 0.31 | 1.69 | 37 | 21 | 85 |
| GP (M2 point) | 2.86 | +0.81 | +0.29 | 5.02% | 0.54 | 2.94 | 65 | 48 | 147 |
| quintile | — | +0.82 | **+0.54** | 3.50% | 0.18 | 1.00 | 22 | 27 | 50 |

Net SR difference APG − quintile is −0.22, 90% block-bootstrap [−0.80, +0.46].
**Bar (i) FAIL, (ii) pass.** OOS not touched.

**What the gradient learned, and why that was all it could learn:**

1. **g and κ moved together (1.73 each).** GP's policy shape depends on γ/κ,
   so the shape is *unchanged* from M2. All the APG did was scale the book down
   to hit the vol target at the calibration date. That scaling cut GMV
   2.94 → 1.69 and borrow 147 → 85 bps, which is worth +0.03 net SR. The band
   and the hedging shrink barely moved off their initial values.
2. **The trust gap is 9×: sim SR 2.89 → realized 0.32.** In the surrogate, the
   identified β is realized in full on whatever portfolio the policy holds, and
   the residuals really are Σ-distributed. Against that alpha, the true costs
   look small, so the simulator's gradient correctly says there's no point
   slowing down or de-levering.
3. **M3 had already measured the real reasons to slow down,** and none of them
   exist in the surrogate: realized alpha is 0.39–0.86 of modeled, and
   realized risk is 1.3–1.56× of modeled, both worsening with aggressiveness.
   A gradient through a model can only find what the model contains.

**Conclusion.** The costs LQ couldn't represent were a real, measured part of
M2's loss. Putting them into a differentiable simulator was necessary but not
sufficient. The binding defect is the **surrogate's alpha model**: a
cross-sectional average premium, t≈2, treated as a per-stock expected return
that the optimizer can concentrate on. This is the VP finding in trading form.
A cheap model's gradient is only as good as its value surface where the policy
goes, and an optimizer goes precisely where the value surface is most wrong.

**What a next round would need, and why it isn't run here.** The surrogate would
need alpha estimation error (β drawn from its sampling distribution plus
stock-level alpha noise, so realized/model alpha falls with concentration) and
a risk gap in optimized directions. Both must be calibrated **causally**,
because the only calibration data so far (M3) is the dev scoring window. M3 and
M4 have also already shown on dev that inertia and de-levering help. So any M5
designed now is no longer blind on dev, and only the single OOS look could
judge it.

## M5 + M6 — pre-registration (FIXED 2026-09-26, before any M5/M6 code was run)

These are the two remaining parts of the VP prescription (`vlasov-poisson/FINDINGS.md`
§15–17): use the surrogate for its gradient under an effort penalty (M6), and
choose among gradient-derived candidates pessimistically, never by surrogate
value (M5). **Both are dev-contaminated.** M3 and M4 have already shown on the
scoring window that inertia and de-levering help. A dev pass permits only the
one OOS look, which decides.

### M5 — walk-forward pessimistic selection (`m5_pessimistic.py`)

- **Pool:** 31 candidates, the 30 M3 GP controllers (g × κ grid, fixed before
  M3 ran) plus the M4 APG policy. Every candidate's path uses only causal
  inputs.
- **Re-selection:** each January from 2020 to 2024, using only months realized
  before that date. Before 2020-01 the book is the M2 controller. The scored
  window is 2020-01 → 2024-11. Switching controllers is costed, because the
  stitched book is scored like any other.
- **"Truth":** each candidate's realized net SR over the past window. Backtests
  are cheap, so every candidate is truth-evaluated. VP's 5-solve budget has no
  counterpart here. What pessimism guards against is the **noise** in a short
  truth window.
- **Correction:** ridge (α=1, standardized features) from surrogate-side features
  to past realized net SR. The features are model-predicted net SR, log GMV,
  log turnover, and **their squares**, because VP §16 found curvature essential
  for representing an interior optimum. σ̂ comes from a moving-block bootstrap
  over past months (block 6, 200 draws, ridge refit per draw).
- **Selection:** argmax(ŷ − k·σ̂), with **k = 2** (VP). k ∈ {0, 1, 4} is reported
  as a diagnostic, with no bar.
- **Bar ($100M, 2020-01 → 2024-11):** net SR ≥ quintile + 0.10 **and** gross SR ≥
  quintile − 0.05.

### M6 — model-based RL, neural policy, APG in the surrogate (`m6_mbrl.py`)

- **Surrogate:** identical to M4's (calibrated at 2019-01-04, causal).
- **Policy:** x = x_GP + Δ_NN. x_GP is the M2 closed form (g = κ = 1). Δ_NN is
  **one shared MLP applied per stock**, which makes it permutation-equivariant
  across the cross-section. Its per-stock inputs, standardized cross-sectionally,
  are x_prev, x_GP, v, q, σ, log ADV, (Σx_prev)_i and λ_i. Architecture is
  2×32 tanh. The last layer is initialized to zero, so training starts *exactly*
  at M2.
- **Objective:** mean net − 100·(vol / target − 1)² − ρ_e·mean one-way turnover.
  ρ_e is set once at initialization so the effort term equals **10%** of
  |mean sim net return| (VP §15). Adam, lr 1e-3, 300 steps, 64 paths ×
  60 months. Frozen, then replayed on history.
- **Bar ($100M, 2019-01 → 2024-11):** same as M4. Sim-predicted vs realized SR
  is reported.

## M5 — result: DEV PASS, contaminated (2026-09-26, `python m5_pessimistic.py`, ~30 s)

| $100M, 2020-01 → 2024-11, 59 mo | gross SR | **net SR** | turn/mo | GMV |
|---|---|---|---|---|
| pessimistic k = 0, 1, 2, 4 (identical) | +1.62 | **+1.23** | 0.32 | 5.79 |
| surrogate value only (argmax model-predicted SR) | +0.94 | +0.25 | 0.13 | 0.39 |
| M2 GP, fixed | +0.98 | +0.49 | 0.53 | 3.01 |
| hindsight best in pool (g=¼, κ=64) | +1.62 | +1.24 | 0.30 | 5.79 |
| quintile | +1.02 | **+0.75** | 0.17 | 1.00 |

Net SR difference (k=2) − quintile is **+0.48**, 90% block-bootstrap
[−0.29, +1.28]. **Bar (i) PASS, (ii) PASS.**

**Read it skeptically:**

1. **Pessimism never arbitrated.** Every k picked the same candidate at every
   re-selection. That candidate is the grid corner g=¼, κ=64, chosen from the
   first 12 months onward (ridge in-sample ρ 0.94–0.96). The rule's
   optimizer's-curse protection was never tested. A single controller
   dominated.
2. **It is the knob M3 revealed.** The rule is causal: each choice uses only
   past months. But the grid, fixed before M3 ran, contains a corner that M3
   then showed wins on these same months. A corner optimum also means the
   grid was too narrow, and extending it now would be post-hoc. **It will not be
   extended.**
3. **5.8× gross.** M5's pre-registration did not require vol-matching, so this
   is a legitimate Sharpe comparison with costs and borrow charged. But it is a
   highly levered book. g=¼ means a quarter of the vol-matched risk aversion.
4. **VP's other half replicates exactly.** Choosing by the surrogate's own value
   ("surrogate value only") picks g=8, κ=1 and scores 0.25. **The surrogate's
   value ranking is worse than useless, and truth-scored selection among
   surrogate-structured candidates is what works.** That is VP §17's
   prescription, in trading.
5. The mechanism is heavy inertia on a slow signal: gross SR 1.62 vs 1.02,
   before costs. It is consistent with M3. Whether it survives outside 2019–24
   is exactly what the OOS look would test.

## M6 — result: FAIL (2026-09-26, `python m6_mbrl.py`, ~6 min)

Verified first: a zero-initialized policy equals the GP closed form to 1e-12,
and the policy is permutation-equivariant across stocks (`test_controller.py`).
ρ_e = 0.00178 per unit turnover, which is 10% of the initial |sim net|.

| $100M dev, 71 mo | sim SR | gross SR | **net SR** | real vol | turn/mo | GMV | lin | imp | borrow |
|---|---|---|---|---|---|---|---|---|---|
| MBRL (M6) | 2.63 | +0.82 | **+0.36** | 3.50% | 0.41 | 1.67 | 49 | 32 | 82 |
| GP (M2) | 2.86 | +0.81 | +0.29 | 5.06% | 0.55 | 2.96 | 66 | 49 | 148 |
| quintile | — | +0.82 | **+0.54** | 3.50% | 0.18 | 1.00 | 22 | 27 | 50 |

Net SR difference −0.18, 90% block-bootstrap [−0.83, +0.39]. **Bar (i) FAIL.**
OOS not touched.

The same lesson as M4, with a far more flexible policy. Most of what it learned
was to meet the vol target, taking GMV from 2.96 to 1.67. Turnover *per unit
GMV* rose (0.25 vs 0.19), and the 10% effort penalty barely bound. The trust gap
is 7×, 2.63 → 0.36. Policy capacity is not the constraint. The surrogate's alpha
is.

## Synthesis across M1–M6

| stage | uses the surrogate for | dev net SR vs quintile | verdict |
|---|---|---|---|
| M2 | exact optimum of its value | 0.29 vs 0.54 | fail |
| M4 | gradient of its value (4-param structured policy) | 0.32 vs 0.54 | fail |
| M6 | gradient of its value (neural policy, effort penalty) | 0.36 vs 0.54 | fail |
| M5 | *structure only*: candidates ranked by **realized history**, pessimistically | 1.23 vs 0.75 | pass (contaminated) |

The VP recipe transferred in its negative form without qualification. A
surrogate whose values are wrong where the optimizer goes cannot be rescued by
better optimization, whether closed form, APG, or a neural policy. It transferred
in its positive form, surrogate-shaped candidates chosen on truth, only under
contamination. The one clean test left is the OOS look for M5.

## Run

```bash
cd dynamic_trading
python identify.py          # M1
python m2_controller.py     # M2
python m3_trust_gap.py      # M3
python m4_apg.py            # M4
python m5_pessimistic.py    # M5 (needs results/m4_apg.json)
python m6_mbrl.py           # M6
python -c "import test_controller as t; [getattr(t,f)() for f in dir(t) if f.startswith('test_')]"
```
