# Multi-Strategy Equity Research System

A survivorship-free, lookahead-audited research platform for cross-sectional and
cross-asset systematic strategies on US equities and futures. The defining feature
of this repo is not a winning strategy — it is **rigor**: every sleeve is tested on
a point-in-time universe with realistic costs, validated on data the model never
saw, and **retired with evidence when it fails**. Most of what follows is honest
negative results, written up alongside the one edge that survived.

> **What a reviewer should take from this repo:** the hard part of quant research is
> not building models — it is not fooling yourself. This project demonstrates the
> discipline that prevents it (survivorship correction, lookahead audits, true
> out-of-sample tests, purged/embargoed cross-validation) and applies it ruthlessly,
> including to its own ideas.

## Scoreboard (honest, true out-of-sample where applicable)

| Sleeve | Method | Result | Status |
|---|---|---|---|
| **`trend_model`** | Cross-asset time-series momentum (futures) | dev **0.61**, but roll-free replication: dev 0.31, **OOS −0.19 to +0.38 by data source** | **dev-robust, OOS unresolved** (2026-07 addendum) |
| `factor_model` | Value/quality/momentum/low-vol/size, sector-neutral | dev **0.16** (was 0.62: split lookahead) → **OOS +0.25** | no dev edge; OOS marginal |
| `carry_model` | Futures carry, eq+rates (KMPV via free-data proxies) | dev 0.08, **OOS −1.20** | dead (US-only carry = 2 macro bets) |
| `vix_model` | VIX term-structure switch (VRP via ETPs) | dev 0.38 / OOS −0.01; timing LOSES to static short-vol | shelved (timing subtracts value) |
| `earnings_model` | PEAD / earnings-surprise drift | **OOS Sharpe −0.69** | dead (crowding decay) |
| `statarb_model` | OU-process pairs (cointegration) | **−0.30 gross** | dead (no edge in large-cap) |
| `smallcap_factor` | Same factor engine, S&P 600 | apparent IC t=3.3 was **survivorship** | unsupported (artifact) |
| `insider_model` | Form-4 insider buying | Sharpe 0.13 / −0.01 | dead |
| `sector_rotation` | Cross-sector momentum | dev ~0, **OOS −0.67**; 3 reformulations failed pre-set bars | dead (OOS-confirmed) |
| `pinn_rl` | Physics-informed RL (Fokker-Planck + SAC) | lookahead artifact | **retired** |
| `dynamic_trading` | Model-based control on value+quality: GP LQ (M2), analytic policy gradients (M4), neural MBRL (M6), VP pessimistic selection (M5) | all built on the split-contaminated panel; surrogate-optimized M2/M4/M6 failed bars even so; M5's dev pass **void** | method results stand as method results; no dev edge underneath |

Analysis layers: **`factor_research`** (Fama-MacBeth premia, IC decay, turnover &
capacity) and **`signal_combiner`** (ridge vs XGBoost under purged/embargoed CV).

## The methodology (what makes the results trustworthy)

- **Survivorship-free, point-in-time universe** — `shared/universe_pit.py`
  reconstructs S&P 500 membership *as of each date* and keeps delisted names with
  partial history. Correcting this alone turned a spurious size factor from +187%
  (Sharpe 1.44) to a literature-consistent +23% (0.26). See
  [SURVIVORSHIP_FINDING.md](SURVIVORSHIP_FINDING.md).
- **Lookahead discipline** — signals read only data available at the decision time;
  vols and weights use strictly-prior windows. A lookahead audit retired the entire
  PINN-RL sleeve. See [LOOKAHEAD_FINDING.md](LOOKAHEAD_FINDING.md). A second audit
  (2026-09) found that market cap = split-adjusted price × as-reported EDGAR shares
  leaked **future splits** into every price multiple. It *was* the dev-period value
  premium (t 1.90 → −0.51 once corrected). Market cap now comes from
  `shared/market_cap.py`, and `factors.py` refuses to run without it. See
  [SPLIT_FINDING.md](SPLIT_FINDING.md).
- **True out-of-sample** — every config runs through 2026-06; the 2025+ window was
  never seen during development. `oos_report.py` splits dev vs OOS per sleeve. This
  is where the two-sleeve book that looked like Sharpe 0.75 revealed itself as ~0.
  See [OOS_FINDING.md](OOS_FINDING.md).
- **Purged + embargoed walk-forward CV** — `signal_combiner` (López de Prado) so a
  training label's forward return cannot overlap the test month.
- **Realistic frictions** — transaction costs, slippage, borrow fees on shorts, and
  (for futures) roll-gap defenses are modeled throughout.

## Highlights

**The edge that survived dev — then met its own audit (`trend_model`).**
Cross-asset time-series momentum localizes to equity-index and rates futures
(decided on development data only); the dev-selected book read **0.61 → 0.38**
out of sample on front-month "=F" data. Then the sleeve's own data caveat was
*tested* instead of disclosed: an identical roll-free replication on ETF excess
returns (`trend_model/validate_etf_proxy.py`) gives dev +0.31 but **OOS −0.19**
— the dev premium is real across data sources, the OOS sign is not resolvable
on free data. The claim was downgraded accordingly.
[TREND_FINDING.md](TREND_FINDING.md) (see the 2026-07 addendum).

**Killing the follow-up idea properly (`carry_model`).** Futures carry (KMPV
2018) is the textbook complement to trend — and the free-data, US-only
expression of it (ETF dividend yield − bills; Treasury term spread) failed a
pre-registered bar on its one and only run: dev 0.08, OOS −1.20, despite
delivering the promised −0.04 correlation to trend. With 8 US instruments,
"carry" degenerates into two macro bets. Anomaly real; this expression of it, no.

**Catching a fake edge (`smallcap_factor`).** Running the identical factor engine on
small-caps produced a *highly significant* cross-sectional IC (t = 3.35). It was
fake: dropping the `size` factor collapsed IC to t = 0.32, because current-constituent
small-caps condition on survival and `size` returned a survivorship-driven +3,566%.
The same toolkit later exposed a subtler one in the large-cap value premium itself
(split lookahead, [SPLIT_FINDING.md](SPLIT_FINDING.md)). The illusions are the point.

**ML done honestly (`signal_combiner`).** A linear-vs-XGBoost bake-off under
purged/embargoed CV. On the split-corrected panel every combiner is negative in dev
(L/S Sharpe −0.17 to −0.38): there is no dev signal among the five factors to
combine. The original "simplest combiner wins" result rested on the lookahead.

**The statistical layer (`factor_research`).** Fama-MacBeth factor premia with
Newey-West t-stats, IC-decay profiles, turnover / transaction-cost breakeven, and
square-root market-impact capacity analysis. On corrected market caps, no factor
has a full-sample premium above |t| = 2 except a *negative* size premium
(t = −2.67). Value is t = 0.43 (dev −0.51, OOS +4.37).

## Repository layout

```
shared/universe_pit.py     Point-in-time, survivorship-free universe + prices
factor_model/              Cross-sectional value/quality/momentum/low-vol/size
earnings_model/            PEAD earnings-drift sleeve
trend_model/               Cross-asset TSMOM (futures) — dev-robust; OOS data-dependent
carry_model/               Futures carry, eq+rates (dead — free-data expression has no edge)
vix_model/                 VIX term-structure sleeve (shelved — timing loses to static VRP)
options_data/              Daily options-chain snapshot collector (point-in-time archive)
statarb_model/             OU-process statistical-arbitrage (pairs)
smallcap_factor/           Factor engine on S&P 600 (survivorship case study)
insider_model/             Form-4 insider-buying sleeve
sector_rotation/           Cross-sector momentum sleeve
signal_combiner/           Ridge vs XGBoost combiner, purged/embargoed CV
factor_research/           Fama-MacBeth, IC decay, turnover & capacity analysis
pinn_rl/                   Physics-informed RL sleeve (RETIRED — lookahead)
dynamic_trading/           Model-based control: identified LQ model → GP → APG (failed bars)
combine_strategies.py      Risk-parity multi-sleeve combiner
oos_report.py              Development vs true-OOS report per sleeve

*_FINDING.md               Write-ups: survivorship, lookahead, OOS, trend
*/README.md                Per-sleeve methodology, results, and caveats
```

## Setup

```bash
mamba create -n trading-model python=3.11 -y && mamba activate trading-model
pip install -r requirements.txt
```

Most sleeves run standalone from their own directory (`cd factor_model && python run.py`),
optionally with `--oos-start 2025-01-01` for the out-of-sample-only view. The
point-in-time price/fundamentals caches are shared and reused across sleeves.

## A note on the PINN-RL sleeve

The repo began as a physics-informed RL system (Fokker-Planck latent density → Soft
Actor-Critic sizer). Its early results were **lookahead artifacts**; after the leaks
were fixed it shows no edge, and it is retired (`pinn_rl/`,
[LOOKAHEAD_FINDING.md](LOOKAHEAD_FINDING.md)). It remains in the repo as the first
case study in the discipline that defines the rest of the work: a model is only as
real as the test that tried to break it.
