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
| **`trend_model`** | Cross-asset time-series momentum (futures) | dev Sharpe **0.61 → OOS 0.38** | **survived OOS** — only one |
| `factor_model` | Value/quality/momentum/low-vol/size, sector-neutral | dev ~0.6 → **OOS +0.25** | marginal, degraded |
| `earnings_model` | PEAD / earnings-surprise drift | **OOS Sharpe −0.69** | dead (crowding decay) |
| `statarb_model` | OU-process pairs (cointegration) | **−0.30 gross** | dead (no edge in large-cap) |
| `smallcap_factor` | Same factor engine, S&P 600 | apparent IC t=3.3 was **survivorship** | unsupported (artifact) |
| `insider_model` | Form-4 insider buying | Sharpe 0.13 / −0.01 | dead |
| `sector_rotation` | Cross-sector momentum | Sharpe 0.03 | dead |
| `pinn_rl` | Physics-informed RL (Fokker-Planck + SAC) | lookahead artifact | **retired** |

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
  PINN-RL sleeve. See [LOOKAHEAD_FINDING.md](LOOKAHEAD_FINDING.md).
- **True out-of-sample** — every config runs through 2026-06; the 2025+ window was
  never seen during development. `oos_report.py` splits dev vs OOS per sleeve. This
  is where the two-sleeve book that looked like Sharpe 0.75 revealed itself as ~0.
  See [OOS_FINDING.md](OOS_FINDING.md).
- **Purged + embargoed walk-forward CV** — `signal_combiner` (López de Prado) so a
  training label's forward return cannot overlap the test month.
- **Realistic frictions** — transaction costs, slippage, borrow fees on shorts, and
  (for futures) roll-gap defenses are modeled throughout.

## Highlights

**The one edge that survived (`trend_model`).** Cross-asset time-series momentum is
the anomaly with a century of out-of-sample support. The premium localizes to
equity-index and rates futures (decided on development data only); the dev-selected
book held **0.61 → 0.38** out of sample — the first positive true-OOS result in the
repo. [TREND_FINDING.md](TREND_FINDING.md).

**Catching a fake edge (`smallcap_factor`).** Running the identical factor engine on
small-caps produced a *highly significant* cross-sectional IC (t = 3.35). It was
fake: dropping the `size` factor collapsed IC to t = 0.32, because current-constituent
small-caps condition on survival and `size` returned a survivorship-driven +3,566%.
The same toolkit that found a real value premium (Fama-MacBeth t ≈ 3 in
`factor_research`) exposed the illusion — that contrast is the point.

**ML done honestly (`signal_combiner`).** A linear-vs-XGBoost bake-off under
purged/embargoed CV: XGBoost does **not** beat a regularized linear model, and
learned combiners are regime-fragile vs. the naive composite. At equity-factor
signal-to-noise, the simplest combiner wins — demonstrated, not asserted.

**The statistical layer (`factor_research`).** Fama-MacBeth factor premia with
Newey-West t-stats (value is the lone robust premium), IC-decay profiles, turnover /
transaction-cost breakeven (>50 bps), and square-root market-impact capacity
analysis (~$5B AUM before Sharpe halves). The evaluation a desk runs before sizing.

## Repository layout

```
shared/universe_pit.py     Point-in-time, survivorship-free universe + prices
factor_model/              Cross-sectional value/quality/momentum/low-vol/size
earnings_model/            PEAD earnings-drift sleeve
trend_model/               Cross-asset time-series momentum (futures)  ← OOS survivor
statarb_model/             OU-process statistical-arbitrage (pairs)
smallcap_factor/           Factor engine on S&P 600 (survivorship case study)
insider_model/             Form-4 insider-buying sleeve
sector_rotation/           Cross-sector momentum sleeve
signal_combiner/           Ridge vs XGBoost combiner, purged/embargoed CV
factor_research/           Fama-MacBeth, IC decay, turnover & capacity analysis
pinn_rl/                   Physics-informed RL sleeve (RETIRED — lookahead)
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
