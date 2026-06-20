# statarb_model — OU statistical-arbitrage (pairs)  ·  SHELVED (no edge)

Mean-reversion sleeve, built to fill the one structural gap in the book: every
other sleeve (factor momentum, PEAD, cross-asset trend) is trend-following in
spirit, so the portfolio had **zero mean-reversion exposure**. Pairs trading is
also the canonical *tradeable* SDE: model the spread of two cointegrated names as
an Ornstein-Uhlenbeck process `dS = θ(μ − S)dt + σ dW` and trade deviations from μ.

**Verdict (2026-06): no edge in US large-cap. Shelved**, alongside the insider
(0.13 / −0.01) and sector-rotation (0.03) candidates.

## Result

| config | Sharpe | total | vol |
|---|---|---|---|
| canonical (net, 10 bps) | **−0.59** | −9.7% | 1.6% |
| canonical (**gross**, 0 bps) | **−0.30** | −5.2% | 1.6% |

The negative result is **not a cost problem** — it is negative *gross of costs*.
A 15-config diagnostic sweep (half-life band ∈ {5–63, 10–63, 15–63, 20–126, 10–40}
× entry_z ∈ {1.5, 2.0, 2.5}), run gross on the full sample, gave Sharpes ranging
**−0.31 to +0.18, clustered at zero**. There is no robust mean-reversion edge to
extract here, before costs, before the OOS test.

## Why this is "no signal," not a bug

- **The sweep flips sign between adjacent parameters.** `entry_z = 2.0` was
  systematically the *worst* cell while 1.5 and 2.5 sat near zero. A real signal
  moves roughly monotonically as the entry threshold widens; one that flips sign
  across a smooth parameter is noise. The single positive cell (+0.18) is gross,
  in-sample, and cherry-picked from 15 tries — surrounded by negative neighbours.
- **Direction was verified.** A sign error would show large *consistent* negative
  gross Sharpe (≈ −1+) that inverts to positive when flipped. We see ≈ 0, which is
  genuinely "no signal," not "inverted signal."
- **It matches the literature.** Do & Faff (2010) document distance/cointegration
  pairs returns decaying to near-zero after ~2002 as the trade crowded. Classic
  pairs trading in liquid US large-cap is one of the most arbitraged-away
  anomalies; this backtest reproduces that decay rather than beating it.

## Method (the code is correct and reusable; the universe is the problem)

Lookahead-safe rolling formation → trading split (see `run.py` docstring):
1. **Within-sector** candidate pairs (shared economic driver → plausible
   cointegration; cuts C(616,2)=190k to a tractable search).
2. Correlation pre-filter → **Engle-Granger**: OLS `logP_A = α + β logP_B + s`,
   ADF on residual `s` (p < 0.05, β > 0).
3. **OU fit** via the residual's AR(1): `θ = −ln(b)`, `half-life = ln2/θ`,
   `μ = a/(1−b)`, `σ_eq = std(s)`; keep tradeable half-lives, rank by ADF p-value.
4. **Freeze (α, β, μ, σ_eq)** and trade the next window only: z-score entry/exit,
   beta-neutral legs `w_A = 1/(1+β), w_B = −β/(1+β)`. A delisting force-closes the
   pair. Every parameter is estimated strictly before the window it trades.

Ranking by lowest ADF p-value pulled half-life to the ~5-day floor every window —
i.e. it selected the *statistically tightest* residuals, which are the most likely
to be in-sample noise. Forcing longer, economic half-lives (the sweep) did not
help: the edge is absent across the board, not hidden in one corner of the grid.

## What would be needed to revisit (not planned)
Mean-reversion that still pays tends to live where this universe isn't: shorter
horizons (intraday microstructure), ETF-vs-constituent baskets, or less-arbitraged
universes (small-cap, international). All are larger lifts than a config tweak. The
honest call is to shelve pairs and move on.

## Run
```
python run.py                          # canonical net book
python run.py --oos-start 2025-01-01   # (not informative — gross full-sample is already ~0)
```
