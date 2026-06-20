# smallcap_factor — same factor engine, small-cap pond  ·  thesis NOT supported

The repo's recurring lesson is that every equity edge dies because US large-cap is
the most arbitraged market on earth. Factor premia (value, quality, size especially)
are documented to be *stronger* where there's less coverage and slower arbitrage — so
this sleeve runs the **identical** machinery as `factor_model` (same sector-neutral
factor z-scores, same EDGAR fundamentals, same long-short quintile construction, same
costs and dev/OOS split) on the **S&P 600 small-cap** universe. Only the pond changes,
so any difference is a pond effect, not a method effect.

**Verdict (2026-06): thesis not supported with tradeable data.** The one strong signal
that appears is a survivorship artifact; once removed, there is no cross-sectional edge.

## Result

| composite | total | Sharpe | IC | IC t-stat |
|---|---|---|---|---|
| all 5 factors | −16.1% | +0.02 | +0.0282 | **+3.35** |
| **ex-size** (4 factors) | −79.0% | −0.91 | +0.0029 | **+0.32** |

Standalone factor L/S Sharpe (gross, full sample):

| value | momentum | quality | low_vol | **size** |
|---|---|---|---|---|
| +0.44 | −0.16 | −0.64 | −1.16 | **+3.21 (total +3,566%)** |

## The entire signal was a survivorship artifact

The all-5 composite shows a *highly significant* cross-sectional IC (t = 3.35). It is
fake. **Drop the `size` factor and the IC collapses to t = 0.32 — insignificant.** The
whole apparent small-cap edge was the size factor, and `size` returned **+3,566%**.

That number is not alpha; it is the repo's founding bias re-demonstrated. There is no
free point-in-time S&P 600 membership, so we used *today's* constituents — which
conditions on survival. The smallest names that survived into the current index are
precisely the ones that did not go to zero, so "buy the smallest" is a look-ahead on
who lived. This is the same artifact that inflated the original sector_model's size
factor to +187% (`SURVIVORSHIP_FINDING.md`) — but **~19× larger here**, because
small-caps delist far more often than large-caps. Small-cap is exactly where this bias
bites hardest, as flagged up front in `universe.py`.

## What survives the bias

Stripped of `size`, there is no aggregate cross-sectional edge in this universe
(IC t = 0.32). The only individually-positive real factor is **value (+0.44)**, and it
merely *matches* large-cap value (+0.56) rather than beating it; `quality` and `low_vol`
are actually *worse* in small-cap. So even the trustworthy part of the experiment does
not support "factors work better in small-cap" — at least not at a magnitude that
survives contact with the data we can get for free.

## Honest limitations / what a real test needs

- **Point-in-time S&P 600 membership** (not freely available) — the only way to clean
  the size factor and trust the composite. Without it, magnitudes here are unreliable.
- Factor attribution Sharpes above are **gross**; net of small-cap costs (15 bps + 3%
  borrow, already applied to the composite) the weak positives erode further.
- Conclusion: the pond thesis is not refuted in principle, but it cannot be *confirmed*
  with current-constituents data, and the clean signal that remains is ~zero. Shelved
  alongside the other honest negatives.

## Run
```
python run.py                          # full sample, all 5 factors
python run.py --oos-start 2025-01-01   # OOS-only
# diagnostic: drop the survivorship-coupled size factor via factors.composite in config
```
