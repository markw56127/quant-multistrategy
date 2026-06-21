# Sector Model — the origin model (SUPERSEDED, kept as the survivorship case study)

The first model in the repo: a long-only, sector-level allocator (HMM regime
detection + per-sector momentum/quality signals) benchmarked against SPY. It is
**superseded** and retained for two reasons — its data infrastructure is reused
across the repo, and it is the case study that defines the project's standard.

> **Why it's superseded:** the original sector_model reported a strong excess return
> over SPY (+47%) on the *current* S&P 500 constituents. That was **survivorship bias**:
> today's index excludes everything that went bankrupt or was delisted. Rebuilt on a
> point-in-time, survivorship-free universe (`../shared/universe_pit.py`), the same
> model's excess return over SPY flipped from **+47% to −41%**, and a long-short size
> factor it relied on fell from +187% (Sharpe 1.44) to +23% (0.26). Full write-up:
> [../SURVIVORSHIP_FINDING.md](../SURVIVORSHIP_FINDING.md).

What survived the post-mortem moved into the sleeves that are actually evaluated:
- the survivorship-free universe / price / sector machinery → `../shared/universe_pit.py`
  (reused by every later sleeve);
- cross-sectional factor signals → `../factor_model/` (value+quality, the part that
  held up);
- the sector-rotation layer → `../sector_rotation/` (tested standalone, Sharpe 0.03,
  shelved — the post-mortem cleared it of *bias*, not of *weakness*).

See the top-level [../README.md](../README.md) for the full scoreboard. The lesson
this model taught — that an impressive backtest is worthless until it survives a
point-in-time universe — is applied to everything built after it.
