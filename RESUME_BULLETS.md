# Resume rewrite — Multi-Strategy Equity Trading System

Three problems with the current bullets, all fixable:
1. **PINN-RL is sold as a working capability** — but the repo retired it for
   lookahead leaks (commit 8efd66b). An interviewer asks "what was its Sharpe?"
   and the current phrasing torpedoes you. Reframe as a *diagnosis* (your
   strongest move — it matches the survivorship bullet that already lands).
2. **"over 50 US equities"** — actual repo is **616 survivorship-free tickers**.
   You're underselling by 12x.
3. **The true-OOS test isn't on the page** — it's the single best rigor signal
   you have. Add it.

---

## Rewritten project block (drop-in)

**Multi-Strategy Equity Trading System** *(Python, PyTorch, LightGBM, Stable-Baselines3, Gymnasium)* — Apr 2026 – Present

- Built a survivorship-free backtester over **616 US equities** on a shared
  point-in-time universe with transaction costs, slippage, and borrow fees;
  implemented six strategy sleeves (value/quality/momentum factors, sector
  rotation, long–short, PEAD earnings drift, regime, and a PINN–RL research sleeve).

- Diagnosed survivorship bias inflating backtest alpha: corrected a long–short
  size factor from a spurious **+187% (Sharpe 1.44) to a literature-consistent
  +23% (Sharpe 0.26)**, and flipped sector-model excess return over SPY from
  **+47% to −41%** once delisted names were retained.

- Architected the PINN–RL sleeve (β-VAE/Fokker–Planck latent market-state factors
  → Soft Actor–Critic position sizer, evaluated via 50k-path Monte Carlo on
  Sharpe/Sortino/VaR/CVaR); **traced lookahead leakage that had inflated its edge
  and retired the sleeve** after a clean re-test showed no alpha.

- Combined surviving market-neutral sleeves into an expanding-window risk-parity
  book (in-sample Sharpe ~1.0), then ran a **true out-of-sample test on 18 months
  of held-out post-development data**: the book degraded from **0.75 to ~0 Sharpe**,
  confirming the equity-factor edge did not survive and preventing a false-positive
  deployment.

---

## Why this version wins

- Every bullet now survives interrogation — there is no claim an interviewer can
  puncture, because the failures are *stated as findings*.
- It reads as a researcher who **kills their own ideas with held-out data** —
  exactly the screen at Two Sigma / Citadel / DRW / AQR-style shops.
- The Fokker–Planck / SAC / Monte Carlo machinery is still there (it's impressive
  and real); it's just no longer claiming an edge it doesn't have.

## PAGE-CONSTRAINED VERSION (3 bullets — use this one)

The resume page is full, so we keep the original 3-bullet count and pack the
trend sleeve + OOS into the existing bullets rather than adding a 4th/5th.

```latex
    \item Built a survivorship-free backtester over 616 US equities and cross-asset futures on a point-in-time universe with transaction costs, slippage, and borrow fees, across factor, long--short, earnings, regime, trend, and PINN--RL sleeves
    \item Diagnosed survivorship bias inflating alpha: corrected a size factor from +187\% (Sharpe 1.44) to +23\% (0.26) and flipped sector excess return over SPY from +47\% to $-$41\%; true out-of-sample tests then retired the equity-factor book (0.75 $\rightarrow$ 0) while a cross-asset trend sleeve held (0.61 $\rightarrow$ 0.38)
    \item Architected a PINN--RL sleeve (Fokker--Planck / $\beta$-VAE latent states $\rightarrow$ Soft Actor--Critic sizer, 50{,}000-path Monte Carlo on Sharpe, Sortino, VaR, CVaR with LLM-scored news sentiment as a drift term); traced and fixed lookahead leakage that had inflated its edge, then retired it on a clean re-test
```

If bullet 2 wraps an extra line, drop "(Sharpe 1.44)" and "(0.26)" to reclaim it.

Honesty note: the trend OOS edge is concentrated in equity trend (rates trend was
negative OOS) and rests on 18 months — true, and an interviewer who probes it gets
a sophisticated answer instead of a punctured claim. Do NOT inflate to "Sharpe 0.6
trend strategy." The four-bullet (expanded) version above is kept only for the case
where you free a line elsewhere.
