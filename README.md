# Trading Model

> **STATUS (2026-06): the PINN-RL model described below is RETIRED.** Its
> results were lookahead artifacts — see [LOOKAHEAD_FINDING.md](LOOKAHEAD_FINDING.md).
> After fixing the leaks it shows no edge (training Sharpe −1 to −3.8, no
> learning trend). The repo's live work is the multi-strategy program:
> `factor_model` + `earnings_model` combined via `combine_strategies.py`
> (honest net Sharpe 0.87). Two sleeve-#3 candidates were built, tested, and
> shelved on pre-set criteria: `insider_model` (Sharpe 0.13 / −0.01 across
> two attempts) and `sector_rotation` (0.03). Negative results recorded in
> their READMEs. Lessons in SURVIVORSHIP_FINDING.md and LOOKAHEAD_FINDING.md.

A research-grade algorithmic trading system that combines physics-informed neural networks (PINNs), variational autoencoders, and deep reinforcement learning to generate portfolio allocation decisions across 50 US equities.

## What the model does

The system treats the latent representation of the market as a probability distribution evolving through time according to a Fokker-Planck partial differential equation. Rather than predicting prices directly, it models the *density* of possible market states and uses that uncertainty to inform portfolio decisions. A Soft Actor-Critic (SAC) reinforcement learning agent then learns to allocate capital by maximizing risk-adjusted returns, using the PINN's output — including physics-grounded uncertainty estimates — as part of its state observation.

**Training pipeline (6 stages):**

1. **Data** — downloads 10 years of daily OHLCV data for 50 stocks across 5 sectors, computes ~35 technical indicators per ticker, and optionally enriches with LLM-based news sentiment via the Claude API
2. **Factor discovery** — sector-level PCA extracts 12 eigen-factors, then a β-VAE compresses all features into a 16-dimensional latent representation with explicit uncertainty estimates
3. **PINN training** — a physics-informed network is trained to satisfy the Fokker-Planck equation on the latent trajectories, with boundary conditions, initial conditions, and sentiment as a drift modifier
4. **RL training** — a SAC agent is trained in a custom Gymnasium environment that uses real historical returns, PINN uncertainty, and transaction costs
5. **Backtesting** — the learned policy and a Lagrangian-constrained portfolio optimizer are evaluated in parallel against SPY
6. **Risk analysis** — Monte Carlo simulation (50,000 paths), Value-at-Risk, Sharpe/Sortino ratios, and confidence bands

## Why this model is interesting

**Novel combination:** Most quant models pick one of these approaches. This one uses the PINN as a *physics prior* that regularizes the latent space learned by the VAE, and then feeds that physics-grounded representation to the RL agent. The Fokker-Planck framing naturally gives a probability distribution over future states rather than a point forecast.

**Uncertainty as a first-class input:** The VAE produces per-timestep uncertainty estimates for each latent factor. The RL agent observes these uncertainties directly, so it can learn to be more conservative when the market is in an uncharted region of latent space.

**Sentiment as drift:** News sentiment (scored by Claude claude-sonnet-4-6 or FinBERT) is injected directly into the Fokker-Planck drift term, giving the physics model a principled way to incorporate non-price information.

**Sensitivity analysis:** The indicator families (trend, momentum, volatility, volume, structure) can be systematically zeroed out to measure each family's contribution to PINN predictive power and RL reward.

## Setup

### Environment

```bash
# Create and activate the conda environment
mamba create -n trading-model python=3.11 -y
mamba activate trading-model

# Install PyMC (complex deps — best via conda-forge)
mamba install -c conda-forge "pymc" "pandas-ta=0.3.14b" -y

# Install PyTorch (Apple Silicon MPS support)
pip install torch torchvision

# Downgrade numpy for pandas-ta compatibility
pip install "numpy<2"

# Install remaining dependencies
pip install scikit-learn yfinance requests newsapi-python anthropic \
  "transformers>=4.36.0" "gymnasium>=0.29.0" "stable-baselines3>=2.2.0" \
  plotly seaborn dash pyyaml tqdm joblib python-dotenv loguru
```

### API keys

```bash
cp .env.example .env
# Edit .env and fill in:
#   ANTHROPIC_API_KEY  — required for Claude-based sentiment scoring
#   NEWSAPI_KEY        — optional, for real-time news
#   POLYGON_API_KEY    — optional, for higher-quality market data
```

## How to run

### Full training pipeline

```bash
mamba activate trading-model
python main.py --mode train
```

This runs all 6 stages end-to-end (~hours, depending on hardware and whether sentiment is enabled). Checkpoints are saved to `checkpoints/`.

**Common flags:**

| Flag | Effect |
|------|--------|
| `--no-sentiment` | Skip LLM sentiment (much faster) |
| `--skip-features` | Load pretrained PCA + VAE from `checkpoints/` |
| `--skip-pinn` | Load pretrained PINN from `checkpoints/pinn.pt` |
| `--skip-rl` | Load pretrained SAC agent from `checkpoints/sac_agent.pt` |
| `--epochs-pinn N` | Override PINN training epochs (default 5000) |
| `--epochs-rl N` | Override RL episodes (default 200) |
| `--dashboard` | Launch the Dash dashboard after training |

### Quick run (example case — fastest path)

To skip the slow training stages and test the pipeline end-to-end on cached data:

```bash
# First run: pull data and train everything, skip sentiment
python main.py --mode train --no-sentiment --epochs-pinn 500 --epochs-rl 20

# Subsequent runs: everything loaded from checkpoints, no retraining
python main.py --mode train --skip-data --skip-features --skip-pinn --skip-rl --no-sentiment
```

Checkpoints saved after the first full run:

| File | Contents |
|------|----------|
| `checkpoints/pca.pkl` | PCA + Factor Analysis + scalers |
| `checkpoints/vae.pt` | VAE weights + normalization stats |
| `checkpoints/pinn.pt` | PINN density/drift/diffusion networks |
| `checkpoints/sac_agent.pt` | SAC actor, critic, target networks |

### Other modes

```bash
# Generate today's signals using pretrained models
python main.py --mode infer

# Backtest only (uses checkpointed policy weights)
python main.py --mode backtest

# Launch the interactive Dash dashboard
python main.py --mode dashboard

# Indicator sensitivity analysis (measures per-family impact)
python main.py --mode sensitivity
```

## Configuration

All hyperparameters are in [config/config.yaml](config/config.yaml):

- `data.tickers` — universe of 50 stocks across 5 sectors
- `pde.diffusion_coeff` — controls how quickly the probability density spreads in latent space
- `pde.chaos_sensitivity` — Lyapunov-like sensitivity coefficient for market chaos
- `pinn.*` — network architecture and physics loss weights
- `rl.*` — SAC hyperparameters (learning rate, entropy coefficient, buffer size)
- `backtest.*` — transaction costs, slippage, position limits, rebalance frequency

## Project structure

```
├── config/config.yaml          # all hyperparameters
├── data/
│   ├── ingestion.py            # yfinance downloader
│   ├── technical_indicators.py # 35 indicators across 5 families
│   ├── sentiment.py            # Claude / FinBERT sentiment scoring
│   └── pipeline.py             # orchestrates data stages
├── features/
│   ├── pca_factors.py          # sector PCA + global factor extraction
│   └── autoencoder.py          # β-VAE with uncertainty estimation
├── physics/
│   ├── pde_system.py           # Fokker-Planck PDE specification
│   └── pinn.py                 # physics-informed neural network solver
├── rl/
│   ├── environment.py          # custom Gymnasium trading env
│   └── sac_agent.py            # Soft Actor-Critic implementation
├── optimization/
│   └── estimators.py           # Lagrangian portfolio optimizer
├── risk/
│   ├── metrics.py              # Sharpe, Sortino, VaR, CVaR
│   └── monte_carlo.py          # GBM path simulation
├── backtest/
│   └── engine.py               # parallel backtest runner
├── visualization/
│   └── dashboard.py            # Plotly Dash dashboard
├── train.py                    # 6-stage training orchestrator
└── main.py                     # CLI entry point
```
