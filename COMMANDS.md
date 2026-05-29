# Trading Model — Command Reference

All commands run from the project root with the conda environment active.

```bash
conda activate trading-model
```

---

## 1. Initial Sanity Test

Confirms the full 6-stage pipeline runs without crashing. Uses minimal epochs and
skips all slow API calls (LLM sentiment, fundamental data fetches). Run this first.

```bash
python main.py --mode train \
  --no-sentiment \
  --no-fundamentals \
  --epochs-pinn 100 \
  --epochs-rl 5
```

**What to verify in the output:**
- PCA logs per-sector cumulative explained variance (should be > 60%)
- `VAE Epoch 100/500` — training loop is running
- `PINN Epoch 100/100 | total=... | phys=...` — values should be decreasing
- `λ_phys=... λ_bc=... λ_ic=...` — adaptive weights are updating
- `Episode 5/5 | Reward=... | Sharpe=...` — any finite number
- `BACKTEST RESULTS` table appears at the end

---

## 2. RL Learning Verification

Loads the PINN from the checkpoint trained in step 1. Runs enough RL episodes to
see whether the Sharpe ratio is trending upward. Run this before committing to an
overnight training.

```bash
mkdir -p logs

python main.py --mode train \
  --no-sentiment \
  --no-fundamentals \
  --skip-pinn \
  --epochs-rl 100 \
  2>&1 | tee logs/rl_check.log
```

Watch the Sharpe trend live in a second terminal:

```bash
grep "Sharpe=" logs/rl_check.log
```

**What to look for:** Sharpe should be trending upward (or recovering from early
negative values) by episode ~50. If it stays flat near 0, try PPO as a sanity
check — it's more stable early in training:

```bash
python main.py --mode train \
  --no-sentiment \
  --no-fundamentals \
  --skip-pinn \
  --algo PPO \
  --epochs-rl 100
```

---

## 3. Overnight Full Training

### Fast variant (no sentiment, no fundamentals — recommended for first run)

```bash
mkdir -p logs checkpoints

nohup python main.py --mode train \
  --no-sentiment \
  --no-fundamentals \
  --epochs-pinn 5000 \
  --epochs-rl 500 \
  > logs/train_$(date +%Y%m%d).log 2>&1 &

echo "Training PID: $!"
```

### Full variant (LLM sentiment + fundamental data — adds ~1–2 hours)

```bash
nohup python main.py --mode train \
  --epochs-pinn 5000 \
  --epochs-rl 500 \
  > logs/train_full_$(date +%Y%m%d).log 2>&1 &

echo "Training PID: $!"
```

### PPO baseline (trains PPO instead of SAC, saves to separate checkpoints)

```bash
nohup python main.py --mode train \
  --no-sentiment \
  --no-fundamentals \
  --algo PPO \
  --skip-pinn \
  --epochs-rl 500 \
  --checkpoints checkpoints_ppo \
  > logs/train_ppo_$(date +%Y%m%d).log 2>&1 &
```

### TRPO baseline

```bash
nohup python main.py --mode train \
  --no-sentiment \
  --no-fundamentals \
  --algo TRPO \
  --skip-pinn \
  --epochs-rl 500 \
  --checkpoints checkpoints_trpo \
  > logs/train_trpo_$(date +%Y%m%d).log 2>&1 &
```

### Monitor an overnight run

```bash
# Current progress (last 50 lines)
tail -50 logs/train_*.log

# Sharpe trend across episodes
grep "Sharpe=" logs/train_*.log | tail -20

# PINN adaptive weight evolution
grep "λ_phys" logs/train_*.log | tail -10

# Check for any errors or NaNs
grep -i "error\|traceback\|nan\|inf" logs/train_*.log | head -20
```

---

## 4. Evaluation and Backtesting

> **Important:** Use the same `--no-sentiment` / `--no-fundamentals` flags that
> you used at training time. Mismatching them changes the feature matrix dimension
> and will cause a shape error.

### In-sample backtest (full training period)

```bash
mkdir -p results

python main.py --mode backtest \
  --no-sentiment \
  --no-fundamentals \
  --out results/backtest_insample.csv
```

### Out-of-sample backtest (specify a hold-out date range)

```bash
python main.py --mode backtest \
  --no-sentiment \
  --no-fundamentals \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --out results/backtest_oos.csv
```

### Backtest a PPO or TRPO checkpoint

```bash
python main.py --mode backtest \
  --no-sentiment \
  --no-fundamentals \
  --checkpoints checkpoints_ppo \
  --out results/backtest_ppo.csv
```

### Backtest with the interactive dashboard

```bash
python main.py --mode backtest \
  --no-sentiment \
  --no-fundamentals \
  --dashboard \
  --dashboard-port 8050
```

Then open `http://localhost:8050` in a browser.

### Launch dashboard only (from existing checkpoints, no recomputation)

```bash
python main.py --mode dashboard
```

---

## 5. Inference — Today's Trade Signals

Uses pretrained checkpoints only. No retraining. Fetches the last 90 days of OHLCV
to compute the current latent state, then queries the SAC policy.

### Full portfolio signal (all 60 assets)

```bash
python main.py --mode infer
```

Output shows recommended long/short weights for every stock and ETF in the universe,
ranked by conviction.

### Single stock deep report

```bash
python main.py --mode infer --ticker NVDA
```

Replace `NVDA` with any ticker in the trained universe. The full pipeline still runs
under the hood (needed to compute the joint latent state), but the output focuses on
that one stock and includes:

- Recommended position size and direction
- PINN drift direction (BULLISH / BEARISH) and magnitude
- Latent-space uncertainty (how confident the model is)
- PINN density p(z,t) — how "normal" the current market state is for this stock
- 20-day Monte Carlo return distribution (10th / median / 90th percentile)
- Key technical indicator snapshot with plain-English interpretation

Examples:

```bash
python main.py --mode infer --ticker AAPL
python main.py --mode infer --ticker MSFT
python main.py --mode infer --ticker JPM
python main.py --mode infer --ticker XOM
```

The ticker must be in the trained universe. To see all available tickers,
check `data.tickers` in `config/config.yaml`.

---

## 6. Skipping Stages (Loading from Checkpoints)

Useful when iterating on a single stage without rerunning the whole pipeline.

| Flag | What it skips | Requires |
|------|--------------|---------|
| `--skip-data` | Re-downloading OHLCV | Cached data in `cache/data/` |
| `--skip-features` | PCA + VAE retraining | `checkpoints/pca.pkl`, `checkpoints/vae.pt` |
| `--skip-pinn` | PINN retraining | `checkpoints/pinn.pt` |
| `--skip-rl` | RL retraining | `checkpoints/sac_agent.pt` (or ppo/trpo equivalent) |

Example — retrain only the RL agent, everything else loaded from cache:

```bash
python main.py --mode train \
  --no-sentiment \
  --no-fundamentals \
  --skip-data \
  --skip-features \
  --skip-pinn \
  --epochs-rl 300
```

---

## 7. Sensitivity Analysis

Systematically zeros out each indicator family (trend, momentum, volatility, volume,
structure) and measures the impact on PINN loss and RL Sharpe. Useful for deciding
which indicators to keep or drop.

```bash
python main.py --mode sensitivity
```

---

## Quick Reference

| Goal | Command |
|------|---------|
| Smoke test | `python main.py --mode train --no-sentiment --no-fundamentals --epochs-pinn 100 --epochs-rl 5` |
| RL learning check | `python main.py --mode train --no-sentiment --no-fundamentals --skip-pinn --epochs-rl 100` |
| Overnight (fast) | `nohup python main.py --mode train --no-sentiment --no-fundamentals --epochs-pinn 5000 --epochs-rl 500 > logs/train.log 2>&1 &` |
| Overnight (full) | `nohup python main.py --mode train --epochs-pinn 5000 --epochs-rl 500 > logs/train_full.log 2>&1 &` |
| Overnight PPO | `nohup python main.py --mode train --no-sentiment --no-fundamentals --algo PPO --skip-pinn --epochs-rl 500 --checkpoints checkpoints_ppo > logs/train_ppo.log 2>&1 &` |
| Overnight TRPO | `nohup python main.py --mode train --no-sentiment --no-fundamentals --algo TRPO --skip-pinn --epochs-rl 500 --checkpoints checkpoints_trpo > logs/train_trpo.log 2>&1 &` |
| In-sample backtest | `python main.py --mode backtest --no-sentiment --no-fundamentals --out results/bt.csv` |
| OOS validation | `python main.py --mode backtest --no-sentiment --no-fundamentals --start-date 2023-01-01 --out results/oos.csv` |
| Backtest + dashboard | `python main.py --mode backtest --no-sentiment --no-fundamentals --dashboard` |
| Today's signals | `python main.py --mode infer` |
| Single stock | `python main.py --mode infer --ticker NVDA` |
| Retrain RL only | `python main.py --mode train --no-sentiment --no-fundamentals --skip-data --skip-features --skip-pinn --epochs-rl 300` |
| Sensitivity sweep | `python main.py --mode sensitivity` |