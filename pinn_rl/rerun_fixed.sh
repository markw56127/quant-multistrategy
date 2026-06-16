#!/usr/bin/env bash
# Re-evaluate the PINN-RL model AFTER the 2026-06 lookahead fixes.
# (See header of previous version / git log for the list of fixes.)
#
# Usage:  cd pinn_rl && bash rerun_fixed.sh
# Expected runtime: ~30-45 min for 150 episodes.

set -u

# Env activation is best-effort: if the calling shell already has the
# trading-model env active (torch importable), we just use it.
if ! python -c "import torch" 2>/dev/null; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate trading-model 2>/dev/null || mamba activate trading-model 2>/dev/null || true
fi
if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: could not import torch. Activate the env first:  conda activate trading-model"
  exit 1
fi

mkdir -p checkpoints_fixed logs results
cp -n checkpoints/pinn.pt checkpoints_fixed/pinn.pt 2>/dev/null || true
if [ ! -f checkpoints_fixed/pinn.pt ]; then
  echo "ERROR: checkpoints/pinn.pt not found — needed for --skip-pinn"
  exit 1
fi

LOG="logs/rerun_fixed_$(date +%Y%m%d_%H%M).log"
nohup python train.py \
  --skip-data \
  --skip-pinn \
  --no-sentiment \
  --no-fundamentals \
  --epochs-rl 150 \
  --checkpoints checkpoints_fixed \
  > "$LOG" 2>&1 &

echo "Started fixed rerun, PID: $!"
echo "Log: $LOG"
echo "Watch with:  tail -f $LOG | grep -E 'Episode|BACKTEST|Sharpe|Error'"
echo ""
echo "How to read the result:"
echo "  - Old in-sample backtest showed SAC Sharpe ~ -0.16 without the engine leak."
echo "  - Below ~0.5 in-sample confirms no real edge; chapter closed."
echo "  - Anything >2 now deserves suspicion, not celebration."
