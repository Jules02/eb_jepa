#!/bin/bash
#SBATCH --job-name=gs_baselines
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=gs_baselines_%j.out
#SBATCH --error=gs_baselines_%j.err
# All Gray-Scott iso-protocol baselines in one job: persistence, linear extrapolation,
# climatology (free), + U-Net and FNO (trained one-step, autoregressive rollout). Same
# VRMSE metric / data / stride / test split as the JEPA eval, so the numbers are directly
# comparable. All baselines predict fields directly (no decoder floor).
#
# Usage:  sbatch run_baselines.sh [H] [EPOCHS] [SPLIT] [STRIDE]
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

H="${1:-30}"
EPOCHS="${2:-20}"
SPLIT="${3:-test}"
STRIDE="${4:-4}"

echo "=== gs baselines | H=$H epochs=$EPOCHS split=$SPLIT stride=$STRIDE ==="
uv run --project "$REPO" python -m examples.gray_scott.baselines \
  --H "$H" --epochs "$EPOCHS" --split "$SPLIT" --stride "$STRIDE" --which all
echo "=== DONE ==="
