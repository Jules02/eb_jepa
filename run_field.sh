#!/bin/bash
#SBATCH --job-name=gs_field
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=gs_field_%j.out
#SBATCH --error=gs_field_%j.err
# Distributional + perceptual field metrics per phase/horizon: sliced-Wasserstein (spatial
# 2D + value-space (A,B) cube-pooled) and VGG-Gatys style distance, for JEPA + all baselines.
# -> results/field_*.png + field_metrics_data.npz   (VGG19 weights downloaded once)
#
# Usage:  sbatch run_field.sh [CKPT] [PER_REGIME] [H]
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

CKPT="${1:-/lustre/work/vivatech-jepadormi/abenmanso/checkpoints/gray_scott/vjepa_v2/epoch_25.pth.tar}"
PER_REGIME="${2:-20}"
H="${3:-60}"

cd "$REPO"
uv run --project "$REPO" python -m examples.gray_scott.field_metrics \
  --ckpt "$CKPT" --per_regime "$PER_REGIME" --H "$H" --split test
echo "=== DONE -> results/field_*.png ==="
