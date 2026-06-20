#!/bin/bash
#SBATCH --job-name=gs_render
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=gs_render_%j.out
#SBATCH --error=gs_render_%j.err
# Render JEPA + truth field rollouts for the 4 slide phases -> outputs/slides_fields.npz,
# then build outputs/gif_regimes.gif (2x2 dynamic/static) + outputs/gif_diff.gif (JEPA error).
#
# Usage:  sbatch run_render_slides.sh [CKPT] [FRAMES]
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

CKPT="${1:-/lustre/work/vivatech-jepadormi/abenmanso/checkpoints/gray_scott/vjepa_v2/epoch_25.pth.tar}"
FRAMES="${2:-60}"

cd "$REPO"
uv run --project "$REPO" python -m examples.gray_scott.render_slides \
  --ckpt "$CKPT" --frames "$FRAMES" --split test
echo "=== DONE -> outputs/gif_regimes.gif, outputs/gif_diff.gif ==="
