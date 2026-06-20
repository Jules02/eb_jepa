#!/bin/bash
#SBATCH --job-name=gs_pca
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=gs_pca_%j.out
#SBATCH --error=gs_pca_%j.err
# PCA of Gray-Scott states in image space vs JEPA latent space, colored by phase.
# -> results/pca_image_latent.png
#
# Usage:  sbatch run_pca.sh [CKPT] [PER_REGIME] [FRAMES_PER_CLIP]
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

CKPT="${1:-/lustre/work/vivatech-jepadormi/abenmanso/checkpoints/gray_scott/vjepa_v2/epoch_25.pth.tar}"
PER_REGIME="${2:-20}"
FPC="${3:-6}"

cd "$REPO"
uv run --project "$REPO" python -m examples.gray_scott.plot_pca \
  --ckpt "$CKPT" --per_regime "$PER_REGIME" --fpc "$FPC" --split test
echo "=== DONE -> results/pca_image_latent.png ==="
