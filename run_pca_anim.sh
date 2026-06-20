#!/bin/bash
#SBATCH --job-name=gs_pcaAnim
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=gs_pcaAnim_%j.out
#SBATCH --error=gs_pcaAnim_%j.err
# Animated latent-PCA / state-entropy GIF -> outputs/gif_pca_entropy.gif (+ pca_anim_data.npz).
#
# Usage:  sbatch run_pca_anim.sh [CKPT] [PER_REGIME] [H]
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

CKPT="${1:-/lustre/work/vivatech-jepadormi/abenmanso/checkpoints/gray_scott/vjepa_v2/epoch_25.pth.tar}"
PER_REGIME="${2:-12}"
H="${3:-60}"

cd "$REPO"
uv run --project "$REPO" python -m examples.gray_scott.pca_entropy_anim \
  --ckpt "$CKPT" --per_regime "$PER_REGIME" --H "$H" --split test
echo "=== DONE -> outputs/gif_pca_entropy.gif ==="
