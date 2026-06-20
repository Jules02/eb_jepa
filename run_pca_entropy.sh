#!/bin/bash
#SBATCH --job-name=gs_pcaE
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=gs_pcaE_%j.out
#SBATCH --error=gs_pcaE_%j.err
# Latent PCA per phase colored by (1) normalized time and (2) (A,B) concentration entropy.
# -> results/pca_color_time.png, results/pca_color_entropy.png
#
# Usage:  sbatch run_pca_entropy.sh [CKPT] [PER_REGIME] [H]
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

CKPT="${1:-/lustre/work/vivatech-jepadormi/abenmanso/checkpoints/gray_scott/vjepa_v2/epoch_25.pth.tar}"
PER_REGIME="${2:-20}"
H="${3:-60}"

cd "$REPO"
uv run --project "$REPO" python -m examples.gray_scott.plot_pca_entropy \
  --ckpt "$CKPT" --per_regime "$PER_REGIME" --H "$H" --split test
echo "=== DONE -> results/pca_color_{time,entropy}.png ==="
