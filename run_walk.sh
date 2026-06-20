#!/bin/bash
#SBATCH --job-name=gs_walk
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=gs_walk_%j.out
#SBATCH --error=gs_walk_%j.err
# Random-walk vs directed-walk analysis of Gray-Scott trajectories per phase (latent &
# image): MSD exponent alpha, direction autocorrelation, straightness. -> results/walk_*.png
#
# Usage:  sbatch run_walk.sh [CKPT] [PER_REGIME] [H]
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

CKPT="${1:-/lustre/work/vivatech-jepadormi/abenmanso/checkpoints/gray_scott/vjepa_v2/epoch_25.pth.tar}"
PER_REGIME="${2:-20}"
H="${3:-60}"

cd "$REPO"
uv run --project "$REPO" python -m examples.gray_scott.latent_walk \
  --ckpt "$CKPT" --per_regime "$PER_REGIME" --H "$H" --split test
echo "=== DONE -> results/walk_*.png ==="
