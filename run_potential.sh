#!/bin/bash
#SBATCH --job-name=gs_pot
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=gs_pot_%j.out
#SBATCH --error=gs_pot_%j.err
# Does a monotone 'time potential' Phi(z) exist per phase (learned + held-out tested)?
# -> results/potential_pca_time.png, potential_collapse.png, potential_summary.png
#
# Usage:  sbatch run_potential.sh [CKPT] [PER_REGIME] [H]
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

CKPT="${1:-/lustre/work/vivatech-jepadormi/abenmanso/checkpoints/gray_scott/vjepa_v2/epoch_25.pth.tar}"
PER_REGIME="${2:-20}"
H="${3:-60}"

cd "$REPO"
uv run --project "$REPO" python -m examples.gray_scott.latent_potential \
  --ckpt "$CKPT" --per_regime "$PER_REGIME" --H "$H" --split test
echo "=== DONE -> results/potential_*.png ==="
