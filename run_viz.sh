#!/bin/bash
#SBATCH --job-name=gs_viz
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=gs_viz_%j.out
#SBATCH --error=gs_viz_%j.err
# Render 120-frame rollout GIFs (truth/jepa/unet/fno/persistence/linear) — ONE GIF PER
# Gray-Scott phase (gliders/bubbles/maze/worms/spirals/spots) -> results/viz_<phase>.gif.
# U-Net/FNO are loaded from cache if present, else trained+saved once.
#
# Usage:  sbatch run_viz.sh [CKPT] [H] [PER_REGIME]
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

CKPT="${1:-/lustre/work/vivatech-jepadormi/abenmanso/checkpoints/gray_scott/vjepa_v2/epoch_10.pth.tar}"
H="${2:-120}"
PER_REGIME="${3:-1}"   # GIFs per phase (1 -> 6 GIFs; 2 -> 12 GIFs ...)

cd "$REPO"
uv run --project "$REPO" python -m examples.gray_scott.viz_rollouts \
  --ckpt "$CKPT" --H "$H" --split test --mode regimes --per_regime "$PER_REGIME" --out_dir results
echo "=== DONE -> results/viz_<phase>.gif ==="
