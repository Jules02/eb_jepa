#!/bin/bash
#SBATCH --job-name=gs_final
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=gs_final_%j.out
#SBATCH --error=gs_final_%j.err
# Final per-phase VRMSE table (mean ± std over clips) for JEPA + all baselines at
# h=1,15,30,60. Same fixed clips + paper metric for every model. Cached decoder/U-Net/FNO.
#
# Usage:  sbatch run_final_table.sh [CKPT] [PER_REGIME] [H]
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

CKPT="${1:-/lustre/work/vivatech-jepadormi/abenmanso/checkpoints/gray_scott/vjepa_v2/epoch_25.pth.tar}"
PER_REGIME="${2:-20}"   # clips per phase -> mean±std over this many
H="${3:-60}"

cd "$REPO"
uv run --project "$REPO" python -m examples.gray_scott.final_table \
  --ckpt "$CKPT" --per_regime "$PER_REGIME" --H "$H" --horizons 1,15,30,60 --split test
echo "=== DONE ==="
