#!/bin/bash
#SBATCH --job-name=gs_eval_regimes
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_eval_regimes_%j.out
#SBATCH --error=slurm_eval_regimes_%j.err
#
# Per-regime Gray-Scott VRMSE benchmark on a GPU.
#
# Usage:
#   CKPT=/path/to/latest.pth.tar sbatch examples/gray_scott/slurm_eval_regimes.sh
#   CKPT=... H=30 N=80 sbatch examples/gray_scott/slurm_eval_regimes.sh
#
# CKPT   checkpoint to evaluate (required; no default — point it at YOUR run)
# H      rollout horizon (default 10; use 30 for the full Well Table-3 windows)
# N      clips sampled per regime (default 64)
# METRIC vrmse=The Well mean-of-ratios (default) | pooled=denominator-stable
#        diagnostic (use on low-F regimes where vrmse blows up on near-uniform frames)
set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

: "${CKPT:?set CKPT=/path/to/latest.pth.tar}"
H="${H:-10}"
N="${N:-64}"
METRIC="${METRIC:-vrmse}"   # vrmse (The Well paper) | pooled (denominator-stable)

echo "=== per-regime eval: ckpt=$CKPT  H=$H  N=$N  metric=$METRIC ==="
uv run --project "$REPO" python -m examples.gray_scott.eval_regimes \
    --ckpt "$CKPT" --H "$H" --n-per-regime "$N" --metric "$METRIC" \
    --outdir "$REPO/examples/gray_scott/viz"
echo "=== Done -> $REPO/examples/gray_scott/viz/regime_vrmse_{bars,curves}.png ==="
