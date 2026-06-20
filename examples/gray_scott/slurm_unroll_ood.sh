#!/bin/bash
#SBATCH --job-name=gs_ood
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_ood_%j.out
#SBATCH --error=slurm_ood_%j.err
#
# Unroll JEPA from an UNSEEN (F,k) — "real physics" eyeball test.
#
# Usage:
#   CKPT=/path/to/latest.pth.tar F=0.020 K=0.0515 sbatch examples/gray_scott/slurm_unroll_ood.sh
#   CKPT=... F=0.020 K=0.0515 SRC=spirals H=60 sbatch examples/gray_scott/slurm_unroll_ood.sh
#
# CKPT  checkpoint (required)         F/K   the UNSEEN feed/kill rates (required)
# SRC   source regime to seed/calibrate from (default spirals)
# H     rollout horizon (default 60)
set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

: "${CKPT:?set CKPT=/path/to/latest.pth.tar}"
: "${F:?set F=<feed rate of unseen regime>}"
: "${K:?set K=<kill rate of unseen regime>}"
SRC="${SRC:-spirals}"
H="${H:-60}"

echo "=== OOD unroll: ckpt=$CKPT  (F,k)=($F,$K)  src=$SRC  H=$H ==="
uv run --project "$REPO" python -m examples.gray_scott.unroll_ood \
    --ckpt "$CKPT" --F "$F" --k "$K" --source-regime "$SRC" --H "$H" \
    --outdir "$REPO/examples/gray_scott/viz"
echo "=== Done -> $REPO/examples/gray_scott/viz/ood_*.gif ==="
