#!/bin/bash
#SBATCH --job-name=gs_decoder
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=gs_decoder_%j.out
#SBATCH --error=gs_decoder_%j.err
# Train a latent->field decoder ON TOP of a frozen V-JEPA checkpoint (probe for VRMSE).
# Encoder/predictor frozen; only the decoder learns MSE(decode(encode(x)), x) on TRAIN.
# The trained decoder is cached back INTO the checkpoint, so vrmse.sh loads it directly.
#
# Usage:  sbatch train_decoder.sh <ckpt.pth.tar> [epochs] [init_from.pth.tar]
#   init_from : optional — warm-start the decoder from another checkpoint's decoder
#               (e.g. train epoch_25's decoder from epoch_10's -> far fewer epochs).
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

CKPT="${1:?usage: sbatch train_decoder.sh <ckpt.pth.tar> [epochs] [init_from.pth.tar]}"
EPOCHS="${2:-15}"
INIT_FROM="${3:-}"

ARGS=(--ckpt "$CKPT" --epochs "$EPOCHS")
[ -n "$INIT_FROM" ] && ARGS+=(--init_from "$INIT_FROM")

echo "=== train decoder on $CKPT (epochs=$EPOCHS${INIT_FROM:+, warm-start from $INIT_FROM}) ==="
uv run --project "$REPO" python -m examples.gray_scott.train_decoder "${ARGS[@]}"
echo "=== DONE (decoder cached inside $CKPT) ==="
