#!/bin/bash
#SBATCH --job-name=gs_vrmse
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=gs_vrmse_%j.out
#SBATCH --error=gs_vrmse_%j.err
# Field-space VRMSE eval of a Gray-Scott temporal-JEPA checkpoint (The Well Table 3 windows).
# Usage:  sbatch vrmse.sh [CKPT] [H] [SPLIT]
#   CKPT   checkpoint to evaluate (default: aduplessi's gray_scott dev latest)
#   H      autoregressive rollout horizon (default 30 -> windows [6:12] & [13:30])
#   SPLIT  test (default, the clean report) | valid
# eval.py picks the rollout stride = the checkpoint's own training stride automatically.
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312

SRC="${1:-/lustre/work/vivatech-jepadormi/aduplessi/checkpoints/gray_scott/dev/latest.pth.tar}"
H="${2:-30}"
SPLIT="${3:-test}"

# eval.py:build_decoder() may TRAIN a decoder and WRITE it back into the checkpoint. If the
# file is ours (writable) we eval IN PLACE (and reuse the decoder cached by train_decoder.sh).
# If it is read-only (e.g. aduplessi's), we eval a LOCAL COPY in our own space instead.
if [ -w "$SRC" ]; then
    CKPT="$SRC"
    echo "[vrmse] writable checkpoint -> evaluating in place: $CKPT"
else
    CKPT="$EBJEPA_CKPTS/gray_scott/_eval_copies/$(basename "$(dirname "$SRC")")_$(basename "$SRC")"
    mkdir -p "$(dirname "$CKPT")"
    [ -f "$CKPT" ] || cp "$SRC" "$CKPT"
    echo "[vrmse] read-only source -> evaluating local copy: $CKPT"
fi

echo "=== gs VRMSE eval on $CKPT  split=$SPLIT H=$H ==="
uv run --project "$REPO" python -m examples.gray_scott.eval --ckpt "$CKPT" --H "$H" --split "$SPLIT"
echo "=== DONE ==="
