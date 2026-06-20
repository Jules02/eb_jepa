#!/bin/bash
#SBATCH --job-name=gs_vjepa_v2
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=gs_vjepa_v2_%j.out
#SBATCH --error=gs_vjepa_v2_%j.err
# Gray-Scott V-JEPA v2: GroupNorm encoder/predictor (no BatchNorm/EMA pathology),
# AdamW weight_decay + D4 symmetry augmentation (anti-overfit), and a proper validation
# measurement (2000-clip val + val_pred_loss logged separately from the VC term).
# Selection stays honest: NO inline VRMSE; pick the epoch by val_pred_loss, eval VRMSE
# once on TEST afterwards. Checkpoints -> checkpoints/gray_scott/vjepa_v2/.
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
uv run --project "$REPO" python -m examples.gray_scott.main \
  --fname examples/gray_scott/cfgs/train_vjepa_v2.yaml
