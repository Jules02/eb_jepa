#!/bin/bash
#SBATCH --job-name=gs_vjepa
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=gs_vjepa_%j.out
#SBATCH --error=gs_vjepa_%j.err
# Train the Gray-Scott temporal V-JEPA "correctly": same model as aduplessi's run.
# NO VRMSE monitoring (selecting on the reported metric = cheating). Epoch selection is
# done post-hoc on the latent val-loss (the SSL objective). VRMSE is run ONCE, on TEST,
# only after the checkpoint is chosen. Checkpoints -> our own space.
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
uv run --project "$REPO" python -m examples.gray_scott.main \
  --fname examples/gray_scott/cfgs/train_vjepa.yaml
