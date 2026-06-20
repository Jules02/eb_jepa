#!/bin/bash
#SBATCH --job-name=maze_vcreg
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=maze_vcreg_%j.out
#SBATCH --error=maze_vcreg_%j.err
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
uv run --project "$REPO" python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train/maze/train_maze_vcreg.yaml
