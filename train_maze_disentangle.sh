#!/bin/bash
#SBATCH --job-name=maze_disentangle
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=maze_disentangle_%j.out
#SBATCH --error=maze_disentangle_%j.err
# Maze WM training with the h1/h2 disentangle split (geometry/tdist vs control/IDM
# + cross-decorrelation). Identical to train_maze_temporal.sh except the config.
# Isolates ONE variable vs train_maze_temporal: the latent split. After it finishes,
# run viz_distance_landscape (on h1) to check for a ridge at the maze doors.
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
uv run --project "$REPO" python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train/maze/train_maze_disentangle.yaml
