#!/bin/bash
#SBATCH --job-name=gs_eval_large
#SBATCH --output=/lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/slurm_eval_large_%j.out
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --reservation=Vivatech

REPO=/lustre/work/vivatech-jepadormi/aduplessi/eb_jepa
cd $REPO
source env.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== LARGE TRAINING (dstc=64, hpre=64, batch=32, n_frames=16, eval_every=5) ==="
uv run python -m examples.gray_scott.main \
    --fname examples/gray_scott/cfgs/train_large.yaml

echo "=== ALL DONE ==="
