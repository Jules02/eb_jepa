#!/bin/bash
#SBATCH --job-name=gs_probe
#SBATCH --output=/lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/slurm_probe_%j.out
#SBATCH --time=30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-jepadormi

source /lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/env.sh
cd /lustre/work/vivatech-jepadormi/aduplessi/eb_jepa

CKPT=/lustre/work/vivatech-jepadormi/aduplessi/checkpoints/gray_scott/dev/epoch_19.pth.tar

echo "=== F/k probe — JEPA small D=16 stride=4 epoch=19 ==="

uv run python -m examples.gray_scott.probe \
    --ckpt "$CKPT" \
    --time-stride 4 \
    --steps 2000 \
    --batch-size 64 \
    --epoch-size 4000

echo "=== DONE ==="
