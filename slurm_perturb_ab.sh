#!/bin/bash
#SBATCH --job-name=gs_perturb_ab
#SBATCH --output=/lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/slurm_perturb_ab_%j.out
#SBATCH --time=20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-jepadormi

source /lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/env.sh
cd /lustre/work/vivatech-jepadormi/aduplessi/eb_jepa

CKPT=/lustre/work/vivatech-jepadormi/aduplessi/checkpoints/gray_scott/dev/epoch_19.pth.tar

uv run python -m examples.gray_scott.perturb_ab_gif \
    --ckpt "$CKPT" \
    --split valid \
    --H 40 \
    --time-stride 4 \
    --delta 1.0 \
    --fps 8 \
    --outdir examples/gray_scott/viz \
    --tag ep19

echo "=== DONE ==="
