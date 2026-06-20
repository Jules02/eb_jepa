#!/bin/bash
#SBATCH --job-name=gs_viz_v2
#SBATCH --output=/lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/slurm_viz_withv2_%j.out
#SBATCH --time=60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-jepadormi

source /lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/env.sh
cd /lustre/work/vivatech-jepadormi/aduplessi/eb_jepa

CKPT_SMALL=/lustre/work/vivatech-jepadormi/aduplessi/checkpoints/gray_scott/dev/epoch_19.pth.tar
CKPT_V2=/lustre/work/vivatech-jepadormi/abenmanso/checkpoints/gray_scott/vjepa_v2/epoch_10.pth.tar

echo "=== GIFs: Truth | JEPA-small | JEPA-v2 (abenmanso ep10) | UNetClassic | CNextU-Net ==="

export PYTHONPATH="/lustre/work/vivatech-jepadormi/aduplessi/the_well_repo:${PYTHONPATH:-}"

uv run --with "neuraloperator==0.3.0" --with torch-harmonics --with timm --with einops \
    python -m examples.gray_scott.visualize \
        --ckpt "$CKPT_SMALL" \
        --ckpt2 "$CKPT_V2" \
        --H 60 \
        --fps 10 \
        --n 4 \
        --time-stride 4 \
        --outdir examples/gray_scott/viz \
        --seed 42 \
        --baselines \
        --tag withv2

echo "=== DONE ==="
