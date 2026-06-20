#!/bin/bash
#SBATCH --job-name=gs_pca
#SBATCH --output=/lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/slurm_pca_%j.out
#SBATCH --time=20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-jepadormi

source /lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/env.sh
cd /lustre/work/vivatech-jepadormi/aduplessi/eb_jepa

CKPT=/lustre/work/vivatech-jepadormi/aduplessi/checkpoints/gray_scott/dev/epoch_19.pth.tar

echo "=== PCA of JEPA-small latents ==="

uv run python -m examples.gray_scott.pca \
    --ckpt "$CKPT" \
    --split valid \
    --n-clips 300 \
    --n-frames 4 \
    --time-stride 4 \
    --outdir examples/gray_scott/viz \
    --tag small_ep19_s4

echo "=== DONE ==="
