#!/bin/bash
#SBATCH --job-name=gs_analysis
#SBATCH --output=/lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/slurm_analysis_%j.out
#SBATCH --time=30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-jepadormi

source /lustre/work/vivatech-jepadormi/aduplessi/eb_jepa/env.sh
cd /lustre/work/vivatech-jepadormi/aduplessi/eb_jepa

CKPT=/lustre/work/vivatech-jepadormi/aduplessi/checkpoints/gray_scott/dev/epoch_19.pth.tar

echo "=== Exp A+B+C: PCA/dynamics-probe/perturbation sensitivity ==="
uv run python -m examples.gray_scott.analysis \
    --ckpt "$CKPT" \
    --split valid \
    --n-clips 200 \
    --n-frames 4 \
    --time-stride 4 \
    --outdir examples/gray_scott/viz \
    --tag small_ep19

echo "=== Exp D: Perturbation GIFs ==="
uv run python -m examples.gray_scott.perturb_gif \
    --ckpt "$CKPT" \
    --split valid \
    --H 30 \
    --time-stride 4 \
    --fps 8 \
    --outdir examples/gray_scott/viz \
    --tag ep19

echo "=== DONE ==="
