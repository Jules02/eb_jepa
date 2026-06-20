"""Evaluate pretrained The Well baselines (FNO/TFNO/UNet/CNextUNet) on gray_scott
with the paper's exact VRMSE metric (mean-of-ratios, per sample).

The Well paper uses TEACHER-FORCING evaluation: each step's prediction uses the
last C_hist=4 ground-truth frames as context (not model outputs). This matches
Table 3 of the paper.  stride=1 (native 10-second steps) as in their training.

Usage: uv run --with neuraloperator==0.3.0 --with torch-harmonics --with timm \\
           --with einops python examples/gray_scott/eval_baselines.py
"""
import numpy as np
import torch

from examples.gray_scott._well_baselines import stub_heavy_well_models

# Stub heavy unused models so the_well imports without needing timm etc.
stub_heavy_well_models()

from the_well.benchmark.models.fno import FNO
from the_well.benchmark.models.tfno import TFNO
from the_well.benchmark.models.unet_classic import UNetClassic
from the_well.benchmark.models.unet_convnext import UNetConvNext

from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NC = 2       # channels (A=inhibitor, B=activator)
C_HIST = 4   # The Well training config: n_steps_input=4 for all models
H = 30       # evaluation horizon (steps)
WELL_WINDOWS = {"6:12": (5, 12), "13:30": (12, 30)}
MODELS = {
    "FNO":        FNO,
    "TFNO":       TFNO,
    "UNetClassic": UNetClassic,
    "CNextU-Net": UNetConvNext,
}
HF_SUFFIX = "gray_scott_reaction_diffusion"


def load_model(name, cls):
    hf_name = {
        "FNO": "FNO", "TFNO": "TFNO",
        "UNetClassic": "UNetClassic", "CNextU-Net": "UNetConvNext",
    }[name]
    model = cls.from_pretrained(f"polymathic-ai/{hf_name}-{HF_SUFFIX}")
    return model.to(DEVICE).eval()


@torch.no_grad()
def eval_model_teacher_forcing(model, loader):
    """Teacher-forcing evaluation: each step uses ground-truth context.

    Matches The Well paper's Table 3 protocol exactly.
    For each trajectory frame t in [C_HIST, C_HIST+H):
      - input: ground-truth frames [t-C_HIST : t] → [B, C_HIST*NC, H, W]
      - output: model prediction for frame t
      - score: paper VRMSE vs ground truth frame t
    """
    psum = np.zeros((H, NC))
    pcnt = np.zeros(H)

    for batch in loader:
        x = batch["video"].to(DEVICE)          # [B, NC, T, H, W]
        B, _, T, Hs, Ws = x.shape
        x_cl = x.permute(0, 2, 3, 4, 1)       # [B, T, H, W, NC] channels-last

        for h in range(H):
            t = C_HIST + h
            if t >= T:
                break

            # Ground-truth context window
            ctx = x_cl[:, t - C_HIST:t]       # [B, C_HIST, H, W, NC]
            inp = ctx.permute(0, 1, 4, 2, 3).flatten(1, 2)  # [B, C_HIST*NC, H, W]

            pred = model(inp)                  # [B, NC, H, W]
            pred_cl = pred.permute(0, 2, 3, 1)  # [B, H, W, NC]
            true = x_cl[:, t]                  # [B, H, W, NC]

            mse = ((pred_cl - true) ** 2).mean(dim=(-3, -2))   # [B, NC]
            var = true.var(dim=(-3, -2))                         # [B, NC]
            vr = torch.sqrt(mse / (var + 1e-7))                 # [B, NC]
            psum[h] += vr.sum(dim=0).cpu().numpy()
            pcnt[h] += B

    per_ch = psum / np.maximum(pcnt[:, None], 1)   # [H, NC]
    return per_ch


def window(per_ch, start, end):
    end = min(end, H)
    return float(per_ch[start:end].mean())


def main():
    # stride=1 matches The Well paper training (native 10-second steps)
    dcfg = GrayScottConfig(split="valid", n_frames=C_HIST + H, time_stride=1,
                           epoch_size=400, batch_size=8, num_workers=8)
    loader = make_loader(dcfg, shuffle=False)

    print("Teacher-forcing eval at stride=1 — matches The Well Table 3")
    print(f"{'Model':15s}  {'h=1':>7}  {'[6:12]':>7}  {'[13:30]':>8}")
    print("-" * 50)

    for name, cls in MODELS.items():
        print(f"Loading {name}...", flush=True)
        try:
            model = load_model(name, cls)
            per_ch = eval_model_teacher_forcing(model, loader)   # [H, NC]
            avg = per_ch.mean(axis=-1)                           # [H]

            h1 = float(avg[0])
            w612 = window(avg, 5, 12)
            w1330 = window(avg, 12, 30)
            print(f"{name:15s}  {h1:7.3f}  {w612:7.3f}  {w1330:8.3f}", flush=True)
            print(f"  A h=1={per_ch[0,0]:.3f}  B h=1={per_ch[0,1]:.3f}", flush=True)
        except Exception as e:
            import traceback
            print(f"{name:15s}  ERROR: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
