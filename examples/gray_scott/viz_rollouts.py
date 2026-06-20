"""Render autoregressive rollouts (GIFs) for all models on a few fixed Gray-Scott clips.

For each chosen clip it rolls every model H steps and writes results/viz_<id>.gif, one panel
per model (ground truth, JEPA-decoded, U-Net, FNO, persistence, linear) animated over H frames.
Uses the SAME fixed eval clips as the VRMSE table, so the GIFs match the reported numbers.

U-Net / FNO are retrained here (their weights were not saved by baselines.py) — fast on a
GB200, qualitatively identical to the table's.

Run:  python -m examples.gray_scott.viz_rollouts --ckpt <jepa.pth.tar> --H 30 [--clips 0,200,399]
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from omegaconf import OmegaConf

from examples.gray_scott.eval import load_jepa, build_decoder, rollout_latents, C
from examples.gray_scott.eval_common import (
    load_or_build_fixed_eval, build_regime_clips, default_cache_dir)
from examples.gray_scott.baselines import (
    FNO2d, load_or_train, step_model, step_persistence, step_linear)
from eb_jepa.architectures import ResUNet


@torch.no_grad()
def field_seq(step_fn, x, H):
    """Autoregressive field rollout. x:[B,2,C+H,H,W] -> preds [B,2,H,H,W]."""
    ctx = x[:, :, :C].clone()
    out = []
    for _ in range(H):
        pred = step_fn(ctx)
        out.append(pred)
        ctx = torch.cat([ctx[:, :, 1:], pred.unsqueeze(2)], dim=2)
    return torch.stack(out, dim=2)


def to_rgb(f2, ranges):
    """[2,Hs,Ws] (A,B) -> [Hs,Ws,3] RGB: A=red, B=green (yellow where both high).
    ranges = ((a_min,a_max),(b_min,b_max)) from the clip's ground truth, fixed across
    frames/models so colors are comparable."""
    (a0, a1), (b0, b1) = ranges
    A = ((f2[0] - a0) / (a1 - a0 + 1e-8)).clamp(0, 1)
    B = ((f2[1] - b0) / (b1 - b0 + 1e-8)).clamp(0, 1)
    rgb = torch.stack([A, B, torch.zeros_like(A)], dim=-1)
    return rgb.detach().cpu().numpy()


def main():
    a = sys.argv
    def opt(flag, default, cast=str):
        return cast(a[a.index(flag) + 1]) if flag in a else default
    ckpt_path = a[a.index("--ckpt") + 1]
    H = opt("--H", 120, int)
    split = opt("--split", "test")
    epochs = opt("--epochs", 20, int)
    ch = opt("--channel", "rgb")                      # "rgb" (A=red,B=green) | "0" | "1"
    mode = opt("--mode", "regimes")                   # regimes (one GIF per phase) | clips
    per_regime = opt("--per_regime", 1, int)          # GIFs per regime in 'regimes' mode
    clip_ids = [int(c) for c in opt("--clips", "0,200,399").split(",")]
    out_dir = opt("--out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    stride = int(cfg.data.get("time_stride", 4))
    jepa, encoder = load_jepa(ckpt, device)
    decoder = build_decoder(int(cfg.model.dstc), device, ckpt_path=ckpt_path)
    print(f"[viz] jepa epoch={ckpt.get('epoch')} stride={stride} H={H} mode={mode}", flush=True)

    if mode == "regimes":
        # one clip per Gray-Scott phase (gliders/bubbles/maze/worms/spirals/spots)
        clips, tags, titles = build_regime_clips(split, C + H, stride, per_regime)
    else:
        clips = load_or_build_fixed_eval(split, C + H, stride, 400, 0, default_cache_dir())
        clips = clips[clip_ids]
        tags = [f"clip{c}" for c in clip_ids]
        titles = list(tags)
    x = clips.to(device).float()                      # [N,2,C+H,Hs,Ws]

    # trained baselines — load cached weights if present, else train + save once
    unet = ResUNet(in_d=2 * C, h_d=32, out_d=2, norm="group").to(device)
    load_or_train("unet", unet, device, stride, epochs)
    fno = FNO2d(in_c=2 * C, out_c=2, width=32, modes=16, n_layers=4).to(device)
    load_or_train("fno", fno, device, stride, epochs)

    # roll out every model (no_grad: the decoder/encoder params require grad otherwise)
    with torch.no_grad():
        seqs = {
            "truth": x[:, :, C:C + H],
            "jepa": decoder(rollout_latents(jepa, x, H, device)[:, :, C:]),
            "unet": field_seq(step_model(unet), x, H),
            "fno": field_seq(step_model(fno), x, H),
            "persistence": field_seq(step_persistence, x, H),
            "linear": field_seq(step_linear, x, H),
        }
    names = ["truth", "jepa", "unet", "fno", "persistence", "linear"]

    rgb = (ch == "rgb")
    cband = "R=A, G=B" if rgb else f"channel {'AB'[int(ch)]}"
    for i, (tag, title) in enumerate(zip(tags, titles)):
        gt = seqs["truth"][i]                          # [2,H,Hs,Ws]
        ranges = ((float(gt[0].min()), float(gt[0].max())),
                  (float(gt[1].min()), float(gt[1].max())))
        frames = []
        for h in range(H):
            fig, axes = plt.subplots(1, len(names), figsize=(2.0 * len(names), 2.4))
            for ax, nm in zip(axes, names):
                if rgb:
                    ax.imshow(to_rgb(seqs[nm][i, :, h], ranges))
                else:
                    c = int(ch)
                    ax.imshow(seqs[nm][i, c, h].detach().cpu().numpy(),
                              vmin=ranges[c][0], vmax=ranges[c][1], cmap="viridis")
                ax.set_title(nm, fontsize=9); ax.axis("off")
            fig.suptitle(f"{title}  |  frame {h + 1}/{H}  ({cband})", fontsize=10)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
        path = os.path.join(out_dir, f"viz_{tag}.gif")
        imageio.mimsave(path, frames, format="GIF", duration=0.16)
        print(f"[viz] wrote {path} ({H} frames)", flush=True)
    print("[viz] DONE", flush=True)


if __name__ == "__main__":
    main()
