"""Render JEPA + ground-truth field rollouts for the 4 slide phases, save them, build GIFs.

Saves outputs/slides_fields.npz (float16, physical fields, reusable for re-styling the GIFs
without GPU), then builds outputs/gif_regimes.gif + outputs/gif_diff.gif.

Run: python -m examples.gray_scott.render_slides --ckpt <jepa.pth.tar> [--frames 36]
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.gray_scott.eval import load_jepa, build_decoder, rollout_latents, C
from examples.gray_scott.eval_common import build_regime_clips
from examples.gray_scott.gif_slides import build_both
from examples.gray_scott.baselines import FNO2d, load_or_train, step_model
from eb_jepa.architectures import ResUNet
from eb_jepa.datasets.gray_scott.dataset import MEAN, STD

_M = np.array(MEAN).reshape(2, 1, 1, 1)
_S = np.array(STD).reshape(2, 1, 1, 1)
WANT = ["gliders", "spirals", "bubbles", "worms"]


def _phys(t):                                  # tensor [1,2,Hg,Hs,Ws] -> physical np [2,Hg,Hs,Ws] f16
    return (t.detach().cpu().numpy()[0] * _S + _M).clip(0, 1).astype(np.float16)


def _field_roll(step, x, Hg):
    ctx = x[:, :, :C].clone(); out = []
    for _ in range(Hg):
        p = step(ctx); out.append(p); ctx = torch.cat([ctx[:, :, 1:], p.unsqueeze(2)], 2)
    return torch.stack(out, dim=2)             # [1,2,Hg,Hs,Ws]


def main():
    a = sys.argv
    def opt(f, d, c=str):
        return c(a[a.index(f) + 1]) if f in a else d
    ckpt_path = a[a.index("--ckpt") + 1]
    Hg = opt("--frames", 60, int)
    split = opt("--split", "test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"]); stride = int(cfg.data.get("time_stride", 4))
    jepa, encoder = load_jepa(ckpt, device)
    decoder = build_decoder(int(cfg.model.dstc), device, ckpt_path=ckpt_path)
    unet = ResUNet(in_d=2 * C, h_d=32, out_d=2, norm="group").to(device); load_or_train("unet", unet, device, stride, 20)
    fno = FNO2d(in_c=2 * C, out_c=2, width=32, modes=16, n_layers=4).to(device); load_or_train("fno", fno, device, stride, 20)
    # Build clips with the SAME frame budget as the original viz_rollouts (H=120) so the
    # sampled trajectory (t0) is identical; then keep only the first Hg frames.
    nbuild = C + max(120, Hg)
    clips, tags, titles = build_regime_clips(split, nbuild, stride, per_regime=1, seed=0)
    phase_of = [t.split()[0] for t in titles]
    print(f"[render] epoch={ckpt.get('epoch')} frames={Hg} (build {nbuild}) phases={WANT}", flush=True)

    data = {}
    for ph in WANT:
        x = clips[phase_of.index(ph):phase_of.index(ph) + 1].to(device).float()   # [1,2,nbuild,128,128]
        with torch.no_grad():
            data[f"truth_{ph}"] = _phys(x[:, :, C:C + Hg])
            data[f"jepa_{ph}"] = _phys(decoder(rollout_latents(jepa, x, Hg, device)[:, :, C:]))
            data[f"unet_{ph}"] = _phys(_field_roll(step_model(unet), x, Hg))
            data[f"fno_{ph}"] = _phys(_field_roll(step_model(fno), x, Hg))
        print(f"[render] {ph} done", flush=True)

    os.makedirs("outputs", exist_ok=True)
    np.savez("outputs/slides_fields.npz", **data)
    print("[render] saved outputs/slides_fields.npz", flush=True)
    build_both("outputs/slides_fields.npz")
    print("[render] DONE", flush=True)


if __name__ == "__main__":
    main()
