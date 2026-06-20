"""Final PER-PHASE VRMSE table (mean ± std over clips) for JEPA + all baselines.

For each Gray-Scott phase (gliders/bubbles/maze/worms/spirals/spots) and each model, reports
The Well VRMSE (paper mean-of-ratios) at horizons h=1,15,30,60 as mean ± std ACROSS that
phase's clips. Same fixed clips + same metric for every model -> directly comparable.
Field baselines (persistence/linear/climatology/unet/fno) are floor-free; the JEPA carries
its decoder floor (reported on the last row).

Run: python -m examples.gray_scott.final_table --ckpt <jepa.pth.tar> [--per_regime 20] [--H 60]
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.gray_scott.eval import load_jepa, build_decoder, rollout_latents, C
from examples.gray_scott.eval_common import build_regime_clips, iter_batches
from examples.gray_scott.baselines import (
    FNO2d, load_or_train, step_model, step_persistence, step_linear, step_climatology)
from eb_jepa.architectures import ResUNet

EPS = 1e-7


def _vr(pred, true):
    """[B,2,Hs,Ws] -> [B] per-clip VRMSE (paper formula, mean over channels)."""
    mse = ((pred - true) ** 2).mean(dim=(-2, -1))             # [B,2]
    return torch.sqrt(mse / (true.var(dim=(-2, -1)) + EPS)).mean(dim=1)   # [B]


@torch.no_grad()
def per_clip_field(step_fn, clips, device, H, bs=8):
    out = []
    for xb in iter_batches(clips, bs):
        x = xb.to(device); ctx = x[:, :, :C].clone(); vr = []
        for h in range(H):
            pred = step_fn(ctx)
            vr.append(_vr(pred, x[:, :, C + h]))
            ctx = torch.cat([ctx[:, :, 1:], pred.unsqueeze(2)], dim=2)
        out.append(torch.stack(vr, dim=1))                    # [B,H]
    return torch.cat(out, dim=0).cpu().numpy()                # [N,H]


@torch.no_grad()
def per_clip_jepa(jepa, decoder, clips, device, H, bs=8):
    out = []
    for xb in iter_batches(clips, bs):
        x = xb.to(device)
        pf = decoder(rollout_latents(jepa, x, H, device)[:, :, C:])   # [B,2,H,Hs,Ws]
        out.append(torch.stack([_vr(pf[:, :, h], x[:, :, C + h]) for h in range(H)], dim=1))
    return torch.cat(out, dim=0).cpu().numpy()


@torch.no_grad()
def per_clip_floor(encoder, decoder, clips, device, H, bs=8):
    out = []
    for xb in iter_batches(clips, bs):
        x = xb.to(device); vr = []
        for h in range(H):
            true = x[:, :, C + h]
            rec = decoder(encoder(true.unsqueeze(2))).squeeze(2)
            vr.append(_vr(rec, true))
        out.append(torch.stack(vr, dim=1))
    return torch.cat(out, dim=0).cpu().numpy()


def main():
    a = sys.argv
    def opt(f, d, c=str):
        return c(a[a.index(f) + 1]) if f in a else d
    ckpt_path = a[a.index("--ckpt") + 1]
    H = opt("--H", 60, int)
    per_regime = opt("--per_regime", 20, int)
    split = opt("--split", "test")
    horizons = [int(x) for x in opt("--horizons", "1,15,30,60").split(",") if int(x) <= H]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"]); stride = int(cfg.data.get("time_stride", 4))
    jepa, encoder = load_jepa(ckpt, device)
    decoder = build_decoder(int(cfg.model.dstc), device, ckpt_path=ckpt_path)
    print(f"[final] ckpt epoch={ckpt.get('epoch')} stride={stride} H={H} "
          f"per_regime={per_regime} horizons={horizons}", flush=True)

    clips, tags, titles = build_regime_clips(split, C + H, stride, per_regime)
    phases = np.array([t.split()[0] for t in titles])         # phase per clip
    phase_order = list(dict.fromkeys(phases.tolist()))
    print(f"[final] {len(clips)} clips over {len(phase_order)} phases: {phase_order}", flush=True)

    unet = ResUNet(in_d=2 * C, h_d=32, out_d=2, norm="group").to(device)
    load_or_train("unet", unet, device, stride, 20)
    fno = FNO2d(in_c=2 * C, out_c=2, width=32, modes=16, n_layers=4).to(device)
    load_or_train("fno", fno, device, stride, 20)

    vr = {
        "jepa": per_clip_jepa(jepa, decoder, clips, device, H),
        "persistence": per_clip_field(step_persistence, clips, device, H),
        "linear": per_clip_field(step_linear, clips, device, H),
        "climatology": per_clip_field(step_climatology, clips, device, H),
        "unet": per_clip_field(step_model(unet), clips, device, H),
        "fno": per_clip_field(step_model(fno), clips, device, H),
        "floor": per_clip_floor(encoder, decoder, clips, device, H),
    }
    models = ["jepa", "unet", "fno", "floor"]   # persistence/linear/climatology dropped

    def block(name, mask):
        n = int(mask.sum()) if mask is not None else len(clips)
        print(f"\n=== {name}  (n={n} clips) ===", flush=True)
        print(f"{'model':12s} " + "  ".join(f"{('h' + str(h)):>13s}" for h in horizons))
        for m in models:
            rows = vr[m] if mask is None else vr[m][mask]
            cells = [f"{rows[:, h-1].mean():.3f}±{rows[:, h-1].std():.3f}" for h in horizons]
            print(f"{m:12s} " + "  ".join(f"{c:>13s}" for c in cells))

    for ph in phase_order:
        block(ph, phases == ph)
    block("ALL PHASES", None)
    print("\n[final] DONE  (mean±std over clips; field baselines are floor-free)", flush=True)


if __name__ == "__main__":
    main()
