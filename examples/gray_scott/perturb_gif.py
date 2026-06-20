"""Generate GIFs showing JEPA prediction under A-channel perturbations.

Takes real clips, adds epsilon to z-scored A in the first context frame,
rolls JEPA forward, and decodes. Output: 5-panel GIF per sample:
  Truth | Original | ε=0.5 | ε=1.0 | ε=2.0

This answers: does the encoder latch onto the PDE regime or the initial conditions?
If the GIF barely changes across epsilon, the encoder is regime-focused (good for JEPA).
If it changes dramatically, initial conditions matter more.

Usage:
  uv run python -m examples.gray_scott.perturb_gif --ckpt <...> --outdir examples/gray_scott/viz
"""
import re
import os
import argparse
import glob

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import NT, MEAN, STD, ROOT
from examples.gray_scott.eval import C, load_jepa, build_decoder, rollout_latents

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILONS = [0.0, 0.5, 1.0, 2.0]   # in z-scored A units (σ ≈ 0.24 physical)


def _parse_fk(path):
    m = re.search(r'_F_([\d.]+)_k_([\d.]+)\.hdf5', path)
    return float(m.group(1)), float(m.group(2))


def _parse_regime(path):
    m = re.search(r'diffusion_([a-z]+)_F_', os.path.basename(path))
    return m.group(1) if m else "unknown"


def _load_clip(f, ntraj, span, stride, rng):
    tr = int(rng.integers(ntraj))
    t0 = int(rng.integers(0, max(1, NT - span + 1)))
    sl = slice(t0, t0 + span, stride)
    A = f["t0_fields/A"][tr, sl]
    B = f["t0_fields/B"][tr, sl]
    x = np.stack([A, B], axis=0).astype(np.float32)
    return (x - MEAN[:, None, None, None]) / STD[:, None, None, None]


def _denorm(arr):
    out = arr.copy()
    for ch in (0, 1):
        out[ch] = arr[ch] * STD[ch] + MEAN[ch]
    return out


def _norm01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _rgb(fields, scale):
    (loA, hiA), (loB, hiB) = scale
    A = _norm01(fields[0], loA, hiA)
    B = _norm01(fields[1], loB, hiB)
    return np.stack([A, B, np.zeros_like(A)], axis=-1)


@torch.no_grad()
def make_perturb_gif(x_base, jepa, decoder, H, regime, fk, outdir, tag, fps=8):
    """x_base: [1, 2, C+H, Hs, Ws] — one clip, z-scored."""
    x_base = x_base.to(DEVICE)

    # Build truth: [2, H, Hs, Ws] denormed
    truth = _denorm(x_base[0, :, C:C+H].cpu().numpy())

    scale = [(float(truth[0].min()), float(truth[0].max())),
             (float(truth[1].min()), float(truth[1].max()))]

    panels = [("Truth (R=A G=B)", _rgb(truth, scale), {})]
    latent_diffs = {}

    z0 = None
    for eps in EPSILONS:
        x_pert = x_base.clone()
        x_pert[:, 0, :C] += eps    # add eps to A channel in all context frames

        pred_z = rollout_latents(jepa, x_pert, H, DEVICE)  # [1,D,C+H,h,w]
        if z0 is None:
            z0 = pred_z.clone()
        else:
            rel = ((pred_z - z0).norm() / z0.norm().clamp(1e-8)).item()
            latent_diffs[eps] = rel

        pred_fields = decoder(pred_z[:, :, C:])  # [1,2,H,Hs,Ws]
        pred = _denorm(pred_fields[0].cpu().numpy())  # [2,H,Hs,Ws]

        label = f"ε={eps:.1f}" if eps > 0 else "Original"
        panels.append((label, _rgb(pred, scale), {}))

    # GIF
    N = len(panels)
    T = H
    fig, axes = plt.subplots(1, N, figsize=(3*N, 3.4))
    ims = []
    for ax, (label, data, render) in zip(axes, panels):
        im = ax.imshow(data[0], **render)
        ax.set_title(label, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        ims.append((im, data))
    sup = fig.suptitle("", fontsize=10)

    def update(f):
        for im, data in ims:
            im.set_data(data[f])
        sup.set_text(f"{regime} (F={fk[0]:.3f}, k={fk[1]:.3f})  t={f+1}/{T}")
        return [im for im, _ in ims] + [sup]

    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
    out = os.path.join(outdir, f"perturb_{regime}_{tag}.gif")
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out}", flush=True)

    # Latent difference summary
    for eps, rd in latent_diffs.items():
        print(f"  {regime:10s} ε={eps:.1f} → ||Δz||/||z||={rd:.4f}", flush=True)
    return latent_diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--n-frames", type=int, default=4)  # C + lookahead (C=2 used for JEPA)
    ap.add_argument("--time-stride", type=int, default=4)
    ap.add_argument("--outdir", default="examples/gray_scott/viz")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    jepa, _ = load_jepa(ckpt, DEVICE)
    dstc = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    decoder = build_decoder(dstc, DEVICE, ckpt_path=args.ckpt)
    ep = ckpt.get("epoch", "?")
    tag = args.tag or f"ep{ep}_s{args.time_stride}"
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[perturb-gif] epoch={ep}, H={args.H}, stride={args.time_stride}", flush=True)

    files = sorted(glob.glob(os.path.join(ROOT, "data", args.split, "*.hdf5")))
    rng = np.random.default_rng(7)
    span = (C + args.H - 1) * args.time_stride + 1

    print("\nLatent relative changes (||Δz||/||z||) by regime and epsilon:")
    for path in files:
        fk = _parse_fk(path)
        regime = _parse_regime(path)
        with h5py.File(path, "r") as f:
            ntraj = f["t0_fields/A"].shape[0]
            x_np = _load_clip(f, ntraj, span, args.time_stride, rng)
        x = torch.from_numpy(x_np).unsqueeze(0)   # [1,2,T,H,W]
        make_perturb_gif(x, jepa, decoder, args.H, regime, fk,
                         args.outdir, tag, fps=args.fps)

    print("[perturb-gif] done", flush=True)


if __name__ == "__main__":
    main()
