"""PCA of Gray-Scott states in image space vs JEPA latent space, colored by phase.

Samples frames from each phase (gliders/bubbles/maze/spirals/spots/worms), runs a 2D PCA
on (a) the flattened z-scored fields and (b) the JEPA encoder latent (avg-pooled), and
scatters PC1 vs PC2 colored by phase. Shows whether the learned latent organizes the
6 reaction-diffusion regimes more cleanly than raw pixels.

Run: python -m examples.gray_scott.plot_pca --ckpt <jepa.pth.tar> [--per_regime 20] [--fpc 6]
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

from examples.gray_scott.eval import load_jepa, C
from examples.gray_scott.eval_common import build_regime_clips


@torch.no_grad()
def encode_latents(encoder, frames, device, bs=32, pool=4):
    """frames [N,2,Hs,Ws] -> avg-pooled latent, flattened [N, 16*(Hs/pool)*(Ws/pool)]."""
    out = []
    for i in range(0, frames.shape[0], bs):
        x = frames[i:i + bs].to(device).float().unsqueeze(2)   # [B,2,1,H,W]
        z = encoder(x).squeeze(2)                              # [B,16,H,W]
        z = F.avg_pool2d(z, pool)                              # [B,16,H/p,W/p]
        out.append(z.flatten(1).cpu())
    return torch.cat(out, 0).numpy()


def main():
    a = sys.argv
    def opt(f, d, c=str):
        return c(a[a.index(f) + 1]) if f in a else d
    ckpt_path = a[a.index("--ckpt") + 1]
    per_regime = opt("--per_regime", 20, int)
    fpc = opt("--fpc", 6, int)              # frames sampled per clip
    split = opt("--split", "test")
    H = opt("--H", 30, int)
    out_dir = opt("--out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"]); stride = int(cfg.data.get("time_stride", 4))
    _, encoder = load_jepa(ckpt, device)

    clips, tags, titles = build_regime_clips(split, C + H, stride, per_regime)   # [Nc,2,T,H,W]
    T = clips.shape[2]
    idx = np.linspace(0, T - 1, fpc, dtype=int)
    # one point per (clip, sampled frame)
    frames = clips[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, 2, clips.shape[-2], clips.shape[-1])
    phase_per_clip = np.array([t.split()[0] for t in titles])
    phases = np.repeat(phase_per_clip, fpc)
    PHASES = list(dict.fromkeys(phase_per_clip.tolist()))
    print(f"[pca] {frames.shape[0]} points ({per_regime} clips x {fpc} frames x {len(PHASES)} phases)",
          flush=True)

    X_img = frames.flatten(1).float().numpy()                 # [N, 2*128*128]
    X_lat = encode_latents(encoder, frames, device)           # [N, 16*32*32]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    cmap = plt.get_cmap("tab10")
    cols = {p: cmap(i) for i, p in enumerate(PHASES)}
    for ax, (X, name) in zip(axes, [(X_img, "image space (raw fields)"),
                                    (X_lat, "JEPA latent space (encoder)")]):
        pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
        Z = pca.fit_transform(X)
        for p in PHASES:
            m = phases == p
            ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.6, color=cols[p], label=p, edgecolors="none")
        ev = pca.explained_variance_ratio_
        ax.set_title(f"PCA — {name}\nPC1 {ev[0]*100:.0f}%  PC2 {ev[1]*100:.0f}%", fontsize=12)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(title="phase", fontsize=9, markerscale=1.6, frameon=False)
    fig.suptitle(f"Gray-Scott PCA: image vs latent  (epoch {ckpt.get('epoch')}, {split})", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "pca_image_latent.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[pca] wrote {path}\n[pca] DONE", flush=True)


if __name__ == "__main__":
    main()
