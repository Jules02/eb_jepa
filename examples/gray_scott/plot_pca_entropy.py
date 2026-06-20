"""Latent PCA per phase, colored two ways (same projection):
  - pca_color_time.png    : by normalized time t            (like potential_pca_time.png)
  - pca_color_entropy.png : by the (A,B) concentration entropy of the underlying state

The entropy is the Shannon entropy of the joint (A,B) value histogram of each TRUE frame
(a physical order parameter). Comparing the two colorings shows whether the latent's
geometry is organized by *time* and/or by *state disorder*.

Run: python -m examples.gray_scott.plot_pca_entropy --ckpt <jepa.pth.tar> [--per_regime 20] [--H 60]
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

from examples.gray_scott.eval import load_jepa, C
from examples.gray_scott.eval_common import build_regime_clips
from examples.gray_scott.latent_walk import encode_traj_latent
from eb_jepa.datasets.gray_scott.dataset import MEAN, STD

_M = np.array(MEAN)[:, None, None, None]
_S = np.array(STD)[:, None, None, None]


def ab_entropy(frame_phys, bins=32):
    """frame_phys [2,Hs,Ws] physical -> Shannon entropy (nats) of the (A,B) value histogram."""
    A = frame_phys[0].ravel(); B = frame_phys[1].ravel()
    h, _, _ = np.histogram2d(A, B, bins=bins, range=[[0, 1], [0, 1]])
    p = h.ravel(); p = p[p > 0]; p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def main():
    a = sys.argv
    def opt(f, d, c=str):
        return c(a[a.index(f) + 1]) if f in a else d
    ckpt_path = a[a.index("--ckpt") + 1]
    per_regime = opt("--per_regime", 20, int)
    H = opt("--H", 60, int)
    split = opt("--split", "test")
    pool = opt("--pool", 4, int)
    out_dir = opt("--out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"]); stride = int(cfg.data.get("time_stride", 4))
    _, encoder = load_jepa(ckpt, device)
    clips, tags, titles = build_regime_clips(split, C + H, stride, per_regime)
    phases = np.array([t.split()[0] for t in titles])
    PHASES = list(dict.fromkeys(phases.tolist()))
    T = C + H
    tnorm = np.linspace(0.0, 1.0, T)
    print(f"[pcaE] {len(clips)} clips x T={T} over {PHASES}", flush=True)

    Zby = {p: [] for p in PHASES}; Eby = {p: [] for p in PHASES}
    for n in range(len(clips)):
        Z = encode_traj_latent(encoder, clips[n], device, pool)          # [T,D]
        phys = np.clip(clips[n].numpy() * _S + _M, 0, 1)                  # [2,T,Hs,Ws]
        ent = np.array([ab_entropy(phys[:, k]) for k in range(T)])        # [T]
        Zby[phases[n]].append(Z); Eby[phases[n]].append(ent)
        if (n + 1) % 20 == 0:
            print(f"[pcaE] {n+1}/{len(clips)}", flush=True)

    # precompute one PCA per phase (shared by both colorings)
    proj = {}
    for p in PHASES:
        Z = np.vstack(Zby[p])
        proj[p] = (PCA(2, svd_solver="randomized", random_state=0).fit_transform(Z),
                   np.tile(tnorm, len(Zby[p])), np.concatenate(Eby[p]))

    def make_fig(which, cmap, label, fname):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, p in zip(axes.ravel(), PHASES):
            Z2, tcol, ecol = proj[p]
            col = tcol if which == "time" else ecol
            sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=col, cmap=cmap, s=10, alpha=0.75)
            for k in range(min(6, len(Zby[p]))):
                seg = Z2[k * T:(k + 1) * T]
                ax.plot(seg[:, 0], seg[:, 1], color="0.5", lw=0.4, alpha=0.5)
            ax.set_title(p, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=axes, label=label, shrink=0.6)
        fig.suptitle(f"Latent PCA per phase colored by {label}  (epoch {ckpt.get('epoch')}, {split})",
                     fontsize=13)
        fig.savefig(os.path.join(out_dir, fname), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[pcaE] wrote {fname}", flush=True)

    make_fig("time", "viridis", "normalized time t", "pca_color_time.png")
    make_fig("entropy", "plasma", "(A,B) concentration entropy (nats)", "pca_color_entropy.png")
    print("[pcaE] DONE", flush=True)


if __name__ == "__main__":
    main()
