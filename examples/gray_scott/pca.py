"""PCA of JEPA encoder latents colored by Gray-Scott (F, k) parameters.

Encodes many clips from all regimes, pools each latent over (T,h,w) -> [N, D],
runs PCA, and produces scatter plots PC1 vs PC2 colored by regime / F / k.
Also reports Pearson correlation of each PC with F and k.

Usage:
  uv run python -m examples.gray_scott.pca --ckpt <epoch_19.pth.tar> --outdir examples/gray_scott/viz
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
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import NT, MEAN, STD, ROOT
from examples.gray_scott.eval import load_jepa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_fk(path):
    m = re.search(r'_F_([\d.]+)_k_([\d.]+)\.hdf5', path)
    return float(m.group(1)), float(m.group(2))


def _parse_regime(path):
    m = re.search(r'diffusion_([a-z]+)_F_', os.path.basename(path))
    return m.group(1) if m else "unknown"


@torch.no_grad()
def collect_latents(encoder, files, n_clips_per_file, n_frames, time_stride, batch_size=32):
    """Encode clips and pool latents. Returns (latents [N,D], fk [N,2], regimes [N])."""
    latents, fks, regimes = [], [], []
    rng = np.random.default_rng(0)
    span = (n_frames - 1) * time_stride + 1

    clips_buf, meta_buf = [], []
    def flush():
        if not clips_buf:
            return
        x = torch.stack(clips_buf).to(DEVICE)         # [B, 2, T, H, W]
        z = encoder(x)                                 # [B, D, T, h, w]
        latents.append(z.mean(dim=(2, 3, 4)).cpu())   # [B, D]
        for fk, regime in meta_buf:
            fks.append(fk)
            regimes.append(regime)
        clips_buf.clear(); meta_buf.clear()

    for path in files:
        fk = _parse_fk(path)
        regime = _parse_regime(path)
        with h5py.File(path, "r") as f:
            ntraj = f["t0_fields/A"].shape[0]
            for _ in range(n_clips_per_file):
                tr = int(rng.integers(ntraj))
                t0 = int(rng.integers(0, max(1, NT - span + 1)))
                sl = slice(t0, t0 + span, time_stride)
                A = f["t0_fields/A"][tr, sl]
                B = f["t0_fields/B"][tr, sl]
                x = np.stack([A, B], axis=0).astype(np.float32)
                x = (x - MEAN[:, None, None, None]) / STD[:, None, None, None]
                clips_buf.append(torch.from_numpy(x))
                meta_buf.append((fk, regime))
                if len(clips_buf) >= batch_size:
                    flush()
    flush()

    latents = torch.cat(latents).numpy()              # [N, D]
    fks = np.array(fks, dtype=np.float32)             # [N, 2]
    return latents, fks, regimes


def run_pca(latents, n_components=10):
    """PCA via SVD. Returns (scores [N,K], components [K,D], explained_var [K])."""
    X = latents - latents.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, S.shape[0])
    scores = U[:, :k] * S[:k]
    var = S[:k] ** 2 / (latents.shape[0] - 1)
    total_var = (latents ** 2).sum() / (latents.shape[0] - 1) - (latents.mean(axis=0) ** 2).sum() / (latents.shape[0] - 1)
    # simpler: total variance = sum of all singular values squared / (N-1)
    total_var_correct = (S ** 2).sum() / (latents.shape[0] - 1)
    explained = var / total_var_correct
    return scores, Vt[:k], explained


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    return float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0


def make_pca_plots(scores, fks, regimes, explained, outdir, tag):
    """Three scatter plots of PC1 vs PC2: colored by regime, F, and k."""
    os.makedirs(outdir, exist_ok=True)

    pc1, pc2 = scores[:, 0], scores[:, 1]
    F_vals, k_vals = fks[:, 0], fks[:, 1]

    unique_regimes = sorted(set(regimes))
    cmap_cat = plt.get_cmap("tab10")
    regime_color = {r: cmap_cat(i) for i, r in enumerate(unique_regimes)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"PCA of JEPA encoder latents  [{tag}]\n"
        f"PC1={explained[0]*100:.1f}%  PC2={explained[1]*100:.1f}%  "
        f"PC3={explained[2]*100:.1f}%" if len(explained) > 2 else "",
        fontsize=12)

    # Panel 1: colored by regime
    ax = axes[0]
    for regime in unique_regimes:
        mask = np.array([r == regime for r in regimes])
        ax.scatter(pc1[mask], pc2[mask], label=regime,
                   color=regime_color[regime], alpha=0.6, s=20)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("Colored by regime"); ax.legend(fontsize=8, markerscale=2)

    # Panel 2: colored by F
    ax = axes[1]
    sc = ax.scatter(pc1, pc2, c=F_vals, cmap="plasma", alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax, label="F")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("Colored by F")

    # Panel 3: colored by k
    ax = axes[2]
    sc = ax.scatter(pc1, pc2, c=k_vals, cmap="viridis", alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax, label="k")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("Colored by k")

    out = os.path.join(outdir, f"pca_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)

    # Correlation table
    print(f"\n{'PC':>4}  {'var%':>6}  {'r(F)':>8}  {'r(k)':>8}")
    for i in range(min(scores.shape[1], 10)):
        rF = pearson(scores[:, i], F_vals)
        rk = pearson(scores[:, i], k_vals)
        print(f"  {i+1:2d}  {explained[i]*100:6.2f}%  {rF:8.4f}  {rk:8.4f}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--n-clips", type=int, default=200, help="clips per regime file")
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--time-stride", type=int, default=4)
    ap.add_argument("--outdir", default="examples/gray_scott/viz")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    _, encoder = load_jepa(ckpt, DEVICE)
    encoder.eval()
    D = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    ep = ckpt.get("epoch", "?")
    tag = args.tag or f"D{D}_ep{ep}_s{args.time_stride}"
    print(f"[pca] encoder D={D}, epoch={ep}, stride={args.time_stride}, split={args.split}", flush=True)

    files = sorted(glob.glob(os.path.join(ROOT, "data", args.split, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 in {ROOT}/data/{args.split}")
    print(f"[pca] {len(files)} files, {args.n_clips} clips/file -> "
          f"{len(files)*args.n_clips} total", flush=True)
    for p in files:
        fk = _parse_fk(p)
        r = _parse_regime(p)
        print(f"  {r:10s}  F={fk[0]:.4f}  k={fk[1]:.4f}", flush=True)

    latents, fks, regimes = collect_latents(
        encoder, files, args.n_clips, args.n_frames, args.time_stride)
    print(f"[pca] latents shape: {latents.shape}", flush=True)

    scores, components, explained = run_pca(latents, n_components=min(D, 10))
    print(f"[pca] explained variance: "
          + "  ".join(f"PC{i+1}={explained[i]*100:.1f}%" for i in range(min(5, len(explained)))),
          flush=True)

    make_pca_plots(scores, fks, regimes, explained, args.outdir, tag)
    print(f"[pca] done", flush=True)


if __name__ == "__main__":
    main()
