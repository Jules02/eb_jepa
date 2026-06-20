"""Latent space analysis for Gray-Scott JEPA encoder.

Three experiments:
  A. PC1 vs PC4 scatter — shows F/k encoded in minor PCA component
  B. Dynamics-filtered probe — pool only high-temporal-variance spatial positions,
     then train linear probe: tests if static regions are the bottleneck
  C. Perturbation latent sensitivity — add epsilon to A-channel first frame,
     measure how much the pooled latent changes (encoder robustness vs sensitivity)

Usage:
  uv run python -m examples.gray_scott.analysis --ckpt <epoch_19.pth.tar>
"""
import re
import os
import argparse
import glob

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import NT, MEAN, STD, ROOT
from examples.gray_scott.eval import load_jepa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REGIMES = ["bubbles", "gliders", "maze", "spirals", "spots", "worms"]
CMAP_CAT = plt.get_cmap("tab10")


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


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x**2).sum() * (y**2).sum())
    return float(np.dot(x, y) / d) if d > 1e-12 else 0.0


# ── Data collection ────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_all(encoder, files, n_clips, n_frames, stride, batch_size=32):
    """Collect pooled latents AND spatial latents (for dynamics masking).

    Returns:
      z_pool  [N, D]         — mean-pooled over (T, H, W)
      z_spat  [N, D, H, W]  — mean-pooled over T only, full spatial (sampled)
      x_clips [N, 2, T, H, W] — raw clips (z-scored) for variance computation
      fks     [N, 2]
      regimes [N] (strings)
    """
    rng = np.random.default_rng(0)
    span = (n_frames - 1) * stride + 1

    clips_buf, meta_buf = [], []
    z_pool_list, z_spat_list, x_list = [], [], []
    fks, regimes = [], []

    def flush():
        if not clips_buf:
            return
        x = torch.stack(clips_buf).to(DEVICE)    # [B, 2, T, H, W]
        z = encoder(x)                            # [B, D, T, H, W]
        z_pool_list.append(z.mean(dim=(2, 3, 4)).cpu().float())   # [B, D]
        z_spat_list.append(z.mean(dim=2).cpu().float())           # [B, D, H, W]
        x_list.append(x.cpu().float())
        for fk, regime in meta_buf:
            fks.append(fk); regimes.append(regime)
        clips_buf.clear(); meta_buf.clear()

    for path in files:
        fk = _parse_fk(path)
        regime = _parse_regime(path)
        with h5py.File(path, "r") as f:
            ntraj = f["t0_fields/A"].shape[0]
            for _ in range(n_clips):
                x_np = _load_clip(f, ntraj, span, stride, rng)
                clips_buf.append(torch.from_numpy(x_np))
                meta_buf.append((fk, regime))
                if len(clips_buf) >= batch_size:
                    flush()
    flush()

    z_pool = torch.cat(z_pool_list).numpy()    # [N, D]
    z_spat = torch.cat(z_spat_list).numpy()    # [N, D, H, W]
    x_arr  = torch.cat(x_list).numpy()         # [N, 2, T, H, W]
    fks    = np.array(fks, dtype=np.float32)
    return z_pool, z_spat, x_arr, fks, regimes


# ── Experiment A: better PCA ───────────────────────────────────────────────────

def run_pca(latents, n_components=10):
    X = latents - latents.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, S.shape[0])
    scores = U[:, :k] * S[:k]
    total_var = (S ** 2).sum() / (latents.shape[0] - 1)
    var = S[:k] ** 2 / (latents.shape[0] - 1)
    return scores, Vt[:k], var / total_var


def plot_pca_pc1_pc4(scores, fks, regimes, explained, outdir, tag):
    F_vals, k_vals = fks[:, 0], fks[:, 1]
    unique_regimes = sorted(set(regimes))
    regime_color = {r: CMAP_CAT(i) for i, r in enumerate(unique_regimes)}

    # Compute per-regime centroid in (PC1, PC4) space
    centroids = {}
    for r in unique_regimes:
        mask = np.array([x == r for x in regimes])
        centroids[r] = (scores[mask, 0].mean(), scores[mask, 3].mean())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"JEPA-small latent PCA  [{tag}]", fontsize=12)

    for ax_idx, (xi, yi, xlabel, ylabel) in enumerate([
        (0, 3, f"PC1 ({explained[0]*100:.1f}%)", f"PC4 ({explained[3]*100:.1f}%)"),
        (0, 1, f"PC1 ({explained[0]*100:.1f}%)", f"PC2 ({explained[1]*100:.1f}%)"),
        (0, 3, f"PC1 ({explained[0]*100:.1f}%)", f"PC4 ({explained[3]*100:.1f}%)"),
    ]):
        ax = axes[ax_idx]
        if ax_idx < 2:
            for r in unique_regimes:
                mask = np.array([x == r for x in regimes])
                ax.scatter(scores[mask, xi], scores[mask, yi],
                           label=r, color=regime_color[r], alpha=0.5, s=15)
            # label centroids
            for r, (cx, cy) in centroids.items():
                ax.annotate(r, (cx, cy), fontsize=8, fontweight="bold",
                            ha="center", va="bottom",
                            color=regime_color[r])
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title("Colored by regime")
            if ax_idx == 0:
                ax.legend(fontsize=7, markerscale=2)
        else:
            # Panel 3: PC1 vs PC4, colored by f+k
            fk_sum = F_vals + k_vals
            sc = ax.scatter(scores[:, xi], scores[:, yi], c=fk_sum,
                            cmap="plasma", alpha=0.6, s=15)
            plt.colorbar(sc, ax=ax, label="f + k")
            # label centroids
            for r, (cx, cy) in centroids.items():
                fk = fks[np.array([x == r for x in regimes])][0]
                ax.annotate(f"{r}\n(f+k={fk[0]+fk[1]:.3f})",
                            (cx, cy), fontsize=7, ha="center", va="bottom")
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title("PC1 vs PC4, colored by f+k")

    out = os.path.join(outdir, f"pca_pc1pc4_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [A] wrote {out}", flush=True)

    # Correlation summary
    print("\n[A] PC correlations with F, k, f+k:")
    print(f"  {'PC':>4}  {'var%':>6}  {'r(F)':>8}  {'r(k)':>8}  {'r(f+k)':>8}")
    for i in range(min(scores.shape[1], 8)):
        rF  = pearson(scores[:, i], F_vals)
        rk  = pearson(scores[:, i], k_vals)
        rfk = pearson(scores[:, i], F_vals + k_vals)
        print(f"  {i+1:2d}  {explained[i]*100:6.2f}%  {rF:8.4f}  {rk:8.4f}  {rfk:8.4f}")

    # Centroid pairwise: latent dist vs (f,k) dist
    print("\n[A] Regime centroid distances (PC1+PC2+PC4 space) vs (f,k) Euclidean dist:")
    for i, r1 in enumerate(unique_regimes):
        for r2 in unique_regimes[i+1:]:
            c1, c2 = scores[np.array([x == r1 for x in regimes])][:, [0,1,3]].mean(0), \
                     scores[np.array([x == r2 for x in regimes])][:, [0,1,3]].mean(0)
            fk1 = fks[np.array([x == r1 for x in regimes])][0]
            fk2 = fks[np.array([x == r2 for x in regimes])][0]
            lat_d = np.linalg.norm(c1 - c2)
            fk_d  = np.linalg.norm(fk1 - fk2)
            print(f"  {r1:8s}-{r2:8s}  latent_dist={lat_d:.3f}  fk_dist={fk_d:.4f}")


# ── Experiment B: dynamics-filtered probe ─────────────────────────────────────

def r2(pred, target):
    ss_res = ((pred - target) ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-8)


def dynamics_filtered_probe(z_spat, x_arr, fks, regimes, outdir, tag,
                             quantile=0.5, n_steps=1000, lr=1e-3):
    """Pool latent from dynamic (high temporal-variance) spatial positions only."""
    print(f"\n[B] Dynamics-filtered probe (top {int((1-quantile)*100)}% temporal-variance positions):")

    # Temporal variance of input field: [N, 2, T, H, W] -> var over T -> [N, 2, H, W]
    field_var = x_arr.var(axis=2).mean(axis=1)   # [N, H, W]  (mean over channels)

    # Per-sample threshold
    thresholds = np.quantile(field_var.reshape(field_var.shape[0], -1),
                             quantile, axis=1)      # [N]

    # Pool only high-var positions: z_spat [N, D, H, W]
    z_dyn_list = []
    n_active = []
    for i in range(z_spat.shape[0]):
        mask = field_var[i] > thresholds[i]        # [H, W] bool
        n_active.append(int(mask.sum()))
        z_i = z_spat[i][:, mask]                   # [D, n_active]
        z_dyn_list.append(z_i.mean(axis=1))        # [D]
    z_dyn = np.stack(z_dyn_list)                   # [N, D]
    print(f"  active positions per clip: {np.mean(n_active):.0f} / {field_var.shape[1]*field_var.shape[2]}", flush=True)

    # Also run baseline (all positions)
    z_all = z_spat.mean(axis=(2, 3))              # [N, D] — same as original pool

    results = {}
    for label, Z in [("all_positions", z_all), ("dynamic_only", z_dyn)]:
        # 80/20 train/val split
        N = Z.shape[0]
        idx = np.random.default_rng(1).permutation(N)
        tr_idx, va_idx = idx[:int(0.8*N)], idx[int(0.8*N):]

        Z_tr, fk_tr = Z[tr_idx], fks[tr_idx]
        Z_va, fk_va = Z[va_idx], fks[va_idx]

        # Normalize
        mu, sigma = Z_tr.mean(0), Z_tr.std(0).clip(1e-8)
        Z_tr_n = (Z_tr - mu) / sigma
        Z_va_n = (Z_va - mu) / sigma

        # Linear probe
        D = Z.shape[1]
        probe = nn.Linear(D, 2)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        Zt = torch.from_numpy(Z_tr_n).float()
        ft = torch.from_numpy(fk_tr).float()
        for _ in range(n_steps):
            loss = nn.functional.mse_loss(probe(Zt), ft)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            pred_va = probe(torch.from_numpy(Z_va_n).float()).numpy()
        r2s = r2(pred_va, fk_va)
        mae = np.abs(pred_va - fk_va).mean(axis=0)
        results[label] = (r2s, mae)
        print(f"  [{label}]  R²(F)={r2s[0]:.4f}  R²(k)={r2s[1]:.4f}  "
              f"MAE(F)={mae[0]:.5f}  MAE(k)={mae[1]:.5f}", flush=True)

    return results


# ── Experiment C: perturbation sensitivity ─────────────────────────────────────

@torch.no_grad()
def perturbation_sensitivity(encoder, files, n_frames, stride, outdir, tag,
                             epsilons=(0.2, 0.5, 1.0, 2.0)):
    """Add epsilon to A channel of first frame, measure latent change per regime."""
    print(f"\n[C] Perturbation sensitivity (add ε to z-scored A in frame 0):")
    rng = np.random.default_rng(42)
    span = (n_frames - 1) * stride + 1
    n_clips_per_regime = 50

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Encoder sensitivity to A-channel perturbation  [{tag}]", fontsize=12)

    all_results = {}
    for ri, path in enumerate(files):
        fk = _parse_fk(path)
        regime = _parse_regime(path)
        color = CMAP_CAT(ri)

        clips = []
        with h5py.File(path, "r") as f:
            ntraj = f["t0_fields/A"].shape[0]
            for _ in range(n_clips_per_regime):
                clips.append(torch.from_numpy(_load_clip(f, ntraj, span, stride, rng)))
        x = torch.stack(clips).to(DEVICE)   # [B, 2, T, H, W]

        # Baseline latent
        z0 = encoder(x).mean(dim=(2, 3, 4)).cpu().float()  # [B, D]

        eps_deltas = []
        for eps in epsilons:
            x_pert = x.clone()
            x_pert[:, 0, 0] += eps         # perturb A in frame 0 only
            z_eps = encoder(x_pert).mean(dim=(2, 3, 4)).cpu().float()
            rel_change = ((z_eps - z0).norm(dim=1) / z0.norm(dim=1).clamp(1e-8)).mean().item()
            eps_deltas.append(rel_change)

        all_results[regime] = (epsilons, eps_deltas, fk)
        axes[0].plot(epsilons, eps_deltas, "o-", label=f"{regime} (F={fk[0]:.3f},k={fk[1]:.3f})",
                     color=color)

    axes[0].set_xlabel("ε (added to z-scored A, frame 0)"); axes[0].set_ylabel("||Δz|| / ||z||")
    axes[0].set_title("Relative latent change vs perturbation size")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Panel 2: sensitivity at ε=1.0 vs f+k
    eps_ref = 1.0
    eps_idx = list(epsilons).index(eps_ref) if eps_ref in epsilons else len(epsilons) // 2
    fk_sums, sensitivities, labels = [], [], []
    for regime, (eps_list, deltas, fk) in all_results.items():
        fk_sums.append(fk[0] + fk[1])
        sensitivities.append(deltas[eps_idx])
        labels.append(regime)

    axes[1].scatter(fk_sums, sensitivities, c=[CMAP_CAT(i) for i in range(len(labels))], s=80)
    for i, lbl in enumerate(labels):
        axes[1].annotate(lbl, (fk_sums[i], sensitivities[i]), fontsize=9,
                         xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("f + k (parameter sum)"); axes[1].set_ylabel(f"||Δz|| / ||z|| at ε={eps_ref}")
    axes[1].set_title("Does sensitivity correlate with dynamics speed (f+k)?")
    axes[1].grid(True, alpha=0.3)

    r_sens_fk = pearson(np.array(fk_sums), np.array(sensitivities))
    print(f"  Pearson r(sensitivity, f+k) = {r_sens_fk:.4f}", flush=True)
    for regime, (eps_list, deltas, fk) in all_results.items():
        print(f"  {regime:10s}  f+k={fk[0]+fk[1]:.4f}  "
              + "  ".join(f"ε={e:.1f}→{d:.4f}" for e, d in zip(eps_list, deltas)), flush=True)

    out = os.path.join(outdir, f"perturbation_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [C] wrote {out}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n-clips", type=int, default=200)
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
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[analysis] D={D}, epoch={ep}, stride={args.time_stride}, split={args.split}", flush=True)

    files = sorted(glob.glob(os.path.join(ROOT, "data", args.split, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 in {ROOT}/data/{args.split}")
    print(f"[analysis] {len(files)} regime files, {args.n_clips} clips/file", flush=True)

    # Collect all latents
    print("\n[collect] encoding clips...", flush=True)
    z_pool, z_spat, x_arr, fks, regimes = collect_all(
        encoder, files, args.n_clips, args.n_frames, args.time_stride)
    print(f"[collect] z_pool={z_pool.shape}  z_spat={z_spat.shape}  x={x_arr.shape}", flush=True)

    # A: PCA
    print("\n=== Experiment A: PCA ===", flush=True)
    scores, _, explained = run_pca(z_pool, n_components=min(D, 10))
    plot_pca_pc1_pc4(scores, fks, regimes, explained, args.outdir, tag)

    # B: dynamics-filtered probe
    print("\n=== Experiment B: Dynamics-filtered probe ===", flush=True)
    dynamics_filtered_probe(z_spat, x_arr, fks, regimes, args.outdir, tag)

    # C: perturbation sensitivity
    print("\n=== Experiment C: Perturbation sensitivity ===", flush=True)
    perturbation_sensitivity(encoder, files, args.n_frames, args.time_stride,
                             args.outdir, tag)

    print("\n[analysis] all done", flush=True)


if __name__ == "__main__":
    main()
