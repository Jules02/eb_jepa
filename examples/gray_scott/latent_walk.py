"""Random-walk vs directed-walk analysis of Gray-Scott trajectories, per phase.

For each phase, encodes its trajectories and characterises the geometry of the path z_t
(latent) and x_t (image) with three scale-invariant diagnostics:
  - alpha : MSD(tau) ~ tau^alpha exponent   (1 = diffusive/random, 2 = ballistic/directed,
                                             <1 = trapped/converged)
  - C(1)  : consecutive-step direction autocorrelation (0 = random, 1 = directed) + persistence time
  - S     : straightness = net displacement / path length  (0 = random, 1 = straight line)

Spatial fields/latents are avg-pooled (keeps the moving-pattern motion, stays light).
Saves results/walk_data.npz + 3 figures.

Run: python -m examples.gray_scott.latent_walk --ckpt <jepa.pth.tar> [--per_regime 20] [--H 60]
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

from examples.gray_scott.eval import load_jepa, C
from examples.gray_scott.eval_common import build_regime_clips


@torch.no_grad()
def encode_traj_latent(encoder, clip, device, pool=4, bs=64):
    """clip [2,T,H,W] -> latent walk [T, 16*(H/pool)*(W/pool)]."""
    T = clip.shape[1]
    frames = clip.permute(1, 0, 2, 3).to(device).float()        # [T,2,H,W]
    out = []
    for i in range(0, T, bs):
        z = encoder(frames[i:i + bs].unsqueeze(2)).squeeze(2)   # [b,16,H,W]
        out.append(F.avg_pool2d(z, pool).flatten(1).cpu())
    return torch.cat(out, 0).numpy()


def image_traj(clip, pool=4):
    """clip [2,T,H,W] -> image walk [T, 2*(H/pool)*(W/pool)]."""
    x = clip.permute(1, 0, 2, 3).float()                        # [T,2,H,W]
    return F.avg_pool2d(x, pool).flatten(1).numpy()


def walk_metrics(Z):
    """Z [T,D] -> dict(alpha, C1, tpers, S, msd[T-1], Cc[maxlag], taus)."""
    T = Z.shape[0]
    taus = np.arange(1, T)
    msd = np.array([np.mean(np.sum((Z[t:] - Z[:-t]) ** 2, axis=1)) for t in taus])
    fit = (taus <= max(2, (T - 1) // 2)) & (msd > 0)
    alpha = float(np.polyfit(np.log(taus[fit]), np.log(msd[fit]), 1)[0]) if fit.sum() >= 2 else np.nan
    dz = np.diff(Z, axis=0)                                      # [T-1,D]
    s = np.linalg.norm(dz, axis=1)
    u = dz / (s[:, None] + 1e-12)
    maxlag = max(1, (T - 1) // 2)
    Cc = np.array([np.mean(np.sum(u[:-k] * u[k:], axis=1)) for k in range(1, maxlag + 1)])
    C1 = float(Cc[0]) if len(Cc) else np.nan
    below = np.where(Cc < 1.0 / np.e)[0]
    tpers = float(below[0] + 1) if len(below) else float(maxlag)
    S = float(np.linalg.norm(Z[-1] - Z[0]) / (s.sum() + 1e-12))
    return dict(alpha=alpha, C1=C1, tpers=tpers, S=S, msd=msd, Cc=Cc, taus=taus)


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
    print(f"[walk] {len(clips)} trajectories x T={C+H} frames over {PHASES}", flush=True)

    rec = {sp: {k: [] for k in ("alpha", "C1", "tpers", "S", "msd", "Cc")}
           for sp in ("latent", "image")}
    clip_phase = []
    for n in range(len(clips)):
        zl = encode_traj_latent(encoder, clips[n], device, pool)
        zi = image_traj(clips[n], pool)
        for space, Z in (("latent", zl), ("image", zi)):
            m = walk_metrics(Z)
            for k in ("alpha", "C1", "tpers", "S"):
                rec[space][k].append(m[k])
            rec[space]["msd"].append(m["msd"] / (m["msd"][0] + 1e-12))   # normalise to MSD(1)=1
            rec[space]["Cc"].append(m["Cc"])
        clip_phase.append(phases[n])
        if (n + 1) % 20 == 0:
            print(f"[walk] {n+1}/{len(clips)}", flush=True)
    clip_phase = np.array(clip_phase)
    for sp in rec:
        for k in rec[sp]:
            rec[sp][k] = np.array(rec[sp][k])
    taus = np.arange(1, C + H)

    np.savez(os.path.join(out_dir, "walk_data.npz"), phases=clip_phase, taus=taus,
             **{f"{sp}_{k}": rec[sp][k] for sp in rec for k in rec[sp]})
    print(f"[walk] saved {out_dir}/walk_data.npz", flush=True)

    cmap = plt.get_cmap("tab10")
    col = {p: cmap(i) for i, p in enumerate(PHASES)}

    # ---- table ----
    print("\nphase       | alpha_lat  C1_lat  S_lat | alpha_img  S_img  (mean over clips)")
    for p in PHASES:
        mk = clip_phase == p
        print(f"{p:11s} | {rec['latent']['alpha'][mk].mean():8.2f}  "
              f"{rec['latent']['C1'][mk].mean():6.2f}  {rec['latent']['S'][mk].mean():5.2f} | "
              f"{rec['image']['alpha'][mk].mean():8.2f}  {rec['image']['S'][mk].mean():5.2f}", flush=True)

    # ---- Fig 1: MSD log-log, latent | image ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, sp in zip(axes, ("latent", "image")):
        for p in PHASES:
            mk = clip_phase == p
            med = np.median(rec[sp]["msd"][mk], 0)
            al = np.nanmean(rec[sp]["alpha"][mk])
            ax.plot(taus, med, color=col[p], lw=1.8, label=f"{p}  (α={al:.2f})")
        ax.plot(taus, taus.astype(float), "k--", lw=0.8, alpha=0.5)
        ax.plot(taus, taus.astype(float) ** 2, "k:", lw=0.8, alpha=0.5)
        ax.text(taus[-1], taus[-1], " α=1 (random)", fontsize=7, color="0.4")
        ax.text(taus[len(taus)//2], (taus[len(taus)//2]) ** 2, " α=2 (directed)", fontsize=7, color="0.4")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("lag τ (steps)"); ax.set_ylabel("MSD(τ) / MSD(1)")
        ax.set_title(f"{sp} space"); ax.legend(fontsize=8)
    fig.suptitle(f"MSD scaling per phase  (epoch {ckpt.get('epoch')}, {split})", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "walk_msd.png"), dpi=140); plt.close(fig)
    print("[walk] wrote walk_msd.png", flush=True)

    # ---- Fig 2: direction autocorrelation C(τ) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, sp in zip(axes, ("latent", "image")):
        for p in PHASES:
            mk = clip_phase == p
            med = np.median(rec[sp]["Cc"][mk], 0)
            ax.plot(np.arange(1, len(med) + 1), med, color=col[p], lw=1.8, label=p)
        ax.axhline(0, color="0.7", lw=0.7); ax.axhline(1 / np.e, ls="--", color="0.7", lw=0.7)
        ax.set_xlabel("lag τ (steps)"); ax.set_ylabel("direction autocorr C(τ)")
        ax.set_title(f"{sp} space"); ax.legend(fontsize=8)
    fig.suptitle(f"Step-direction persistence per phase  (epoch {ckpt.get('epoch')})", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "walk_autocorr.png"), dpi=140); plt.close(fig)
    print("[walk] wrote walk_autocorr.png", flush=True)

    # ---- Fig 3: summary bars (alpha and S per phase, latent vs image) ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(PHASES)); w = 0.38
    for ax, key, ttl, refs in ((a1, "alpha", "MSD exponent α", True),
                               (a2, "S", "straightness S", False)):
        for j, sp in enumerate(("latent", "image")):
            vals = [np.nanmean(rec[sp][key][clip_phase == p]) for p in PHASES]
            err = [np.nanstd(rec[sp][key][clip_phase == p]) for p in PHASES]
            ax.bar(x + j * w, vals, w, yerr=err, capsize=2,
                   color=("tab:blue" if sp == "latent" else "tab:gray"), label=sp)
        if refs:
            ax.axhline(1, ls="--", color="0.6", lw=0.8); ax.text(x[-1], 1, " random", fontsize=8, color="0.5")
            ax.axhline(2, ls=":", color="0.6", lw=0.8); ax.text(x[-1], 2, " directed", fontsize=8, color="0.5")
        ax.set_xticks(x + w / 2); ax.set_xticklabels(PHASES, fontsize=10)
        ax.set_title(ttl); ax.legend(fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Random-walk vs directed-walk summary per phase  (epoch {ckpt.get('epoch')})", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "walk_summary.png"), dpi=140); plt.close(fig)
    print("[walk] wrote walk_summary.png\n[walk] DONE", flush=True)


if __name__ == "__main__":
    main()
