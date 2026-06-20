"""Is there a 'time potential' Phi(z) the latent dynamics descends to advance in time?

Per phase, fits a linear scalar Phi(z) ~ t (normalized time) on a TRAIN set of trajectories
and evaluates it on HELD-OUT trajectories. A good shared time-potential => Phi generalises
(high R2) and is MONOTONE along each trajectory (Spearman ~1, monotonicity ~1). Relaxation
regimes (spots/maze/bubbles/worms) should have one; oscillatory/propagating regimes
(spirals/gliders) cannot have a monotone global potential -> the test fails for them.

Visualises: time-colored PCA per phase, Phi(z_t)-vs-t collapse, and a summary bar chart.
Saves results/potential_*.png + potential_data.npz.

Run: python -m examples.gray_scott.latent_potential --ckpt <jepa.pth.tar> [--per_regime 20] [--H 60]
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

from examples.gray_scott.eval import load_jepa, C
from examples.gray_scott.eval_common import build_regime_clips
from examples.gray_scott.latent_walk import encode_traj_latent


def potential_fit(Zs, T, ntrain):
    """Zs: list of [T,D]. Fit Phi~t on first ntrain trajectories, eval on the rest.
    Returns (model, r2_test, spearman_test[], monotonicity_test[], Phi_all[list])."""
    t = np.linspace(0.0, 1.0, T)
    Xtr = np.vstack(Zs[:ntrain]); ytr = np.tile(t, ntrain)
    model = Ridge(alpha=1.0).fit(Xtr, ytr)
    Xte = np.vstack(Zs[ntrain:]); yte = np.tile(t, len(Zs) - ntrain)
    r2 = float(model.score(Xte, yte))
    sp, mono = [], []
    for Z in Zs[ntrain:]:
        phi = model.predict(Z)
        sp.append(float(spearmanr(phi, t).correlation))
        mono.append(float(np.mean(np.diff(phi) > 0)))
    phi_all = [model.predict(Z) for Z in Zs]
    return model, r2, np.array(sp), np.array(mono), phi_all


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
    t = np.linspace(0.0, 1.0, T)
    print(f"[pot] {len(clips)} trajectories x T={T} over {PHASES}", flush=True)

    # encode latents per phase
    Zby = {p: [] for p in PHASES}
    for n in range(len(clips)):
        Zby[phases[n]].append(encode_traj_latent(encoder, clips[n], device, pool))
        if (n + 1) % 20 == 0:
            print(f"[pot] encoded {n+1}/{len(clips)}", flush=True)

    res = {}
    for p in PHASES:
        Zs = Zby[p]
        ntr = max(1, len(Zs) // 2)
        model, r2, sp, mono, phi_all = potential_fit(Zs, T, ntr)
        res[p] = dict(r2=r2, sp=sp, mono=mono, phi=phi_all, Zs=Zs, ntr=ntr)

    np.savez(os.path.join(out_dir, "potential_data.npz"),
             phases=np.array(PHASES),
             **{f"{p}_r2": np.array(res[p]["r2"]) for p in PHASES},
             **{f"{p}_sp": res[p]["sp"] for p in PHASES},
             **{f"{p}_mono": res[p]["mono"] for p in PHASES})
    print(f"[pot] saved {out_dir}/potential_data.npz", flush=True)

    print("\nphase       | R2_test  Spearman  monotonicity   verdict")
    for p in PHASES:
        r2 = res[p]["r2"]; sm = np.nanmean(res[p]["sp"]); mo = res[p]["mono"].mean()
        verdict = "DIRECTED potential" if (sm > 0.8 and mo > 0.8) else \
                  ("partial" if sm > 0.5 else "NO monotone potential")
        print(f"{p:11s} | {r2:6.2f}   {sm:7.2f}   {mo:7.2f}        {verdict}", flush=True)

    cmap = plt.get_cmap("tab10")
    col = {p: cmap(i) for i, p in enumerate(PHASES)}

    # ---- Fig 1: time-colored PCA per phase ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, p in zip(axes.ravel(), PHASES):
        Z = np.vstack(res[p]["Zs"])
        tt = np.tile(t, len(res[p]["Zs"]))
        Z2 = PCA(n_components=2, svd_solver="randomized", random_state=0).fit_transform(Z)
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=tt, cmap="viridis", s=10, alpha=0.7)
        # draw a few trajectories as faint lines
        for k in range(min(6, len(res[p]["Zs"]))):
            seg = Z2[k * T:(k + 1) * T]
            ax.plot(seg[:, 0], seg[:, 1], color="0.5", lw=0.4, alpha=0.5)
        ax.set_title(f"{p}  (Spearman={np.nanmean(res[p]['sp']):.2f})", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, label="normalized time t", shrink=0.6)
    fig.suptitle(f"Latent PCA colored by time — is there a time gradient?  (epoch {ckpt.get('epoch')})",
                 fontsize=13)
    fig.savefig(os.path.join(out_dir, "potential_pca_time.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[pot] wrote potential_pca_time.png", flush=True)

    # ---- Fig 2: Phi(z_t) vs t collapse (held-out trajectories) ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, p in zip(axes.ravel(), PHASES):
        ntr = res[p]["ntr"]
        for phi in res[p]["phi"][ntr:]:                       # held-out only
            ax.plot(t, phi, color=col[p], lw=1.0, alpha=0.5)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)     # ideal Phi=t
        ax.set_title(f"{p}  (R²={res[p]['r2']:.2f}, mono={res[p]['mono'].mean():.2f})", fontsize=11)
        ax.set_xlabel("true normalized time t"); ax.set_ylabel("learned Φ(z_t)")
    fig.suptitle(f"Time potential Φ(z) vs t on held-out trajectories — monotone & collapsed?  "
                 f"(epoch {ckpt.get('epoch')})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "potential_collapse.png"), dpi=140)
    plt.close(fig)
    print("[pot] wrote potential_collapse.png", flush=True)

    # ---- Fig 3: summary bars ----
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(PHASES)); w = 0.27
    for j, (key, lab) in enumerate([("r2", "R² (held-out)"), ("sp", "Spearman"),
                                    ("mono", "monotonicity")]):
        if key == "r2":
            vals = [max(0.0, res[p]["r2"]) for p in PHASES]; err = [0] * len(PHASES)
        else:
            vals = [np.nanmean(res[p][key]) for p in PHASES]
            err = [np.nanstd(res[p][key]) for p in PHASES]
        ax.bar(x + j * w, vals, w, yerr=err, capsize=2, label=lab)
    ax.axhline(1.0, ls=":", color="0.7", lw=0.8)
    ax.axhline(0.5, ls="--", color="0.7", lw=0.8); ax.text(x[-1] + 0.5, 0.5, " chance (mono)", fontsize=8, color="0.5")
    ax.set_xticks(x + w); ax.set_xticklabels(PHASES, fontsize=11); ax.set_ylim(0, 1.05)
    ax.set_title(f"Time-potential quality per phase  (epoch {ckpt.get('epoch')}, held-out)")
    ax.legend(fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "potential_summary.png"), dpi=140)
    plt.close(fig)
    print("[pot] wrote potential_summary.png\n[pot] DONE", flush=True)


if __name__ == "__main__":
    main()
