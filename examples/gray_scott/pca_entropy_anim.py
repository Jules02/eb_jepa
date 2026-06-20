"""Compute latent PCA + per-frame (A,B) entropy per phase, save npz, build the animated GIF.

Saves outputs/pca_anim_data.npz (Z2_<phase> [k*T,2], ent_<phase> [k*T], T) — reusable for
re-styling the GIF without GPU — then calls gif_pca.build().

Run: python -m examples.gray_scott.pca_entropy_anim --ckpt <jepa.pth.tar> [--per_regime 12] [--H 60]
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

from examples.gray_scott.eval import load_jepa, C
from examples.gray_scott.eval_common import build_regime_clips
from examples.gray_scott.latent_walk import encode_traj_latent
from examples.gray_scott.plot_pca_entropy import ab_entropy
from examples.gray_scott.gif_pca import build, ORDER
from eb_jepa.datasets.gray_scott.dataset import MEAN, STD

_M = np.array(MEAN).reshape(2, 1, 1, 1)
_S = np.array(STD).reshape(2, 1, 1, 1)


def main():
    a = sys.argv
    def opt(f, d, c=str):
        return c(a[a.index(f) + 1]) if f in a else d
    ckpt_path = a[a.index("--ckpt") + 1]
    per_regime = opt("--per_regime", 12, int)
    H = opt("--H", 60, int)
    split = opt("--split", "test")
    pool = opt("--pool", 4, int)
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"]); stride = int(cfg.data.get("time_stride", 4))
    _, encoder = load_jepa(ckpt, device)
    clips, tags, titles = build_regime_clips(split, C + H, stride, per_regime, seed=0)
    phases = np.array([t.split()[0] for t in titles])
    T = C + H
    print(f"[anim] {len(clips)} clips x T={T}, per_regime={per_regime}", flush=True)

    data = {"T": T}
    for ph in ORDER:
        idxs = np.where(phases == ph)[0]
        Zs, ents = [], []
        for n in idxs:
            Zs.append(encode_traj_latent(encoder, clips[n], device, pool))      # [T,D]
            phys = np.clip(clips[n].numpy() * _S + _M, 0, 1)                     # [2,T,Hs,Ws]
            ents.append(np.array([ab_entropy(phys[:, k]) for k in range(T)]))    # [T]
        Z2 = PCA(2, svd_solver="randomized", random_state=0).fit_transform(np.vstack(Zs))
        data[f"Z2_{ph}"] = Z2.astype(np.float32)
        data[f"ent_{ph}"] = np.concatenate(ents).astype(np.float32)
        print(f"[anim] {ph} done", flush=True)

    np.savez("outputs/pca_anim_data.npz", **data)
    print("[anim] saved outputs/pca_anim_data.npz", flush=True)
    build("outputs/pca_anim_data.npz")
    print("[anim] DONE", flush=True)


if __name__ == "__main__":
    main()
