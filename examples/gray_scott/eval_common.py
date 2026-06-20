"""Shared eval machinery so the JEPA and the baselines are scored on EXACTLY the same
clips with EXACTLY the same metric (the only way the two tables are comparable).

Two fixes over the old per-loader / mean-of-ratios approach:
  1. A FIXED, deterministic eval set (same (file, traj, t0) clips for every model, cached
     to disk). The dataset normally draws random clips per __getitem__, so re-iterating a
     loader gives different samples each time — that made persistence read 1.3 in one run
     and 32 in another. Here every model iterates the same materialized tensor.
  2. POOLED VRMSE = sqrt( sum_samples MSE / sum_samples var ) per horizon/channel, instead
     of mean over per-sample sqrt(MSE/var). Pooling the denominator stops near-uniform
     Gray-Scott frames (tiny spatial variance) from blowing the average up to 30 / 200.
"""
import glob
import os

import numpy as np
import torch

from eb_jepa.datasets.gray_scott.dataset import MEAN, STD, NT, ROOT

C = 2                                                  # context length
WELL_WINDOWS = {"6:12": (5, 12), "13:30": (12, 30)}    # 0-indexed [start, end)


def fixed_eval_path(cache_dir, split, n_frames, stride, n_clips, seed):
    return os.path.join(
        cache_dir, f"gs_fixed_{split}_n{n_frames}_s{stride}_N{n_clips}_seed{seed}.pt")


def load_or_build_fixed_eval(split, n_frames, stride, n_clips, seed, cache_dir,
                             data_root=ROOT):
    """Deterministically pick n_clips (file, traj, t0) from `split`, load+z-score, cache.
    Returns a CPU float16 tensor [n_clips, 2, n_frames, 128, 128]."""
    import h5py
    os.makedirs(cache_dir, exist_ok=True)
    path = fixed_eval_path(cache_dir, split, n_frames, stride, n_clips, seed)
    if os.path.exists(path):
        print(f"[fixed-eval] loading {path}", flush=True)
        return torch.load(path)

    files = sorted(glob.glob(os.path.join(data_root, "data", split, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 in {data_root}/data/{split}")
    rng = np.random.default_rng(seed)               # deterministic clip choice
    span = (n_frames - 1) * stride + 1
    handles = {}
    def _h(p):
        if p not in handles:
            handles[p] = h5py.File(p, "r")
        return handles[p]
    ntraj = [_h(p)["t0_fields/A"].shape[0] for p in files]

    clips = []
    for _ in range(n_clips):
        fi = int(rng.integers(len(files)))
        tr = int(rng.integers(ntraj[fi]))
        t0 = int(rng.integers(0, NT - span + 1))
        sl = slice(t0, t0 + span, stride)
        f = _h(files[fi])
        A = f["t0_fields/A"][tr, sl]
        B = f["t0_fields/B"][tr, sl]
        x = np.stack([A, B], axis=0).astype(np.float32)             # [2,T,128,128]
        x = (x - MEAN[:, None, None, None]) / STD[:, None, None, None]
        clips.append(torch.from_numpy(x).half())
    out = torch.stack(clips)                                        # [N,2,T,128,128]
    for f in handles.values():
        f.close()
    tmp = f"{path}.tmp.{os.getpid()}"                               # atomic write (no race)
    torch.save(out, tmp)
    os.replace(tmp, path)
    print(f"[fixed-eval] built {tuple(out.shape)} (seed={seed}) -> {path}", flush=True)
    return out


def build_regime_clips(split, n_frames, stride, per_regime=1, seed=0, data_root=ROOT):
    """One (or more) clip per Gray-Scott regime file (gliders/bubbles/maze/worms/spirals/
    spots). The regime + (F,k) are parsed from the filename. Returns (clips[N,2,T,H,W]
    float16, tags[list], titles[list])."""
    import glob
    import re
    import h5py
    files = sorted(glob.glob(os.path.join(data_root, "data", split, "*.hdf5")))
    span = (n_frames - 1) * stride + 1
    rng = np.random.default_rng(seed)
    clips, tags, titles = [], [], []
    for p in files:
        m = re.search(r"diffusion_([a-z]+)_F_([0-9]+\.[0-9]+)_k_([0-9]+\.[0-9]+)",
                      os.path.basename(p))
        regime = m.group(1) if m else os.path.basename(p).split(".")[0]
        fk = f"F={m.group(2)} k={m.group(3)}" if m else ""
        with h5py.File(p, "r") as f:
            ntraj = f["t0_fields/A"].shape[0]
            for j in range(per_regime):
                tr = int(rng.integers(ntraj))
                t0 = int(rng.integers(0, NT - span + 1))
                sl = slice(t0, t0 + span, stride)
                A = f["t0_fields/A"][tr, sl]
                B = f["t0_fields/B"][tr, sl]
                x = np.stack([A, B], axis=0).astype(np.float32)
                x = (x - MEAN[:, None, None, None]) / STD[:, None, None, None]
                clips.append(torch.from_numpy(x).half())
                tags.append(regime if per_regime == 1 else f"{regime}_{j}")
                titles.append(f"{regime}  {fk}")
    return torch.stack(clips), tags, titles


def iter_batches(x, bs):
    for i in range(0, x.shape[0], bs):
        yield x[i:i + bs].float()


class MeanRatioVRMSE:
    """The Well's VRMSE (paper, App. E.3): per sample/frame sqrt(spatial_MSE / (spatial
    centered-var + eps)), then mean over samples. Mean-of-ratios -> directly comparable to
    The Well's reported numbers, but sensitive to near-uniform frames (tiny denominator)."""
    def __init__(self, H, NC=2, eps=1e-7):
        self.sum = np.zeros((H, NC))
        self.cnt = np.zeros(H)
        self.eps = eps

    def add(self, h, pred, true):                      # [B,2,Hs,Ws]
        mse = ((pred - true) ** 2).mean(dim=(-2, -1))  # [B,2]
        var = true.var(dim=(-2, -1))                   # [B,2] (centered, unbiased)
        pv = torch.sqrt(mse / (var + self.eps))        # [B,2]
        self.sum[h] += pv.sum(dim=0).double().cpu().numpy()
        self.cnt[h] += pred.shape[0]

    def scores(self):
        per_ch = self.sum / np.maximum(self.cnt[:, None], 1)        # [H,2]
        return {"all": per_ch.mean(axis=-1), "u": per_ch[:, 0], "v": per_ch[:, 1]}


class PooledVRMSE:
    """Aggregate MSE and variance across the whole eval set, THEN ratio (per horizon/ch).
    NOT the paper metric — a denominator-stable diagnostic (no per-frame blow-up)."""
    def __init__(self, H, NC=2):
        self.num = np.zeros((H, NC))
        self.den = np.zeros((H, NC))

    def add(self, h, pred, true):                      # [B,2,Hs,Ws]
        mse = ((pred - true) ** 2).mean(dim=(-2, -1))  # [B,2]
        var = true.var(dim=(-2, -1))                   # [B,2]
        self.num[h] += mse.sum(dim=0).double().cpu().numpy()
        self.den[h] += var.sum(dim=0).double().cpu().numpy()

    def scores(self):
        per_ch = np.sqrt(self.num / np.maximum(self.den, 1e-12))    # [H,2]
        return {"all": per_ch.mean(axis=-1), "u": per_ch[:, 0], "v": per_ch[:, 1]}


def make_vrmse(metric, H):
    """metric: 'vrmse' (The Well paper, default) | 'pooled' (stable diagnostic)."""
    return MeanRatioVRMSE(H) if metric == "vrmse" else PooledVRMSE(H)


def window_mean(arr, win):
    s, e = win
    return float(arr[s:min(e, arr.shape[0])].mean())


def default_cache_dir():
    base = os.environ.get("EBJEPA_CKPTS", os.path.join(os.getcwd(), "_ckpts"))
    return os.path.join(base, "gray_scott", "fixed_eval")
