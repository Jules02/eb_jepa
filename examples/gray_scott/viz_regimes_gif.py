"""Presentation GIF: the 6 Gray-Scott regimes animating beside the F-k phase diagram.

The Well's Gray-Scott split is 6 distinct (F, k) regimes, one HDF5 file each. This
builds a single looping GIF with, on the left, the classic Gray-Scott phase diagram
(feed rate F vs kill rate k) annotated with the 6 points the model was trained on,
and on the right a 2x3 grid of those regimes evolving in time, rendered in the same
green/red RGB style as eval_compare.py (R = chemical A, G = chemical B).

No torch / GPU needed — reads the HDF5 directly with h5py.

Run (from repo root, with the project venv active):
  python examples/gray_scott/viz_regimes_gif.py
  python examples/gray_scott/viz_regimes_gif.py --frames 80 --fps 12 --traj 0
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

DATA = ("/lustre/work/pdl17890/udl806719/datasets/the_well/"
        "gray_scott_reaction_diffusion/data")
_RE = re.compile(r"diffusion_([a-z]+)_F_([0-9.]+)_k_([0-9.]+)\.hdf5$")

# Plot order (roughly low->high feed F) and a stable colour per regime, shared
# between the phase-diagram marker and the panel title so the eye can link them.
ORDER = ["gliders", "spirals", "maze", "spots", "worms", "bubbles"]
COLORS = {
    "gliders": "#1f77b4", "spirals": "#9467bd", "maze": "#2ca02c",
    "spots": "#ff7f0e", "worms": "#d62728", "bubbles": "#8c564b",
}


def _norm01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _rgb(A, B):
    """[T,H,W] A,B -> [T,H,W,3] RGB: R=A, G=B (per-channel min/max), same as
    eval_compare.py's green/red composite. Blue stays 0."""
    R = _norm01(A, float(A.min()), float(A.max()))
    G = _norm01(B, float(B.min()), float(B.max()))
    return np.stack([R, G, np.zeros_like(R)], axis=-1)


def load(split, traj, frames, tmax):
    """Return {regime: (F, k, rgb[T,H,W,3])} sampling `frames` timesteps.

    Frames are drawn from t in [0, tmax]; a smaller tmax means smaller jumps
    between frames, so fast regimes (spirals/gliders/worms) animate smoothly
    instead of looking like they fast-forward. Each frame is the green/red
    R=A, G=B composite used in eval_compare.py.
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA, split, "*.hdf5"))):
        m = _RE.search(os.path.basename(p))
        if not m:
            continue
        name, F, k = m.group(1), float(m.group(2)), float(m.group(3))
        with h5py.File(p, "r") as f:
            nt = f["t0_fields/B"].shape[1]
            hi = nt - 1 if tmax <= 0 else min(tmax, nt - 1)
            ts = np.linspace(0, hi, frames).astype(int)
            A = np.asarray(f["t0_fields/A"][traj, ts])    # [T, 128, 128]
            B = np.asarray(f["t0_fields/B"][traj, ts])
        out[name] = (F, k, _rgb(A, B))
        print(f"  loaded {name:8s} F={F:<6} k={k:<6} -> {A.shape}", flush=True)
    return out


def build(data, out_path, fps, dpi):
    names = [n for n in ORDER if n in data]
    T = next(iter(data.values()))[2].shape[0]

    fig = plt.figure(figsize=(15, 6.2))
    fig.suptitle("Gray-Scott reaction-diffusion — the 6 regimes the model trains on",
                 fontsize=16, fontweight="bold", y=0.98)
    mosaic = [["phase", "phase", names[0], names[1], names[2]],
              ["phase", "phase", names[3], names[4], names[5]]]
    ax = fig.subplot_mosaic(mosaic, gridspec_kw=dict(
        width_ratios=[1, 1, 1, 1, 1], wspace=0.08, hspace=0.18))

    # --- left: F-k phase diagram with the 6 training points ---------------------
    ph = ax["phase"]
    for n in names:
        F, k, _ = data[n]
        ph.scatter(k, F, s=220, color=COLORS[n], edgecolor="black",
                   linewidth=1.3, zorder=3)
        ph.annotate(n, (k, F), textcoords="offset points", xytext=(10, 6),
                    fontsize=11, fontweight="bold", color=COLORS[n])
    ph.set_xlabel("kill rate  k", fontsize=12)
    ph.set_ylabel("feed rate  F", fontsize=12)
    ph.set_title("Phase diagram — 6 training points", fontsize=13)
    ph.grid(alpha=0.3)
    ph.margins(0.18)

    # --- right: one animated panel per regime (green/red R=A, G=B composite) ----
    ims = {}
    for n in names:
        F, k, stack = data[n]
        a = ax[n]
        ims[n] = a.imshow(stack[0], interpolation="bilinear", animated=True)
        a.set_title(f"{n}\nF={F}  k={k}", fontsize=10, color=COLORS[n],
                    fontweight="bold")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(COLORS[n]); s.set_linewidth(2.5)

    tlabel = fig.text(0.995, 0.02, "", ha="right", fontsize=10, color="#444")

    def update(i):
        for n in names:
            ims[n].set_array(data[n][2][i])
        tlabel.set_text(f"frame {i + 1}/{T}")
        return list(ims.values()) + [tlabel]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    # static poster at a late frame, where the regimes are fully developed/distinct
    update(int(T * 0.85))
    fig.savefig(out_path.replace(".gif", "_poster.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n[gs-regimes-gif] wrote {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--traj", type=int, default=0, help="trajectory index per regime")
    ap.add_argument("--frames", type=int, default=50, help="timesteps sampled")
    ap.add_argument("--tmax", type=int, default=500,
                    help="sample frames from t in [0, tmax] (<=0 = full 1000; "
                         "smaller = smaller per-frame jumps, smoother fast regimes)")
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--dpi", type=int, default=140, help="GIF render dpi (sharpness)")
    ap.add_argument("--out", default="examples/gray_scott/viz/gray_scott_regimes.gif")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[gs-regimes-gif] loading {args.split} traj {args.traj}, "
          f"{args.frames} frames/regime", flush=True)
    data = load(args.split, args.traj, args.frames, args.tmax)
    build(data, args.out, args.fps, args.dpi)


if __name__ == "__main__":
    main()
