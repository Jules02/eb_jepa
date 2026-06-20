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
from matplotlib.patches import Rectangle
from PIL import Image, ImageSequence

DATA = ("/lustre/work/pdl17890/udl806719/datasets/the_well/"
        "gray_scott_reaction_diffusion/data")
_RE = re.compile(r"diffusion_([a-z]+)_F_([0-9.]+)_k_([0-9.]+)\.hdf5$")

# Plot order (roughly low->high feed F) and a stable colour per regime, shared
# between the phase-diagram marker and the panel title so the eye can link them.
# Colours come straight from tab10 in list order, same convention as
# eval_regimes.py / the field-metrics plots, so a regime is the same colour
# everywhere.
ORDER = ["gliders", "spirals", "maze", "spots", "worms", "bubbles"]
COLORS = {n: c for n, c in zip(ORDER, plt.get_cmap("tab10").colors)}

# gliders/spirals keep evolving (travelling patterns); the other 4 settle into a
# quasi-static texture. They get grouped under labelled overlay rectangles.
DYNAMIC = ["gliders", "spirals"]


def _norm01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _rgb(A, B):
    """[T,H,W] A,B -> [T,H,W,3] RGB: R=A, G=B, the green/red composite used in
    eval_compare.py — red background, green pattern. Blue=0. Each channel is
    normalized per-frame (its own [H,W] min/max at each timestep), independently."""
    def _per_frame(x):
        lo = x.min(axis=(1, 2), keepdims=True)
        hi = x.max(axis=(1, 2), keepdims=True)
        return _norm01(x, lo, hi)
    return np.stack([_per_frame(A), _per_frame(B), np.zeros_like(A)], axis=-1)


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


def _shrink_gif(path, colors):
    """Re-encode an existing GIF with a shared palette + frame optimization.

    PillowWriter writes every frame with its own full RGB palette and no
    optimize pass, which bloats the file. Quantizing all frames to one adaptive
    palette of `colors` entries lets GIF inter-frame optimization (only storing
    changed pixels) actually kick in, typically a 4-6x size cut.
    """
    im = Image.open(path)
    dur = im.info.get("duration", 100)
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    # build the palette from a late, fully-developed frame so the active regimes
    # get the colour budget
    pal = frames[int(len(frames) * 0.85)].quantize(colors=colors,
                                                    method=Image.MEDIANCUT)
    # dither=NONE: dithering adds per-pixel noise that changes every frame,
    # which bloats the palette and kills GIF inter-frame optimization.
    quant = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    quant[0].save(path, save_all=True, append_images=quant[1:], loop=0,
                  duration=dur, optimize=True)


def build(data, out_path, fps, dpi, colors):
    names = [n for n in ORDER if n in data]
    dyn = [n for n in names if n in DYNAMIC]
    stat = [n for n in names if n not in DYNAMIC]
    T = next(iter(data.values()))[2].shape[0]

    fig = plt.figure(figsize=(15, 6.2))
    # dynamic regimes share the left panel column; the 4 static ones fill the 2x2
    # block to their right, so each group is enclosed by one clean rectangle.
    mosaic = [["phase", "phase", dyn[0], stat[0], stat[1]],
              ["phase", "phase", dyn[1], stat[2], stat[3]]]
    ax = fig.subplot_mosaic(mosaic, gridspec_kw=dict(
        width_ratios=[1, 1, 1, 1, 1], wspace=0.08, hspace=0.25))
    # band under the suptitle for the group labels; bottom margin for the x-label
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.09)

    # --- left: F-k phase diagram with the 6 training points ---------------------
    ph = ax["phase"]
    for n in names:
        F, k, _ = data[n]
        ph.scatter(k, F, s=220, color=COLORS[n], edgecolor="black",
                   linewidth=1.3, zorder=3)
        ph.annotate(n, (k, F), textcoords="offset points", xytext=(10, 6),
                    fontsize=11, fontweight="bold", color=COLORS[n])
    ph.set_xlabel("kill rate  k", fontsize=12)
    ph.set_ylabel("feed rate  f", fontsize=12)
    ph.set_title("Phase diagram", fontsize=13)
    ph.grid(alpha=0.3)
    ph.margins(0.18)

    # --- right: one animated panel per regime (green/red R=A, G=B composite) ----
    ims = {}
    for n in names:
        F, k, stack = data[n]
        a = ax[n]
        ims[n] = a.imshow(stack[0], interpolation="bilinear", animated=True)
        a.set_title(f"{n}\nf={F}  k={k}", fontsize=10, color=COLORS[n],
                    fontweight="bold")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(COLORS[n]); s.set_linewidth(2.5)

    # --- group overlays: "dynamic" (gliders/spirals) vs "static" (the rest) -----
    def _group_box(members, color, label):
        boxes = [ax[m].get_position() for m in members]
        x0 = min(b.x0 for b in boxes); x1 = max(b.x1 for b in boxes)
        y0 = min(b.y0 for b in boxes); y1 = max(b.y1 for b in boxes)
        px, top, bot = 0.012, 0.13, 0.02   # top pad clears the per-panel titles
        rx, ry = x0 - px, y0 - bot
        rw, rh = (x1 - x0) + 2 * px, (y1 - y0) + bot + top
        fig.add_artist(Rectangle((rx, ry), rw, rh, transform=fig.transFigure,
                                 fill=False, edgecolor=color, linewidth=2.8,
                                 linestyle="--", zorder=5, clip_on=False))
        # label sits inside the top band, between the border and the panel titles
        fig.text(rx + rw / 2, ry + rh - 0.022, label, ha="center", va="top",
                 fontsize=13, fontweight="bold", color=color, zorder=6)

    _group_box(dyn, "#c1121f", "DYNAMIC")
    _group_box(stat, "#264653", "STATIC")

    tlabel = fig.text(0.99, 0.005, "", ha="right", fontsize=10, color="#444")

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
    raw_mb = os.path.getsize(out_path) / 1e6
    _shrink_gif(out_path, colors)
    print(f"\n[gs-regimes-gif] wrote {out_path} "
          f"({raw_mb:.1f} -> {os.path.getsize(out_path) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--traj", type=int, default=0, help="trajectory index per regime")
    ap.add_argument("--frames", type=int, default=50, help="timesteps sampled")
    ap.add_argument("--tmax", type=int, default=500,
                    help="sample frames from t in [0, tmax] (<=0 = full 1000; "
                         "smaller = smaller per-frame jumps, smoother fast regimes)")
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--dpi", type=int, default=100, help="GIF render dpi (sharpness)")
    ap.add_argument("--colors", type=int, default=64,
                    help="GIF palette size; lower = smaller file (fewer colours)")
    ap.add_argument("--out", default="examples/gray_scott/viz/gray_scott_regimes.gif")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[gs-regimes-gif] loading {args.split} traj {args.traj}, "
          f"{args.frames} frames/regime", flush=True)
    data = load(args.split, args.traj, args.frames, args.tmax)
    build(data, args.out, args.fps, args.dpi, args.colors)


if __name__ == "__main__":
    main()
