"""Build the two conference GIFs from outputs/slides_fields.npz (no GPU/torch).

  gif_rollouts.gif : 4x4 — rows = phases (DYNAMIC: gliders, spirals on top; STATIC: bubbles,
                     worms on bottom), cols = Ground truth, JEPA, U-Net, FNO.  RGB fields.
  gif_diff.gif     : 2x3 — rows = dynamic phases (gliders, spirals), cols = JEPA, U-Net, FNO
                     prediction error |model - truth|  (viridis: dark violet = match, yellow = error).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

OUT = "outputs"
PHASES = ["gliders", "spirals", "bubbles", "worms"]      # rows: dynamic (top 2), static (bottom 2)
SRC = ["truth", "jepa", "unet", "fno"]
SRC_LBL = {"truth": "Ground truth", "jepa": "JEPA", "unet": "U-Net", "fno": "FNO"}
MOD = ["jepa", "unet", "fno"]


def to_rgb(f, ranges):                          # [2,Hs,Ws] -> [Hs,Ws,3] : R=A, G=B, B=0
    # per-channel min-max normalization from the clip's ground-truth range (like viz_rollouts)
    (a0, a1), (b0, b1) = ranges
    A = np.clip((f[0] - a0) / (a1 - a0 + 1e-8), 0, 1)
    B = np.clip((f[1] - b0) / (b1 - b0 + 1e-8), 0, 1)
    return np.stack([A, B, np.zeros_like(A)], axis=-1)


def _gt_ranges(d, ph):
    g = d[f"truth_{ph}"].astype(np.float32)
    return ((float(g[0].min()), float(g[0].max())), (float(g[1].min()), float(g[1].max())))


def _img(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def build_rollouts_gif(d, out, fps=12, dpi=85, fstride=1):
    T = d["truth_gliders"].shape[1]
    rng = {ph: _gt_ranges(d, ph) for ph in PHASES}
    frames = []
    for t in range(0, T, fstride):
        fig, axes = plt.subplots(4, 4, figsize=(8.6, 9.0), dpi=dpi)
        for r, ph in enumerate(PHASES):
            for c, src in enumerate(SRC):
                ax = axes[r, c]
                ax.imshow(to_rgb(d[f"{src}_{ph}"][:, t].astype(np.float32), rng[ph]))
                _clean(ax)
                if r == 0:
                    ax.set_title(SRC_LBL[src], fontsize=14)
                if c == 0:
                    ax.set_ylabel(ph.capitalize(), fontsize=14)
        # DYNAMIC / STATIC separators on the left
        fig.text(0.012, 0.71, "DYNAMIC", rotation=90, va="center", fontsize=13,
                 weight="bold", color="0.30")
        fig.text(0.012, 0.29, "STATIC", rotation=90, va="center", fontsize=13,
                 weight="bold", color="0.30")
        fig.suptitle(f"Gray-Scott rollouts across regimes      t = {t+1}/{T}", fontsize=16)
        fig.tight_layout(rect=[0.03, 0, 1, 0.95])
        fig.subplots_adjust(hspace=0.08, wspace=0.04)
        frames.append(_img(fig)); plt.close(fig)
    imageio.mimsave(out, frames, format="GIF", fps=fps, loop=1)
    print(f"wrote {out}  ({T} frames)")


def build_diff_gif(d, out, fps=12, dpi=90, fstride=1):
    phs = ["gliders", "spirals"]
    cols = ["truth", "jepa", "unet", "fno"]                # truth = RGB reference, then diffs
    T = d["truth_gliders"].shape[1]
    rng = {ph: _gt_ranges(d, ph) for ph in phs}
    diff = {ph: {m: np.abs(d[f"{m}_{ph}"].astype(np.float32)
                           - d[f"truth_{ph}"].astype(np.float32)).mean(0) for m in MOD}
            for ph in phs}                                # [T,Hs,Ws]
    vmax = max(float(np.percentile(diff[ph][m], 99)) for ph in phs for m in MOD)
    err = {ph: {m: diff[ph][m].mean(axis=(1, 2)) for m in MOD} for ph in phs}
    emax = max(err[ph][m].max() for ph in phs for m in MOD) + 1e-9
    frames = []
    for t in range(0, T, fstride):
        fig = plt.figure(figsize=(11.0, 6.8), dpi=dpi)
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 0.16, 1, 0.16],
                              hspace=0.10, wspace=0.05)
        for r, ph in enumerate(phs):
            for c, col in enumerate(cols):
                ax = fig.add_subplot(gs[2 * r, c])
                if col == "truth":
                    ax.imshow(to_rgb(d[f"truth_{ph}"][:, t].astype(np.float32), rng[ph]))
                else:
                    ax.imshow(diff[ph][col][t], cmap="viridis", vmin=0, vmax=vmax)
                _clean(ax)
                if r == 0:
                    ax.set_title(SRC_LBL[col], fontsize=15)
                if c == 0:
                    ax.set_ylabel(ph.capitalize(), fontsize=15)
                if col == "truth":
                    continue                               # no error bar under ground truth
                bax = fig.add_subplot(gs[2 * r + 1, c])
                val = float(err[ph][col][t])
                bax.barh([0], [100], height=1, color="0.9")           # track
                bax.barh([0], [100.0 * val / emax], height=1, color="#d62728")  # fill
                bax.text(50, 0, f"{val:.3f}", ha="center", va="center",
                         fontsize=10, color="0.1", weight="bold")
                bax.set_xlim(0, 100); bax.set_ylim(-0.5, 0.5)
                _clean(bax)
        fig.text(0.5, 0.015, "heat-map: dark = match, yellow = error    •    bar: mean |error|",
                 ha="center", fontsize=10.5, color="0.4")
        fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05)
        frames.append(_img(fig)); plt.close(fig)
    imageio.mimsave(out, frames, format="GIF", fps=fps, loop=1)
    print(f"wrote {out}  ({T // fstride + (T % fstride > 0)} frames)")


def build_both(npz=os.path.join(OUT, "slides_fields.npz")):
    os.makedirs(OUT, exist_ok=True)
    d = dict(np.load(npz))
    build_rollouts_gif(d, os.path.join(OUT, "gif_rollouts.gif"))
    build_diff_gif(d, os.path.join(OUT, "gif_diff.gif"))
    # lightweight versions (every 2nd frame, lower dpi)
    build_rollouts_gif(d, os.path.join(OUT, "gif_rollouts_small.gif"), dpi=55, fstride=2)
    build_diff_gif(d, os.path.join(OUT, "gif_diff_small.gif"), dpi=60, fstride=2)


if __name__ == "__main__":
    build_both()
