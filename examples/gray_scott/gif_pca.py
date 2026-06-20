"""Animated latent-PCA / entropy GIF from outputs/pca_anim_data.npz (no GPU/torch).

Per phase: background = entropy interpolated over the PCA plane (3 nearest data points, IDW,
t-independent); data points in black; a white star marks the current state PCA(s_t) with a
fading trail of recent states. Phases: STATIC (left) | OTHER (mid) | DYNAMIC (right).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.spatial import cKDTree
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

OUT = "outputs"
ORDER = ["bubbles", "worms", "maze", "spots", "gliders", "spirals"]
GROUPS = [("STATIC", 0, 1), ("DYNAMIC", 4, 5)]   # middle phases unlabeled (between the two)
TRAIL = 10
CMAP = "plasma"


def _img(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def surface(Z2, ent, G=180, pad=0.08, k=3):
    x0, x1 = Z2[:, 0].min(), Z2[:, 0].max()
    y0, y1 = Z2[:, 1].min(), Z2[:, 1].max()
    dx, dy = (x1 - x0) * pad + 1e-6, (y1 - y0) * pad + 1e-6
    x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy
    GX, GY = np.meshgrid(np.linspace(x0, x1, G), np.linspace(y0, y1, G))
    d, idx = cKDTree(Z2).query(np.stack([GX.ravel(), GY.ravel()], 1), k=k)
    w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
    return (w * ent[idx]).sum(1).reshape(G, G), [x0, x1, y0, y1]


def build(npz=os.path.join(OUT, "pca_anim_data.npz"),
          out=os.path.join(OUT, "gif_pca_entropy.gif"), fps=12, dpi=95):
    d = dict(np.load(npz, allow_pickle=True))
    T = int(d["T"])
    Z2 = {ph: d[f"Z2_{ph}"] for ph in ORDER}
    ENT = {ph: d[f"ent_{ph}"] for ph in ORDER}
    allent = np.concatenate([ENT[ph] for ph in ORDER])
    vmin, vmax = float(np.percentile(allent, 2)), float(np.percentile(allent, 98))
    norm = Normalize(vmin, vmax)
    surf = {ph: surface(Z2[ph], ENT[ph]) for ph in ORDER}

    frames = []
    for t in range(T):
        fig, axes = plt.subplots(1, 6, figsize=(21, 4.3), dpi=dpi)
        for c, ph in enumerate(ORDER):
            ax = axes[c]; s, ext = surf[ph]
            ax.imshow(s, extent=ext, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax,
                      aspect="auto", alpha=t / max(1, T - 1))    # entropy fades in 0 -> 1
            ax.scatter(Z2[ph][:, 0], Z2[ph][:, 1], s=3, c="k", alpha=0.08, linewidths=0)
            k = len(Z2[ph]) // T                               # one star per trajectory
            for j in range(k):
                tj = Z2[ph][j * T:(j + 1) * T]
                pts = tj[:t + 1]                                # full path so far (kept)
                if len(pts) > 1:
                    segs = np.stack([pts[:-1], pts[1:]], axis=1)
                    lws = np.linspace(0.3, 2.4, len(segs))      # thin at start -> thick at star
                    ax.add_collection(LineCollection(segs, linewidths=lws, colors="k", alpha=0.55))
                ax.scatter(tj[t, 0], tj[t, 1], marker="*", s=150, c="white",
                           edgecolors="k", linewidths=0.8, zorder=6)
            ax.set_title(ph.capitalize(), fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        fig.subplots_adjust(left=0.008, right=0.93, top=0.84, bottom=0.04, wspace=0.07)
        for name, c0, c1 in GROUPS:                            # group headers
            xm = (axes[c0].get_position().x0 + axes[c1].get_position().x1) / 2
            fig.text(xm, 0.90, name, ha="center", fontsize=14, weight="bold", color="0.30")
        sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
        cax = fig.add_axes([0.945, 0.12, 0.011, 0.66])
        fig.colorbar(sm, cax=cax, label="(A,B) state entropy (nats)")
        fig.suptitle(f"Latent PCA trajectory over the state-entropy landscape      t = {t+1}/{T}",
                     fontsize=17, y=0.985)
        frames.append(_img(fig)); plt.close(fig)
    imageio.mimsave(out, frames, format="GIF", fps=fps, loop=1)
    print(f"wrote {out}  ({T} frames)")


if __name__ == "__main__":
    build()
