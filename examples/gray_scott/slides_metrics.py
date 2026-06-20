"""Conference figure: 3 metrics vs rollout horizon, per phase, per model (jepa/unet/fno).

Loads results/vrmse_data.npz + results/field_metrics_data.npz (no GPU). Phases are split
STATIC (bubbles, worms) | DYNAMIC (gliders, spirals). Writes outputs/metrics_vs_h.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 14})

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
HZ = [1, 15, 30, 60]
MODELS = ["jepa", "unet", "fno"]
COL = {"jepa": "#d62728", "unet": "#1f77b4", "fno": "#2ca02c"}
LBL = {"jepa": "JEPA", "unet": "U-Net", "fno": "FNO"}
PHASES = ["bubbles", "worms", "gliders", "spirals"]          # static | static | dyn | dyn

dv = np.load("results/vrmse_data.npz", allow_pickle=True)
pv = dv["phases"].astype(str); hv = dv["horizons"]
df = np.load("results/field_metrics_data.npz", allow_pickle=True)
pf = df["phases"].astype(str); hzf = df["HZ"].tolist()

def med_vrmse(m, ph):
    a = dv[m][pv == ph]                                       # [n, Hfull]
    return np.array([np.median(a[:, h - 1]) for h in HZ])

def med_field(prefix, m, ph):
    a = df[f"{prefix}_{m}"][pf == ph]                         # [n, 4]
    return np.array([np.median(a[:, hzf.index(h)]) for h in HZ])

ROWS = [("VRMSE", lambda m, ph: med_vrmse(m, ph)),
        ("VGG style distance", lambda m, ph: med_field("vgg", m, ph))]

nrow = len(ROWS)
fig, axes = plt.subplots(nrow, 4, figsize=(16, 4.0 * nrow), sharex=True)
for r, (rname, getv) in enumerate(ROWS):
    for c, ph in enumerate(PHASES):
        ax = axes[r, c]
        for m in MODELS:
            ax.plot(HZ, getv(m, ph), "-o", color=COL[m], lw=2.6, ms=7,
                    label=LBL[m], zorder=3 if m == "jepa" else 2)
        cell_max = max(getv(m, ph).max() for m in MODELS) * 1.12 + 1e-9   # LOCAL y-scale
        ax.set_ylim(0, cell_max)
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if r == 0:
            ax.set_title(ph.capitalize(), fontsize=16, pad=8)
        if r == nrow - 1:
            ax.set_xlabel("rollout horizon $h$")
        if c == 0:
            ax.set_ylabel(rname, fontsize=14)
        ax.set_xticks(HZ)
axes[0, 0].legend(loc="upper left", frameon=False, ncol=1)

fig.tight_layout(rect=[0, 0, 1, 0.88])
# STATIC / DYNAMIC group headers (above the column titles) + separator
fig.text(0.29, 0.935, "STATIC", ha="center", fontsize=17, weight="bold", color="0.30")
fig.text(0.74, 0.935, "DYNAMIC", ha="center", fontsize=17, weight="bold", color="0.30")
fig.lines.append(plt.Line2D([0.515, 0.515], [0.03, 0.91], transform=fig.transFigure,
                            color="0.8", lw=1.2, ls="--"))
fig.suptitle("Rollout error vs horizon — JEPA vs neural-operator baselines",
             fontsize=20, y=0.99)
fig.savefig(os.path.join(OUT, "metrics_vs_h.png"), dpi=400, bbox_inches="tight")
print(f"wrote {OUT}/metrics_vs_h.png")
