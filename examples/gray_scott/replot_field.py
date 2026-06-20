"""Re-render the field-metric bar charts from results/field_metrics_data.npz — no GPU.

One bar chart per horizon for W2-spatial and VGG-style, plus the cube-pooled W2-values.
Plotted models: jepa, unet, fno, persistence (linear & climatology excluded from the plots).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = sys.argv
npz = a[a.index("--npz") + 1] if "--npz" in a else "results/field_metrics_data.npz"
out_dir = "results"
d = np.load(npz, allow_pickle=True)
phases = d["phases"].astype(str)
HZ = d["HZ"].tolist()
PHASES = list(dict.fromkeys(phases.tolist()))
ALL = ["jepa", "unet", "fno", "persistence", "linear"]          # for stable colors
PLOT_MODELS = ["jepa", "unet", "fno"]            # persistence/linear excluded from plots
COLORS = {m: c for m, c in zip(ALL, plt.get_cmap("tab10").colors)}
w2s = {m: d[f"w2s_{m}"] for m in ALL}
vgg = {m: d[f"vgg_{m}"] for m in ALL}
w2v = {m: d[f"w2v_{m}"] for m in ALL}


def bars(getval, name, fname):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(PHASES)); w = 0.8 / len(PLOT_MODELS)
    for j, m in enumerate(PLOT_MODELS):
        vals = [np.median(getval(m, p)) for p in PHASES]
        ax.bar(x + j * w, vals, w, color=COLORS[m], label=m)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(PHASES, fontsize=11)
    ax.set_ylabel(name); ax.set_title(f"{name}  per phase per model  (test)")
    ax.legend(ncol=len(PLOT_MODELS), fontsize=9, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.08))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/{fname}")


for i, h in enumerate(HZ):
    bars(lambda m, p, i=i: w2s[m][phases == p][:, i], f"W2 spatial @ h{h}", f"field_w2spatial_h{h}.png")
    bars(lambda m, p, i=i: vgg[m][phases == p][:, i], f"VGG style @ h{h}", f"field_vgg_h{h}.png")
bars(lambda m, p: w2v[m][phases == p], "W2 values (A,B), cube", "field_w2values.png")
