"""Re-render the VRMSE bar charts from results/vrmse_data.npz — NO GPU/torch needed.

Same figures as plot_results.py (linear grouped bars, x=phase, sub-grouped by model), but
loaded straight from the cached per-clip data so styling can be iterated instantly.

Run: python examples/gray_scott/replot_bars.py [--npz results/vrmse_data.npz] [--out_dir results]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a = sys.argv
def opt(f, d):
    return a[a.index(f) + 1] if f in a else d
npz = opt("--npz", "results/vrmse_data.npz")
out_dir = opt("--out_dir", "results")
os.makedirs(out_dir, exist_ok=True)

d = np.load(npz, allow_pickle=True)
phases = d["phases"].astype(str)
H = int(d["horizons"].max())
vr = {k: d[k] for k in d.files if k not in ("phases", "horizons")}
PHASES = list(dict.fromkeys(phases.tolist()))
MODELS = ["jepa", "persistence", "linear", "climatology", "unet", "fno", "floor"]
COLORS = {m: c for m, c in zip(MODELS, plt.get_cmap("tab10").colors)}
BAR_MODELS = ["jepa", "unet", "fno"]   # persistence/linear/climatology dropped
CORE = ["jepa", "unet", "fno"]      # well-behaved -> drive the y-scale
nph = len(PHASES)

for hz in [h for h in (1, 15, 30, 60) if h <= H]:
    med = {m: np.array([np.median(vr[m][phases == ph, hz - 1]) for ph in PHASES])
           for m in BAR_MODELS}
    core_clean = np.array([med[m][i] for m in CORE for i in range(nph)])
    # robust cap: 90th pct of the well-behaved bars (not the max) so a single large core
    # bar (e.g. fno unstable on maze) clips+annotates instead of shrinking everything else
    cap = 1.25 * np.percentile(core_clean, 90) if core_clean.size else \
        max(med[m].max() for m in BAR_MODELS)

    # SOFT-CLIP: below cap = linear (full visibility); above cap = log-compressed into a
    # thin band (cap, ytop], so over-cap bars stay ORDERED among themselves (you see which
    # is bigger) without crushing the rest. True value printed above each compressed bar.
    vmax = max(med[m].max() for m in BAR_MODELS)
    ytop = cap * 1.20
    def softclip(v):
        v = np.asarray(v, float); out = v.copy(); hi = v > cap
        if vmax > cap and hi.any():
            out[hi] = cap + (ytop - cap) * np.log(v[hi] / cap) / np.log(vmax / cap)
        return out

    fig, ax = plt.subplots(figsize=(13, 5.5))
    w = 0.8 / len(BAR_MODELS)
    x = np.arange(nph)
    for j, m in enumerate(BAR_MODELS):
        bx = x + j * w
        hdraw = softclip(med[m])
        ax.bar(bx, hdraw, w, color=COLORS[m], label=m)
        for i, v in enumerate(med[m]):
            if v > cap:
                lbl = f"{v:.1f}" if v < 10 else f"{v:.0f}"
                ax.text(bx[i], hdraw[i], lbl, ha="center", va="bottom",
                        fontsize=6.5, rotation=90, color=COLORS[m])
    ax.axhline(cap, ls="--", lw=0.7, color="0.6")
    ax.text(nph - 0.5, cap, "  soft-clip ↑ (log)", fontsize=7, color="0.45", va="center")
    ax.set_ylim(0, ytop + 0.30 * cap)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(PHASES, fontsize=11)
    ax.set_ylabel(f"VRMSE  (median, h{hz})", fontsize=11)
    ax.set_title(f"VRMSE per phase per model  —  horizon h{hz}  (test)", fontsize=12)
    ax.legend(ncol=len(BAR_MODELS), fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"vrmse_bars_h{hz}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/vrmse_bars_h{hz}.png  (cap={cap:.3f})")
