"""Grouped bar charts of VRMSE per phase per model (JEPA + all baselines).

Recomputes per-clip VRMSE [N,H] for every model (cached decoder/U-Net/FNO -> fast), saves
results/vrmse_data.npz, and renders one LINEAR grouped bar chart per horizon:
  - vrmse_bars_h{1,15,30,60}.png   x = phase, sub-grouped by model, y = median VRMSE.
Degenerate phases (gliders/spirals: near-zero spatial variance -> VRMSE blows up for every
model) would crush a linear axis, so their bars are capped and the true value is printed above.

Run: python -m examples.gray_scott.plot_results --ckpt <jepa.pth.tar> [--per_regime 100] [--H 60]
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from examples.gray_scott.eval import load_jepa, build_decoder, C
from examples.gray_scott.eval_common import build_regime_clips
from examples.gray_scott.final_table import per_clip_jepa, per_clip_field, per_clip_floor
from examples.gray_scott.baselines import (
    FNO2d, load_or_train, step_model, step_persistence, step_linear, step_climatology)
from eb_jepa.architectures import ResUNet

MODELS = ["jepa", "persistence", "linear", "climatology", "unet", "fno", "floor"]
COLORS = {m: c for m, c in zip(MODELS, plt.get_cmap("tab10").colors)}


def main():
    a = sys.argv
    def opt(f, d, c=str):
        return c(a[a.index(f) + 1]) if f in a else d
    ckpt_path = a[a.index("--ckpt") + 1]
    H = opt("--H", 60, int)
    per_regime = opt("--per_regime", 20, int)
    split = opt("--split", "test")
    out_dir = opt("--out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"]); stride = int(cfg.data.get("time_stride", 4))
    jepa, encoder = load_jepa(ckpt, device)
    decoder = build_decoder(int(cfg.model.dstc), device, ckpt_path=ckpt_path)
    clips, tags, titles = build_regime_clips(split, C + H, stride, per_regime)
    phases = np.array([t.split()[0] for t in titles])
    PHASES = list(dict.fromkeys(phases.tolist()))
    print(f"[plot] epoch={ckpt.get('epoch')} {len(clips)} clips, phases={PHASES}", flush=True)

    unet = ResUNet(in_d=2 * C, h_d=32, out_d=2, norm="group").to(device)
    load_or_train("unet", unet, device, stride, 20)
    fno = FNO2d(in_c=2 * C, out_c=2, width=32, modes=16, n_layers=4).to(device)
    load_or_train("fno", fno, device, stride, 20)

    vr = {
        "jepa": per_clip_jepa(jepa, decoder, clips, device, H),
        "persistence": per_clip_field(step_persistence, clips, device, H),
        "linear": per_clip_field(step_linear, clips, device, H),
        "climatology": per_clip_field(step_climatology, clips, device, H),
        "unet": per_clip_field(step_model(unet), clips, device, H),
        "fno": per_clip_field(step_model(fno), clips, device, H),
        "floor": per_clip_floor(encoder, decoder, clips, device, H),
    }
    np.savez(os.path.join(out_dir, "vrmse_data.npz"),
             phases=phases, horizons=np.arange(1, H + 1), **vr)
    print(f"[plot] saved {out_dir}/vrmse_data.npz", flush=True)

    # --- grouped bar charts: x=phase, sub-grouped by model, y=median VRMSE, LINEAR ---
    # One figure per horizon. No log, no error bars, no gridlines.
    BAR_MODELS = ["jepa", "unet", "fno"]   # persistence/linear/climatology dropped
    HORIZONS = [h for h in (1, 15, 30, 60) if h <= H]
    nph = len(PHASES)
    for hz in HORIZONS:
        med = {m: np.array([np.median(vr[m][phases == ph, hz - 1]) for ph in PHASES])
               for m in BAR_MODELS}
        # The y-axis is set ONLY by the well-behaved predictors (jepa/persistence/unet/fno)
        # on the non-degenerate phases, so those bars always fill the axis and stay readable.
        # Everything bigger -- climatology, linear, and the degenerate phases (gliders/spirals,
        # ~0 spatial variance -> VRMSE blows up) -- is CLIPPED at the cap and its true value
        # is printed above the bar. Stays fully linear.
        CORE = ["jepa", "unet", "fno"]
        core_clean = np.array([med[m][i] for m in CORE for i in range(nph)])
        # robust cap: 90th pct of the well-behaved bars (not the max) so a single large core
        # bar (e.g. fno unstable on maze) clips+annotates instead of shrinking everything else
        cap = 1.25 * np.percentile(core_clean, 90) if core_clean.size else \
            max(med[m].max() for m in BAR_MODELS)

        # SOFT-CLIP: below cap = linear; above = log-compressed into (cap, ytop] so over-cap
        # bars stay ORDERED among themselves. True value printed above each compressed bar.
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
                if v > cap:                       # compressed: print the true value above
                    lbl = f"{v:.1f}" if v < 10 else f"{v:.0f}"
                    ax.text(bx[i], hdraw[i], lbl, ha="center", va="bottom",
                            fontsize=6.5, rotation=90, color=COLORS[m])
        ax.axhline(cap, ls="--", lw=0.7, color="0.6")
        ax.text(nph - 0.5, cap, "  soft-clip ↑ (log)", fontsize=7, color="0.45", va="center")
        ax.set_ylim(0, ytop + 0.30 * cap)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(PHASES, fontsize=11)
        ax.set_ylabel(f"VRMSE  (median, h{hz})", fontsize=11)
        ax.set_title(f"VRMSE per phase per model  —  horizon h{hz}  "
                     f"(epoch {ckpt.get('epoch')}, test)", fontsize=12)
        ax.legend(ncol=len(BAR_MODELS), fontsize=9, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.08))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"vrmse_bars_h{hz}.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote vrmse_bars_h{hz}.png", flush=True)
    print("[plot] DONE", flush=True)


if __name__ == "__main__":
    main()
