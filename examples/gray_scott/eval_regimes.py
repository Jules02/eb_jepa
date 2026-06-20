"""Gray-Scott — per-regime VRMSE benchmark.

The Well's Gray-Scott split is 6 distinct F,k regimes (bubbles, gliders, maze,
spirals, spots, worms — see polymathic-ai.org). Each regime is a separate HDF5
file, and the default loader picks one at RANDOM per item, so ``eval.py`` reports
a single VRMSE *averaged blindly over all 6*. That hides where the model actually
struggles: near-static regimes (spots/bubbles) and moving-structure regimes
(gliders/spirals) have wildly different difficulty and persistence baselines.

This script runs the SAME rollout + VRMSE protocol as ``eval.py`` but once per
regime (restricting the loader via ``GrayScottConfig.regime``), then prints a
sorted table and saves a bar chart + per-horizon curves so you can see which
regimes the model is good/bad at — and whether it even beats persistence.

Run:
  python -m examples.gray_scott.eval_regimes --ckpt <.../latest.pth.tar> --H 10
  python -m examples.gray_scott.eval_regimes --ckpt <...> --H 30 --n-per-regime 80
"""
import argparse
import csv
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import (
    GrayScottConfig, make_loader, list_regimes, ROOT)
from examples.gray_scott.eval import (
    C, load_jepa, build_decoder, vrmse_per_horizon, window_vrmse, WELL_WINDOWS)


def _summary(scores, H):
    """Collapse per-horizon arrays into the headline numbers we tabulate."""
    out = {}
    for k in ("jepa", "persistence", "floor", "jepa_v", "floor_v"):
        arr = scores[k]
        out[k] = float(arr.mean())          # mean over the H rollout steps
        out[f"{k}@1"] = float(arr[0])        # first predicted frame
        out[f"{k}@H"] = float(arr[-1])       # last predicted frame
    # The Well Table-3 windows (only those that fit inside H)
    for wname, (_, end) in WELL_WINDOWS.items():
        if end <= H:
            out[f"win_{wname}"] = window_vrmse(scores, wname)["jepa"]
    return out


def benchmark(jepa, encoder, decoder, device, regimes, args, data_root):
    """Run the rollout VRMSE protocol once per regime; return {name: summary}."""
    results, curves = {}, {}
    for name, (F, k) in regimes.items():
        dcfg = GrayScottConfig(
            data_root=data_root, split=args.split, regime=name,
            n_frames=C + args.H, time_stride=args.time_stride,
            epoch_size=args.n_per_regime, batch_size=args.batch_size,
            num_workers=args.num_workers)
        loader = make_loader(dcfg, shuffle=False)
        scores = vrmse_per_horizon(jepa, encoder, decoder, loader, device, args.H,
                                   metric=args.metric)
        results[name] = {"F": F, "k": k, **_summary(scores, args.H)}
        curves[name] = {kk: scores[kk] for kk in ("jepa", "persistence", "floor")}
        s = results[name]
        print(f"  [{name:8s} F={F:<6} k={k:<6}] "
              f"jepa={s['jepa']:.3f}  persist={s['persistence']:.3f}  "
              f"floor={s['floor']:.3f}  (jepa_v={s['jepa_v']:.3f})", flush=True)
    return results, curves


def print_table(results):
    """Sorted best->worst by JEPA VRMSE, with skill vs persistence and floor gap."""
    order = sorted(results, key=lambda n: results[n]["jepa"])
    print("\n=== Per-regime VRMSE (mean over horizon, lower = better) ===", flush=True)
    print(f"{'regime':9s} {'F':>6} {'k':>6} | {'jepa':>7} {'persist':>7} {'floor':>7} "
          f"| {'skill':>6} {'headroom':>8} | {'beats?':>6}", flush=True)
    print("-" * 78, flush=True)
    for n in order:
        s = results[n]
        skill = s["persistence"] / max(s["jepa"], 1e-8)     # >1 => beats persistence
        headroom = s["jepa"] - s["floor"]                   # predictor error above decoder floor
        beats = "yes" if s["jepa"] < s["persistence"] else "NO"
        print(f"{n:9s} {s['F']:>6} {s['k']:>6} | {s['jepa']:>7.3f} {s['persistence']:>7.3f} "
              f"{s['floor']:>7.3f} | {skill:>6.2f} {headroom:>8.3f} | {beats:>6}", flush=True)
    macro = {k: float(np.mean([results[n][k] for n in results]))
             for k in ("jepa", "persistence", "floor")}
    print("-" * 78, flush=True)
    print(f"{'MEAN':9s} {'':>6} {'':>6} | {macro['jepa']:>7.3f} {macro['persistence']:>7.3f} "
          f"{macro['floor']:>7.3f} |  (macro-average across regimes)", flush=True)
    print("\n  skill   = persist/jepa  (>1 means JEPA beats the persistence baseline)\n"
          "  headroom= jepa - floor   (gap above the decoder's irreducible error;\n"
          "            large => the PREDICTOR/rollout is the bottleneck, not the decoder)",
          flush=True)
    return order


def plot_bars(results, order, path):
    """Grouped bars: jepa / persistence / floor per regime (sorted best->worst)."""
    labels = [f"{n}\nF={results[n]['F']} k={results[n]['k']}" for n in order]
    jepa = [results[n]["jepa"] for n in order]
    per = [results[n]["persistence"] for n in order]
    flo = [results[n]["floor"] for n in order]
    x = np.arange(len(order)); w = 0.27
    fig, ax = plt.subplots(figsize=(1.7 * len(order) + 2, 4.6))
    ax.bar(x - w, jepa, w, label="JEPA rollout", color="#2c7fb8")
    ax.bar(x, per, w, label="persistence", color="#bdbdbd")
    ax.bar(x + w, flo, w, label="decoder floor", color="#31a354")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("VRMSE (mean over horizon)")
    ax.set_title("Gray-Scott per-regime VRMSE (lower = better)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_curves(curves, path):
    """JEPA VRMSE vs rollout horizon, one line per regime (+ persistence, dashed)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmap = plt.get_cmap("tab10")
    for i, (name, c) in enumerate(sorted(curves.items())):
        h = np.arange(1, len(c["jepa"]) + 1)
        col = cmap(i % 10)
        ax.plot(h, c["jepa"], "-o", ms=3, color=col, label=name)
        ax.plot(h, c["persistence"], "--", lw=1, color=col, alpha=0.5)
    ax.set_xlabel("rollout step (frames after context)")
    ax.set_ylabel("VRMSE")
    ax.set_title("Per-regime rollout degradation (solid=JEPA, dashed=persistence)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def write_csv(results, path):
    cols = ["regime", "F", "k", "jepa", "persistence", "floor", "jepa_v",
            "jepa@1", "jepa@H"]
    cols += [c for c in next(iter(results.values())) if c.startswith("win_")]
    with open(path, "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(cols)
        for n, s in results.items():
            wtr.writerow([n] + [s.get(c, "") for c in cols[1:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to *.pth.tar (jepa + optional decoder)")
    ap.add_argument("--H", type=int, default=10, help="rollout horizon (frames predicted)")
    ap.add_argument("--n-per-regime", type=int, default=64, help="clips sampled per regime")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--time-stride", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--metric", default="vrmse", choices=["vrmse", "pooled"],
                    help="'vrmse'=The Well mean-of-ratios (blows up on near-uniform "
                         "low-F frames); 'pooled'=denominator-stable diagnostic")
    ap.add_argument("--outdir", default="examples/gray_scott/viz")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    data_root = cfg.data.get("data_root", ROOT)
    jepa, encoder = load_jepa(ckpt, device)
    dstc = int(cfg.model.dstc)
    decoder = build_decoder(dstc, device, ckpt_path=args.ckpt)

    regimes = list_regimes(args.split, data_root)
    print(f"[gs-regimes] ckpt epoch {ckpt.get('epoch')}, H={args.H}, "
          f"{args.n_per_regime}/regime, split={args.split}, metric={args.metric}, "
          f"device={device}", flush=True)
    print(f"[gs-regimes] regimes: {list(regimes)}", flush=True)

    results, curves = benchmark(jepa, encoder, decoder, device, regimes, args, data_root)
    order = print_table(results)

    # suffix outputs by metric so a 'pooled' run doesn't clobber the 'vrmse' one
    sfx = "" if args.metric == "vrmse" else f"_{args.metric}"
    csv_path = os.path.join(args.outdir, f"regime_vrmse{sfx}.csv")
    write_csv(results, csv_path)
    print(f"\n[gs-regimes] wrote {csv_path}", flush=True)
    if not args.no_plot:
        bar = os.path.join(args.outdir, f"regime_vrmse{sfx}_bars.png")
        crv = os.path.join(args.outdir, f"regime_vrmse{sfx}_curves.png")
        plot_bars(results, order, bar)
        plot_curves(curves, crv)
        print(f"[gs-regimes] wrote {bar}\n[gs-regimes] wrote {crv}", flush=True)


if __name__ == "__main__":
    main()
