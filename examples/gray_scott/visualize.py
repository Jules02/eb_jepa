"""Gray-Scott — visualize ground-truth simulations vs JEPA predicted rollouts.

Pulls a few validation clips, rolls the frozen JEPA predictor forward in LATENT
space, decodes each latent back to the 2-channel field, and renders ground truth
vs prediction (vs |error|) side by side. Mirrors the rollout harness in
``eval.py`` (same C context frames, same ``rollout_latents`` + decoder) so the
pictures match the VRMSE numbers.

Two outputs per channel:
  * a static filmstrip PNG  — rows {truth, prediction, |error|}, cols = time
  * an animated GIF         — truth | prediction | error, playing through time

Run:
  python -m examples.gray_scott.visualize --ckpt <.../latest.pth.tar> --H 10
  python -m examples.gray_scott.visualize --ckpt <...> --H 16 --n 4 --channel A
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader, MEAN, STD
from examples.gray_scott.eval import C, load_jepa, build_decoder, rollout_latents

CH = {"A": 0, "B": 1}


def _denorm(field, ch):
    """[..,H,W] z-scored -> physical units for the given channel index."""
    return field * STD[ch] + MEAN[ch]


@torch.no_grad()
def predict_clip(jepa, encoder, decoder, x, H, device):
    """Return denormalised truth / prediction fields for one batch of clips.

    ``x`` is ``[B,2,C+H,H,W]`` (z-scored). The first C frames are context; the
    next H are predicted in latent space and decoded back to fields. We splice
    the C ground-truth context frames in front of the H predicted frames so the
    truth and prediction strips line up frame-for-frame.
    Returns ``truth, pred`` each ``[B,2,C+H,H,W]`` (numpy, physical units).
    """
    pred_z = rollout_latents(jepa, x, H, device)        # [B,D,C+H,h,w]
    pred_future = decoder(pred_z[:, :, C:])             # [B,2,H,H,W]
    pred = torch.cat([x[:, :, :C], pred_future], dim=2)  # context + rollout
    truth = x.cpu().numpy()
    pred = pred.cpu().numpy()
    out = np.empty_like(truth), np.empty_like(pred)
    for arr_in, arr_out in ((truth, out[0]), (pred, out[1])):
        for ch in (0, 1):
            arr_out[:, ch] = _denorm(arr_in[:, ch], ch)
    return out  # truth, pred


def _norm01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _rgb(fields, scale):
    """[2,T,H,W] -> [T,H,W,3] RGB: R=A, G=B (per-channel truth min/max scale)."""
    (loA, hiA), (loB, hiB) = scale
    A = _norm01(fields[0], loA, hiA)
    B = _norm01(fields[1], loB, hiB)
    return np.stack([A, B, np.zeros_like(A)], axis=-1)


def _panels(truth, pred, mode, ch=None):
    """Build the 3 rows {truth, prediction, error} for one sample.

    Each panel is ``(label, frames, render_kwargs)``. ``frames`` is ``[T,H,W]``
    (scalar, drawn with a colormap) or ``[T,H,W,3]`` (RGB, drawn as-is).
    ``mode="single"`` shows one channel; ``mode="composite"`` overlays A,B as RGB
    and reports the combined per-pixel L2 error ``sqrt(ΔA²+ΔB²)`` (both channels,
    honestly aggregated — unlike a mean, which channel A would dominate).
    """
    if mode == "composite":
        scale = [(float(truth[0].min()), float(truth[0].max())),
                 (float(truth[1].min()), float(truth[1].max()))]
        err = np.sqrt(((pred - truth) ** 2).sum(axis=0))   # [T,H,W] over channels
        emax = float(err.max()) or 1e-8
        return [("truth (R=A, G=B)", _rgb(truth, scale), {}),
                ("prediction", _rgb(pred, scale), {}),
                ("L2 error", err, dict(cmap="magma", vmin=0.0, vmax=emax))]
    t, p = truth[ch], pred[ch]                              # [T,H,W]
    err = np.abs(p - t)
    vmin, vmax = float(t.min()), float(t.max())
    emax = float(err.max()) or 1e-8
    return [("truth", t, dict(cmap="viridis", vmin=vmin, vmax=vmax)),
            ("prediction", p, dict(cmap="viridis", vmin=vmin, vmax=vmax)),
            ("|error|", err, dict(cmap="magma", vmin=0.0, vmax=emax))]


def filmstrip(panels, sample_path, title):
    """Static PNG: rows {truth, prediction, error}, cols = time frames."""
    T = panels[0][1].shape[0]
    fig, axes = plt.subplots(3, T, figsize=(1.4 * T, 4.6), squeeze=False)
    for r, (label, data, render) in enumerate(panels):
        for c in range(T):
            ax = axes[r][c]
            im = ax.imshow(data[c], **render)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                tag = f"ctx {c}" if c < C else f"+{c - C + 1}"
                ax.set_title(tag, fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
        if render:  # colorbar only for scalar (colormapped) rows, not RGB
            fig.colorbar(im, ax=axes[r], fraction=0.012, pad=0.01)
    fig.suptitle(title, fontsize=11)
    fig.savefig(sample_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_gif(panels, gif_path, title, fps=8):
    """Animated GIF: truth | prediction | error panels across time."""
    T = panels[0][1].shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    ims = []
    for ax, (label, data, render) in zip(axes, panels):
        im = ax.imshow(data[0], **render)
        ax.set_title(label, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        if render:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ims.append((im, data))
    sup = fig.suptitle("", fontsize=11)

    def update(f):
        for im, data in ims:
            im.set_data(data[f])
        phase = f"context {f}" if f < C else f"rollout +{f - C + 1}"
        sup.set_text(f"{title}   frame {f}/{T - 1}  ({phase})")
        return [im for im, _ in ims] + [sup]

    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to *.pth.tar (jepa + optional decoder)")
    ap.add_argument("--H", type=int, default=10, help="rollout horizon (frames predicted)")
    ap.add_argument("--n", type=int, default=3, help="number of clips to visualize")
    ap.add_argument("--channel", choices=["A", "B", "both", "composite"], default="composite",
                    help="A/B single channel, both (one fig each), or composite "
                         "(RGB overlay R=A G=B + combined L2 error)")
    ap.add_argument("--time-stride", type=int, default=4)
    ap.add_argument("--outdir", default="examples/gray_scott/viz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8, help="GIF frames per second")
    ap.add_argument("--no-gif", action="store_true", help="skip the animated GIFs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    jepa, encoder = load_jepa(ckpt, device)
    dstc = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    decoder = build_decoder(dstc, device, ckpt_path=args.ckpt)
    print(f"[gs-viz] loaded ckpt (epoch {ckpt.get('epoch')}), H={args.H}, "
          f"n={args.n}, device={device}", flush=True)

    dcfg = GrayScottConfig(split="valid", n_frames=C + args.H, time_stride=args.time_stride,
                           epoch_size=args.n, batch_size=args.n, num_workers=2)
    loader = make_loader(dcfg, shuffle=False)
    x = next(iter(loader))["video"].to(device)            # [n,2,C+H,H,W]
    truth, pred = predict_clip(jepa, encoder, decoder, x, args.H, device)

    # views: (filename tag, title label, panel-mode, channel-index)
    if args.channel == "composite":
        views = [("composite", "A+B", "composite", None)]
    elif args.channel == "both":
        views = [("chA", "A", "single", 0), ("chB", "B", "single", 1)]
    else:
        views = [(f"ch{args.channel}", args.channel, "single", CH[args.channel])]

    for i in range(truth.shape[0]):
        for tag, label, mode, ch in views:
            panels = _panels(truth[i], pred[i], mode, ch)
            title = f"Gray-Scott {label} — sample {i} (epoch {ckpt.get('epoch')}, H={args.H})"
            png = os.path.join(args.outdir, f"sample{i}_{tag}_filmstrip.png")
            filmstrip(panels, png, title)
            print(f"  wrote {png}", flush=True)
            if not args.no_gif:
                gif = os.path.join(args.outdir, f"sample{i}_{tag}.gif")
                make_gif(panels, gif, title, fps=args.fps)
                print(f"  wrote {gif}", flush=True)
    print(f"[gs-viz] done -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
