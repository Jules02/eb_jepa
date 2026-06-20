"""Gray-Scott — visualize ground-truth simulations vs JEPA predicted rollouts.

Pulls a few validation clips, rolls the frozen JEPA predictor forward in LATENT
space, decodes each latent back to the 2-channel field, and renders ground truth
vs prediction (vs |error|) side by side. Mirrors the rollout harness in
``eval.py`` (same C context frames, same ``rollout_latents`` + decoder) so the
pictures match the VRMSE numbers.

Two outputs per channel:
  * a static filmstrip PNG  — rows {truth, prediction, |error|}, cols = time
  * an animated GIF         — truth | prediction | error, playing through time

With --baselines: adds UNetClassic and CNextU-Net panels to the same GIF,
producing a 4-panel comparison (Truth | JEPA | UNetClassic | CNextU-Net).

Run:
  python -m examples.gray_scott.visualize --ckpt <.../latest.pth.tar> --H 60
  python -m examples.gray_scott.visualize --ckpt <...> --H 60 --baselines
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

from examples.gray_scott._well_baselines import stub_heavy_well_models

# Stub heavy unused models so the_well imports cleanly without timm
stub_heavy_well_models()

from eb_jepa.architectures import ResUNet
from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader, MEAN, STD
from examples.gray_scott.eval import C, load_jepa, build_decoder, rollout_latents

CH = {"A": 0, "B": 1}
C_UNET = 4
HF_SUFFIX = "gray_scott_reaction_diffusion"


def _denorm(field, ch):
    """[..,H,W] z-scored -> physical units for the given channel index."""
    return field * STD[ch] + MEAN[ch]


def _denorm_arr(arr):
    """numpy array with channel dim=0 ([2,T,H,W]) or dim=1 ([B,2,T,H,W]) -> physical units."""
    out = arr.copy()
    if arr.ndim == 4:           # [2, T, H, W]
        for ch in (0, 1):
            out[ch] = _denorm(arr[ch], ch)
    else:                        # [B, 2, T, H, W]
        for ch in (0, 1):
            out[:, ch] = _denorm(arr[:, ch], ch)
    return out


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


@torch.no_grad()
def _unet_rollout(model, x_ctx, H):
    """Autoregressive UNet rollout. x_ctx: [B, C_UNET, Hs, Ws, NC] channels-last."""
    preds = []
    ctx = x_ctx.clone()
    for _ in range(H):
        inp = ctx.permute(0, 1, 4, 2, 3).flatten(1, 2)  # [B, C*NC, Hs, Ws]
        out_cf = model(inp)                               # [B, NC, Hs, Ws]
        preds.append(out_cf.unsqueeze(1))
        ctx = torch.cat([ctx[:, 1:], out_cf.permute(0, 2, 3, 1).unsqueeze(1)], dim=1)
    return torch.cat(preds, dim=1)                       # [B, H, NC, Hs, Ws]


def _jepa_aligned_pred(jepa, decoder, x, H, device):
    """Run one JEPA rollout and return predictions aligned to the UNet future window.

    Returns [B, 2, H, Hs, Ws] numpy array in normalised space (before denorm).
    """
    H_jepa = (C_UNET - C) + H
    pred_z = rollout_latents(jepa, x, H_jepa, device)   # [B,D,C+H_jepa,h,w]
    jepa_future = decoder(pred_z[:, :, C:])              # [B,2,H_jepa,Hs,Ws]
    return jepa_future[:, :, (C_UNET - C):]             # [B,2,H,Hs,Ws]


@torch.no_grad()
def predict_clip_compare(jepa_models, unet, cnext, x, H, device):
    """Predictions for one or more JEPA models + both UNet baselines.

    jepa_models: list of (label, jepa, decoder) tuples
    x: [B, 2, C_UNET+H, Hs, Ws]
    Returns: truth [B,2,H,Hs,Ws], list of (label, pred [B,2,H,Hs,Ws]) — all numpy, physical units.
    The shared future window is frames [C_UNET, C_UNET+H) in absolute time.
    """
    def to_np(t):
        return _denorm_arr(t.cpu().numpy())

    truth_future = x[:, :, C_UNET:C_UNET + H]

    preds = []
    for label, jepa, decoder in jepa_models:
        aligned = _jepa_aligned_pred(jepa, decoder, x, H, device)
        preds.append((label, to_np(aligned)))

    x_cl = x.permute(0, 2, 3, 4, 1)
    preds.append(("UNetClassic", to_np(_unet_rollout(unet,  x_cl[:, :C_UNET], H).permute(0, 2, 1, 3, 4))))
    preds.append(("CNextU-Net",  to_np(_unet_rollout(cnext, x_cl[:, :C_UNET], H).permute(0, 2, 1, 3, 4))))

    return to_np(truth_future), preds


@torch.no_grad()
def _field_rollout_c2(model, x, H):
    """Autoregressive rollout for field-space models with C=2 context.

    x: [B, 2, C+H, Hs, Ws]; returns [B, 2, H, Hs, Ws] — all z-scored.
    """
    preds = []
    ctx = x[:, :, :C].clone()                    # [B, 2, C, Hs, Ws]
    for _ in range(H):
        inp = ctx.flatten(1, 2)                   # [B, 2*C=4, Hs, Ws]
        out = model(inp)                          # [B, 2, Hs, Ws]
        preds.append(out.unsqueeze(2))
        ctx = torch.cat([ctx[:, :, 1:], out.unsqueeze(2)], dim=2)
    return torch.cat(preds, dim=2)               # [B, 2, H, Hs, Ws]


@torch.no_grad()
def predict_clip_compare_s4(jepa_models, field_models, x, H, device):
    """Predictions for JEPA models + stride-4 field-space baselines (all C=2 context).

    jepa_models: list of (label, jepa, decoder)
    field_models: list of (label, model)
    x: [B, 2, C+H, Hs, Ws]
    Returns: truth [B,2,H,Hs,Ws], list of (label, pred [B,2,H,Hs,Ws]) — numpy, physical units.
    """
    def to_np(t):
        return _denorm_arr(t.cpu().numpy())

    truth_future = x[:, :, C:C + H]             # [B, 2, H, Hs, Ws]

    preds = []
    for label, jepa, decoder in jepa_models:
        pred_z = rollout_latents(jepa, x, H, device)
        jepa_future = decoder(pred_z[:, :, C:])  # [B, 2, H, Hs, Ws]
        preds.append((label, to_np(jepa_future)))

    for label, model in field_models:
        preds.append((label, to_np(_field_rollout_c2(model, x, H))))

    return to_np(truth_future), preds


def _norm01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _rgb(fields, scale):
    """[2,T,H,W] -> [T,H,W,3] RGB: R=A, G=B (per-channel truth min/max scale)."""
    (loA, hiA), (loB, hiB) = scale
    A = _norm01(fields[0], loA, hiA)
    B = _norm01(fields[1], loB, hiB)
    return np.stack([A, B, np.zeros_like(A)], axis=-1)


def _panels(truth, pred, mode, ch=None):
    """Build the 3 rows {truth, prediction, error} for one sample."""
    if mode == "composite":
        scale = [(float(truth[0].min()), float(truth[0].max())),
                 (float(truth[1].min()), float(truth[1].max()))]
        err = np.sqrt(((pred - truth) ** 2).sum(axis=0))
        emax = float(err.max()) or 1e-8
        return [("truth (R=A, G=B)", _rgb(truth, scale), {}),
                ("prediction", _rgb(pred, scale), {}),
                ("L2 error", err, dict(cmap="magma", vmin=0.0, vmax=emax))]
    t, p = truth[ch], pred[ch]
    err = np.abs(p - t)
    vmin, vmax = float(t.min()), float(t.max())
    emax = float(err.max()) or 1e-8
    return [("truth", t, dict(cmap="viridis", vmin=vmin, vmax=vmax)),
            ("prediction", p, dict(cmap="viridis", vmin=vmin, vmax=vmax)),
            ("|error|", err, dict(cmap="magma", vmin=0.0, vmax=emax))]


def _compare_panels(truth, preds):
    """RGB comparison panels: Truth + one panel per model in preds.

    truth: [2, T, Hs, Ws] numpy (physical units)
    preds: list of (label, [2, T, Hs, Ws]) numpy arrays
    Returns list of (label, [T, Hs, Ws, 3], {}) panels.
    """
    scale = [(float(truth[0].min()), float(truth[0].max())),
             (float(truth[1].min()), float(truth[1].max()))]
    panels = [("Truth (R=A, G=B)", _rgb(truth, scale), {})]
    for label, pred in preds:
        panels.append((label, _rgb(pred, scale), {}))
    return panels


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
        if render:
            fig.colorbar(im, ax=axes[r], fraction=0.012, pad=0.01)
    fig.suptitle(title, fontsize=11)
    fig.savefig(sample_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_gif(panels, gif_path, title, fps=8):
    """Animated GIF: N panels side by side, playing through time."""
    N = len(panels)
    T = panels[0][1].shape[0]
    fig, axes = plt.subplots(1, N, figsize=(3 * N, 3.4))
    if N == 1:
        axes = [axes]
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


def make_compare_gif(panels, gif_path, title, fps=8):
    """Animated GIF for comparison mode — no 'context/rollout' phase label."""
    N = len(panels)
    T = panels[0][1].shape[0]
    fig, axes = plt.subplots(1, N, figsize=(3 * N, 3.4))
    if N == 1:
        axes = [axes]
    ims = []
    for ax, (label, data, render) in zip(axes, panels):
        im = ax.imshow(data[0], **render)
        ax.set_title(label, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        ims.append((im, data))
    sup = fig.suptitle("", fontsize=11)

    def update(f):
        for im, data in ims:
            im.set_data(data[f])
        sup.set_text(f"{title}   t={f + 1}/{T}")
        return [im for im, _ in ims] + [sup]

    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to *.pth.tar (jepa + optional decoder)")
    ap.add_argument("--H", type=int, default=60, help="rollout horizon (frames predicted)")
    ap.add_argument("--n", type=int, default=4, help="number of clips to visualize")
    ap.add_argument("--channel", choices=["A", "B", "both", "composite"], default="composite")
    ap.add_argument("--time-stride", type=int, default=4)
    ap.add_argument("--regime", default=None,
                    help="restrict clips to one regime (bubbles/gliders/maze/spirals/spots/worms)")
    ap.add_argument("--outdir", default="examples/gray_scott/viz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=10, help="GIF frames per second")
    ap.add_argument("--no-gif", action="store_true", help="skip the animated GIFs")
    ap.add_argument("--baselines", action="store_true",
                    help="add UNetClassic + CNextU-Net panels for side-by-side comparison")
    ap.add_argument("--baselines-s4", action="store_true",
                    help="add stride-4 ResUNet + FNO panels (weights from EBJEPA_CKPTS)")
    ap.add_argument("--ckpt2", default=None,
                    help="optional second JEPA checkpoint to add as extra panel")
    ap.add_argument("--label2", default=None,
                    help="display label for --ckpt2 panel (default: inferred from checkpoint)")
    ap.add_argument("--tag", default="all",
                    help="suffix for output gif filenames: sample{i}_compare_{tag}.gif")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    jepa, encoder = load_jepa(ckpt, device)
    dstc = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    decoder = build_decoder(dstc, device, ckpt_path=args.ckpt)
    print(f"[gs-viz] loaded ckpt (epoch {ckpt.get('epoch')}), H={args.H}, "
          f"n={args.n}, baselines={args.baselines}, device={device}", flush=True)

    if args.baselines_s4:
        from examples.gray_scott.baselines import FNO2d, baseline_weights_path

        unet_s4 = ResUNet(in_d=2 * C, h_d=32, out_d=2, norm="group").to(device)
        unet_s4.load_state_dict(torch.load(baseline_weights_path(args.time_stride, "unet"),
                                           map_location=device))
        unet_s4.eval()

        fno_s4 = FNO2d(in_c=2 * C, out_c=2, width=32, modes=16, n_layers=4).to(device)
        fno_s4.load_state_dict(torch.load(baseline_weights_path(args.time_stride, "fno"),
                                          map_location=device))
        fno_s4.eval()
        print("[gs-viz] loaded stride=4 ResUNet and FNO baselines", flush=True)

        ep1 = ckpt.get("epoch")
        jepa_models = [(f"JEPA-small (ep{ep1})", jepa, decoder)]
        if args.ckpt2:
            ckpt2 = torch.load(args.ckpt2, map_location=device, weights_only=False)
            jepa2, _ = load_jepa(ckpt2, device)
            dstc2 = int(OmegaConf.create(ckpt2["cfg"]).model.dstc)
            decoder2 = build_decoder(dstc2, device, ckpt_path=args.ckpt2)
            ep2 = ckpt2.get("epoch")
            label2 = args.label2 if args.label2 else f"JEPA-2 D={dstc2} (ep{ep2})"
            jepa_models.append((label2, jepa2, decoder2))
            print(f"[gs-viz] loaded second ckpt (epoch {ep2}, D={dstc2})", flush=True)

        field_models = [("ResUNet-s4", unet_s4), ("FNO-s4", fno_s4)]

        n_frames = C + args.H
        dcfg = GrayScottConfig(split="valid", n_frames=n_frames, time_stride=args.time_stride,
                               epoch_size=args.n, batch_size=args.n, num_workers=2)
        loader = make_loader(dcfg, shuffle=False)
        x = next(iter(loader))["video"].to(device)

        truth_all, preds_all = predict_clip_compare_s4(jepa_models, field_models, x, args.H, device)

        for i in range(truth_all.shape[0]):
            sample_preds = [(lbl, p[i]) for lbl, p in preds_all]
            panels = _compare_panels(truth_all[i], sample_preds)
            title = f"Gray-Scott sample {i} — stride={args.time_stride}"
            if not args.no_gif:
                gif = os.path.join(args.outdir, f"sample{i}_compare_{args.tag}.gif")
                make_compare_gif(panels, gif, title, fps=args.fps)
                print(f"  wrote {gif}", flush=True)
        print(f"[gs-viz] done -> {args.outdir}", flush=True)
        return

    if args.baselines:
        from the_well.benchmark.models.unet_classic import UNetClassic
        from the_well.benchmark.models.unet_convnext import UNetConvNext
        unet  = UNetClassic.from_pretrained(f"polymathic-ai/UNetClassic-{HF_SUFFIX}").to(device).eval()
        cnext = UNetConvNext.from_pretrained(f"polymathic-ai/UNetConvNext-{HF_SUFFIX}").to(device).eval()
        print("[gs-viz] loaded UNetClassic and CNextU-Net", flush=True)

        # Build JEPA model list — always include primary ckpt, optionally a second
        ep1 = ckpt.get("epoch")
        jepa_models = [(f"JEPA-small (ep{ep1})", jepa, decoder)]
        if args.ckpt2:
            ckpt2 = torch.load(args.ckpt2, map_location=device, weights_only=False)
            jepa2, _ = load_jepa(ckpt2, device)
            dstc2 = int(OmegaConf.create(ckpt2["cfg"]).model.dstc)
            decoder2 = build_decoder(dstc2, device, ckpt_path=args.ckpt2)
            ep2 = ckpt2.get("epoch")
            label2 = args.label2 if args.label2 else f"JEPA-2 D={dstc2} (ep{ep2})"
            jepa_models.append((label2, jepa2, decoder2))
            print(f"[gs-viz] loaded second ckpt (epoch {ep2}, D={dstc2})", flush=True)

        n_frames = C_UNET + args.H
        dcfg = GrayScottConfig(split="valid", n_frames=n_frames, time_stride=args.time_stride,
                               epoch_size=args.n, batch_size=args.n, num_workers=2)
        loader = make_loader(dcfg, shuffle=False)
        x = next(iter(loader))["video"].to(device)

        truth_all, preds_all = predict_clip_compare(jepa_models, unet, cnext, x, args.H, device)

        for i in range(truth_all.shape[0]):
            sample_preds = [(lbl, p[i]) for lbl, p in preds_all]
            panels = _compare_panels(truth_all[i], sample_preds)
            title = f"Gray-Scott sample {i} — stride={args.time_stride}"
            if not args.no_gif:
                gif = os.path.join(args.outdir, f"sample{i}_compare_{args.tag}.gif")
                make_compare_gif(panels, gif, title, fps=args.fps)
                print(f"  wrote {gif}", flush=True)
        print(f"[gs-viz] done -> {args.outdir}", flush=True)
        return

    # ── JEPA-only mode (original behaviour) ──────────────────────────────────
    dcfg = GrayScottConfig(split="valid", n_frames=C + args.H, time_stride=args.time_stride,
                           epoch_size=args.n, batch_size=args.n, num_workers=2,
                           regime=args.regime)
    loader = make_loader(dcfg, shuffle=False)
    x = next(iter(loader))["video"].to(device)            # [n,2,C+H,H,W]
    truth, pred = predict_clip(jepa, encoder, decoder, x, args.H, device)

    if args.channel == "composite":
        views = [("composite", "A+B", "composite", None)]
    elif args.channel == "both":
        views = [("chA", "A", "single", 0), ("chB", "B", "single", 1)]
    else:
        views = [(f"ch{args.channel}", args.channel, "single", CH[args.channel])]

    rtag = f"{args.regime}_" if args.regime else ""
    rlabel = f" [{args.regime}]" if args.regime else ""
    for i in range(truth.shape[0]):
        for tag, label, mode, ch in views:
            panels = _panels(truth[i], pred[i], mode, ch)
            title = f"Gray-Scott {label}{rlabel} — sample {i} (epoch {ckpt.get('epoch')}, H={args.H})"
            png = os.path.join(args.outdir, f"{rtag}sample{i}_{tag}_filmstrip.png")
            filmstrip(panels, png, title)
            print(f"  wrote {png}", flush=True)
            if not args.no_gif:
                gif = os.path.join(args.outdir, f"{rtag}sample{i}_{tag}.gif")
                make_gif(panels, gif, title, fps=args.fps)
                print(f"  wrote {gif}", flush=True)
    print(f"[gs-viz] done -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
