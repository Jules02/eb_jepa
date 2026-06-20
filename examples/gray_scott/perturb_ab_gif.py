"""4-panel perturbation GIF: Truth | Original pred | A+delta | B+delta.

For each regime, adds delta to the z-scored A (or B) channel across all context
frames and compares the JEPA's predicted rollout with the unperturbed prediction.

Usage:
  uv run python -m examples.gray_scott.perturb_ab_gif --ckpt <...> --delta 1.0
"""
import re, os, argparse, glob
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import NT, MEAN, STD, ROOT
from examples.gray_scott.eval import C, load_jepa, build_decoder, rollout_latents

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_fk(path):
    m = re.search(r'_F_([\d.]+)_k_([\d.]+)\.hdf5', path)
    return float(m.group(1)), float(m.group(2))


def _parse_regime(path):
    m = re.search(r'diffusion_([a-z]+)_F_', os.path.basename(path))
    return m.group(1) if m else "unknown"


def _load_clip(f, ntraj, span, stride, rng):
    tr = int(rng.integers(ntraj))
    t0 = int(rng.integers(0, max(1, NT - span + 1)))
    sl = slice(t0, t0 + span, stride)
    A = f["t0_fields/A"][tr, sl]
    B = f["t0_fields/B"][tr, sl]
    x = np.stack([A, B], axis=0).astype(np.float32)
    return (x - MEAN[:, None, None, None]) / STD[:, None, None, None]


def _denorm(arr):
    out = arr.copy()
    for ch in (0, 1):
        out[ch] = arr[ch] * STD[ch] + MEAN[ch]
    return out


def _norm01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _rgb(fields, scale):
    (loA, hiA), (loB, hiB) = scale
    A = _norm01(fields[0], loA, hiA)
    B = _norm01(fields[1], loB, hiB)
    return np.stack([A, B, np.zeros_like(A)], axis=-1)


@torch.no_grad()
def decode_rollout(jepa, decoder, x, H):
    """x: [1,2,>=C,H,W] → decoded future [2,H,Hs,Ws] numpy (z-scored)."""
    pred_z = rollout_latents(jepa, x.to(DEVICE), H, DEVICE)   # [1,D,C+H,h,w]
    pred   = decoder(pred_z[:, :, C:])                         # [1,2,H,Hs,Ws]
    return pred[0].cpu().numpy()                               # [2,H,Hs,Ws]


@torch.no_grad()
def make_ab_gif(x_base, jepa, decoder, H, regime, fk, delta, outdir, tag, fps=8):
    """2-row x 4-col GIF.
    Cols: Truth | Original | A+δ | B+δ
    Row 0: channel A (substrate)   Row 1: channel B (activator)
    """
    x_base = x_base.to(DEVICE)

    truth_ph = _denorm(x_base[0, :, C:C+H].cpu().numpy())   # [2,H,Hs,Ws]

    x_A = x_base.clone(); x_A[:, 0, :C] += delta
    x_B = x_base.clone(); x_B[:, 1, :C] += delta

    pred_orig = _denorm(decode_rollout(jepa, decoder, x_base, H))
    pred_A    = _denorm(decode_rollout(jepa, decoder, x_A,    H))
    pred_B    = _denorm(decode_rollout(jepa, decoder, x_B,    H))

    # Latent diffs
    z0 = rollout_latents(jepa, x_base, H, DEVICE)
    zA = rollout_latents(jepa, x_A,    H, DEVICE)
    zB = rollout_latents(jepa, x_B,    H, DEVICE)
    dA = ((zA - z0).norm() / z0.norm().clamp(1e-8)).item()
    dB = ((zB - z0).norm() / z0.norm().clamp(1e-8)).item()
    print(f"  {regime:10s}  ||Δz_A||/||z||={dA:.4f}  ||Δz_B||/||z||={dB:.4f}", flush=True)

    # cols: truth / orig / A+δ / B+δ   ×   rows: ch A / ch B
    col_data  = [truth_ph, pred_orig, pred_A, pred_B]
    col_titles = ["Truth", "Original", f"init A+{delta:.1f}σ", f"init B+{delta:.1f}σ"]
    row_labels = ["A (substrate)", "B (activator)"]

    # per-channel colour range fixed from truth
    vmin = [float(truth_ph[ch].min()) for ch in (0, 1)]
    vmax = [float(truth_ph[ch].max()) for ch in (0, 1)]

    T = H
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    fig.subplots_adjust(hspace=0.08, wspace=0.04)

    ims = []   # (im, data_array [H,Hs,Ws])
    for row, ch in enumerate((0, 1)):
        for col, data in enumerate(col_data):
            ax = axes[row, col]
            im = ax.imshow(data[ch, 0], cmap="viridis",
                           vmin=vmin[ch], vmax=vmax[ch])
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8)
            ims.append((im, data[ch]))   # data[ch]: [H, Hs, Ws]

    sup = fig.suptitle("", fontsize=10)

    def update(f):
        for im, frames in ims:
            im.set_data(frames[f])
        sup.set_text(f"{regime}  F={fk[0]:.3f} k={fk[1]:.3f}   t={f+1}/{T}")
        return [im for im, _ in ims] + [sup]

    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
    out = os.path.join(outdir, f"perturb_ab_{regime}_{tag}.gif")
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--H", type=int, default=40)
    ap.add_argument("--time-stride", type=int, default=4)
    ap.add_argument("--delta", type=float, default=1.0,
                    help="perturbation in z-scored units (1.0 = 1 std of the field)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--outdir", default="examples/gray_scott/viz")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    jepa, _ = load_jepa(ckpt, DEVICE)
    dstc = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    decoder = build_decoder(dstc, DEVICE, ckpt_path=args.ckpt)
    ep = ckpt.get("epoch", "?")
    tag = args.tag or f"ep{ep}_d{args.delta}"
    os.makedirs(args.outdir, exist_ok=True)

    # Physical meaning of delta
    dA_phys = args.delta * float(STD[0])
    dB_phys = args.delta * float(STD[1])
    print(f"[perturb-ab] epoch={ep}  delta={args.delta}σ "
          f"= +{dA_phys:.3f} physical A / +{dB_phys:.3f} physical B", flush=True)

    files = sorted(glob.glob(os.path.join(ROOT, "data", args.split, "*.hdf5")))
    rng = np.random.default_rng(7)
    span = (C + args.H - 1) * args.time_stride + 1

    print("\n  regime      ||Δz_A||/||z||  ||Δz_B||/||z||")
    for path in files:
        fk = _parse_fk(path)
        regime = _parse_regime(path)
        with h5py.File(path, "r") as f:
            ntraj = f["t0_fields/A"].shape[0]
            x_np = _load_clip(f, ntraj, span, args.time_stride, rng)
        x = torch.from_numpy(x_np).unsqueeze(0)
        make_ab_gif(x, jepa, decoder, args.H, regime, fk,
                    args.delta, args.outdir, tag, fps=args.fps)

    print("[perturb-ab] done", flush=True)


if __name__ == "__main__":
    main()
