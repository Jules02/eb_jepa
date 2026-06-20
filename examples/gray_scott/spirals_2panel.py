"""2-panel GIF: Original | Init A+1σ, channel A only, spirals regime."""
import re, os, glob
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
DELTA  = 1.0
H      = 40
FPS    = 8
CKPT   = "/lustre/work/vivatech-jepadormi/aduplessi/checkpoints/gray_scott/dev/epoch_19.pth.tar"
OUTDIR = "examples/gray_scott/viz"


def _parse_fk(path):
    m = re.search(r'_F_([\d.]+)_k_([\d.]+)\.hdf5', path)
    return float(m.group(1)), float(m.group(2))


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


@torch.no_grad()
def decode(jepa, decoder, x):
    pred_z = rollout_latents(jepa, x.to(DEVICE), H, DEVICE)
    pred   = decoder(pred_z[:, :, C:])
    return _denorm(pred[0].cpu().numpy())   # [2, H, Hs, Ws]


def main():
    ckpt    = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    jepa, _ = load_jepa(ckpt, DEVICE)
    dstc    = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    decoder = build_decoder(dstc, DEVICE, ckpt_path=CKPT)
    os.makedirs(OUTDIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(ROOT, "data", "valid", "*.hdf5")))
    spirals_path = next(p for p in files if "spirals" in os.path.basename(p))
    fk = _parse_fk(spirals_path)

    rng  = np.random.default_rng(7)
    span = (C + H - 1) * 4 + 1
    with h5py.File(spirals_path, "r") as f:
        ntraj = f["t0_fields/A"].shape[0]
        x_np  = _load_clip(f, ntraj, span, 4, rng)

    x_base = torch.from_numpy(x_np).unsqueeze(0)
    x_A    = x_base.clone(); x_A[:, 0, :C] += DELTA

    orig  = decode(jepa, decoder, x_base)   # [2,H,Hs,Ws]
    pert  = decode(jepa, decoder, x_A)

    ch = 0   # channel A only
    vmin = min(orig[ch].min(), pert[ch].min())
    vmax = max(orig[ch].max(), pert[ch].max())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    fig.subplots_adjust(wspace=0.05, top=0.84, bottom=0.06, left=0.04, right=0.98)

    titles = ["Original", f"Init A+{DELTA:.1f}σ"]
    data   = [orig[ch], pert[ch]]

    ims = []
    for ax, title, d in zip(axes, titles, data):
        im = ax.imshow(d[0], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ims.append((im, d))

    sup = fig.suptitle("", fontsize=11)

    def update(f):
        for im, d in ims:
            im.set_data(d[f])
        sup.set_text(f"Spirals  F={fk[0]:.3f} k={fk[1]:.3f}   channel A   t={f+1}/{H}")
        return [im for im, _ in ims] + [sup]

    anim = animation.FuncAnimation(fig, update, frames=H, blit=False)
    out  = os.path.join(OUTDIR, "spirals_A_2panel.gif")
    anim.save(out, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
