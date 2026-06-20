"""Quick visualization: original vs reconstruction for a validation clip.

Usage:
    python -m examples.gray_scott.visualize --ckpt <path> [--out recon.png] [--n_frames 6]
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader, MEAN, STD
from examples.gray_scott.eval import load_jepa, build_decoder, _FrameDecoder


def unnorm(x):
    """[2,T,H,W] z-scored -> raw concentrations."""
    m = torch.tensor(MEAN, device=x.device)[:, None, None, None]
    s = torch.tensor(STD, device=x.device)[:, None, None, None]
    return x * s + m


@torch.no_grad()
def make_figure(ckpt_path, n_frames=6, out="recon.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    jepa, encoder = load_jepa(ckpt, device)

    from omegaconf import OmegaConf
    dstc = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    decoder = build_decoder(dstc, device, ckpt_path=ckpt_path)
    decoder.eval()

    # grab one validation clip
    dcfg = GrayScottConfig(split="valid", n_frames=n_frames,
                           time_stride=4, epoch_size=16, batch_size=1, num_workers=0)
    loader = make_loader(dcfg, shuffle=False)
    x = next(iter(loader))["video"].to(device)   # [1, 2, T, H, W]

    z = encoder(x)                               # [1, D, T, H, W]
    recon = decoder(z)                           # [1, 2, T, H, W]

    orig  = unnorm(x[0].cpu())                   # [2, T, H, W]
    rec   = unnorm(recon[0].cpu())               # [2, T, H, W]

    channels = ["u (inhibitor)", "v (activator)"]
    T = orig.shape[1]
    fig, axes = plt.subplots(4, T, figsize=(2.5 * T, 10))
    fig.suptitle(f"Original vs Reconstruction  |  ckpt epoch {ckpt.get('epoch')}  |  D={dstc}", fontsize=12)

    for t in range(T):
        for ci, ch in enumerate(channels):
            row_orig = 2 * ci
            row_rec  = 2 * ci + 1
            vmin = float(orig[ci, t].min())
            vmax = float(orig[ci, t].max())

            axes[row_orig, t].imshow(orig[ci, t].numpy(), vmin=vmin, vmax=vmax, cmap="viridis")
            axes[row_rec,  t].imshow(rec[ci,  t].numpy(), vmin=vmin, vmax=vmax, cmap="viridis")

            if t == 0:
                axes[row_orig, t].set_ylabel(f"orig {ch}", fontsize=8)
                axes[row_rec,  t].set_ylabel(f"recon {ch}", fontsize=8)
            axes[row_orig, t].set_title(f"t={t}", fontsize=8)
            for ax in [axes[row_orig, t], axes[row_rec, t]]:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[viz] saved {out}", flush=True)

    # print per-frame reconstruction MSE
    mse = ((orig - rec) ** 2).mean(dim=(-2, -1))  # [2, T]
    for ci, ch in enumerate(channels):
        vals = " ".join(f"{v:.4f}" for v in mse[ci].tolist())
        print(f"[viz] recon MSE {ch}: {vals}", flush=True)


if __name__ == "__main__":
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    n    = int(sys.argv[sys.argv.index("--n_frames") + 1]) if "--n_frames" in sys.argv else 6
    out  = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "recon.png"
    make_figure(ckpt, n_frames=n, out=out)
