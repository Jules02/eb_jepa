"""Train a small MLP probe on frozen JEPA encoder latents to predict F and k.

The Gray-Scott dataset has 6 regimes, each with fixed (F, k) parameters encoded
in the filename. This tests whether the encoder captures the underlying physics.

Protocol:
  - Encoder frozen (JEPA checkpoint)
  - Latents pooled over (T, h, w) -> [B, D]
  - 2-layer MLP predicts (F, k) via MSE
  - Report R² on validation set

Usage:
  uv run python examples/gray_scott/probe.py --ckpt <path/to/epoch_19.pth.tar>
"""
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import (
    GrayScottConfig, GrayScottDataset, NT, MEAN, STD
)
from examples.gray_scott.eval import C, load_jepa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset with (F, k) labels ────────────────────────────────────────────────

def _parse_fk(path):
    m = re.search(r'_F_([\d.]+)_k_([\d.]+)\.hdf5', path)
    return float(m.group(1)), float(m.group(2))


class FKDataset(GrayScottDataset):
    """GrayScottDataset that also returns the (F, k) regime parameters."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fk = np.array([_parse_fk(p) for p in self.files], dtype=np.float32)
        print(f"[probe] {cfg.split}: {len(self.files)} regimes — "
              + ", ".join(f"F={fk[0]:.3f}/k={fk[1]:.3f}" for fk in self.fk))

    def __getitem__(self, idx):
        import numpy.random as npr
        self._rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        fi = int(self._rng.integers(len(self.files)))
        f = self._h(self.files[fi])
        tr = int(self._rng.integers(self.ntraj[fi]))
        t0 = int(self._rng.integers(0, NT - self.span + 1))
        sl = slice(t0, t0 + self.span, self.cfg.time_stride)
        A = f["t0_fields/A"][tr, sl]
        B = f["t0_fields/B"][tr, sl]
        x = np.stack([A, B], axis=0).astype(np.float32)
        x = (x - MEAN[:, None, None, None]) / STD[:, None, None, None]
        return {"video": torch.from_numpy(x),
                "fk":   torch.from_numpy(self.fk[fi])}


def make_fk_loader(cfg, shuffle=True):
    return torch.utils.data.DataLoader(
        FKDataset(cfg), batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=cfg.num_workers > 0)


# ── Probe model ────────────────────────────────────────────────────────────────

class FKProbe(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 64), nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, z):
        return self.net(z)


# ── Encode batch -> pooled latent ─────────────────────────────────────────────

@torch.no_grad()
def encode(encoder, x):
    """x: [B, 2, T, H, W] -> [B, D] (mean-pooled over T, h, w)."""
    z = encoder(x)           # [B, D, T, h, w]  (TemporalBatchMixin)
    return z.mean(dim=(2, 3, 4))


# ── Train/eval loops ──────────────────────────────────────────────────────────

def r2(pred, target):
    """R² per output dimension."""
    ss_res = ((pred - target) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0)
    return (1 - ss_res / ss_tot.clamp(min=1e-8)).cpu().numpy()


def train_probe(encoder, probe, loader, opt, n_steps):
    probe.train()
    losses = []
    step = 0
    while step < n_steps:
        for batch in loader:
            x   = batch["video"].to(DEVICE)   # [B, 2, T, H, W]
            fk  = batch["fk"].to(DEVICE)      # [B, 2]
            z   = encode(encoder, x)           # [B, D]
            pred = probe(z)
            loss = nn.functional.mse_loss(pred, fk)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            step += 1
            if step >= n_steps:
                break
    return float(np.mean(losses))


@torch.no_grad()
def eval_probe(encoder, probe, loader):
    probe.eval()
    preds, targets = [], []
    for batch in loader:
        x  = batch["video"].to(DEVICE)
        fk = batch["fk"].to(DEVICE)
        z  = encode(encoder, x)
        preds.append(probe(z))
        targets.append(fk)
    preds   = torch.cat(preds)
    targets = torch.cat(targets)
    mse = nn.functional.mse_loss(preds, targets, reduction="none").mean(dim=0).cpu().numpy()
    r2s = r2(preds, targets)
    mae = (preds - targets).abs().mean(dim=0).cpu().numpy()
    return mse, r2s, mae, preds.cpu().numpy(), targets.cpu().numpy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--time-stride", type=int, default=4)
    ap.add_argument("--n-frames", type=int, default=4, help="frames per clip fed to encoder")
    ap.add_argument("--steps", type=int, default=2000, help="probe training steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epoch-size", type=int, default=4000)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    _, encoder = load_jepa(ckpt, DEVICE)
    encoder.eval()
    D = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    print(f"[probe] encoder D={D}, epoch={ckpt.get('epoch')}, stride={args.time_stride}")

    train_cfg = GrayScottConfig(split="train", n_frames=args.n_frames,
                                time_stride=args.time_stride,
                                epoch_size=args.epoch_size,
                                batch_size=args.batch_size, num_workers=4)
    valid_cfg = GrayScottConfig(split="valid", n_frames=args.n_frames,
                                time_stride=args.time_stride,
                                epoch_size=1200, batch_size=args.batch_size, num_workers=4)

    train_loader = make_fk_loader(train_cfg, shuffle=True)
    valid_loader = make_fk_loader(valid_cfg, shuffle=False)

    probe = FKProbe(D).to(DEVICE)
    opt   = torch.optim.Adam(probe.parameters(), lr=args.lr)

    print(f"\nTraining probe for {args.steps} steps...")
    for phase in range(5):
        steps_this = args.steps // 5
        loss = train_probe(encoder, probe, train_loader, opt, steps_this)
        mse, r2s, mae, _, _ = eval_probe(encoder, probe, valid_loader)
        done = (phase + 1) * steps_this
        print(f"  step {done:4d}  train_loss={loss:.6f}  "
              f"val R²(F)={r2s[0]:.4f}  R²(k)={r2s[1]:.4f}  "
              f"MAE(F)={mae[0]:.5f}  MAE(k)={mae[1]:.5f}", flush=True)

    print("\nFinal validation results:")
    mse, r2s, mae, preds, targets = eval_probe(encoder, probe, valid_loader)
    print(f"  R²   — F: {r2s[0]:.4f}   k: {r2s[1]:.4f}   mean: {r2s.mean():.4f}")
    print(f"  MAE  — F: {mae[0]:.5f}  k: {mae[1]:.5f}")
    print(f"  MSE  — F: {mse[0]:.6f}  k: {mse[1]:.6f}")

    # Show per-regime breakdown
    print("\nPer-regime predictions (val):")
    unique = np.unique(targets, axis=0)
    print(f"  {'Regime':20s}  {'True F':>8}  {'True k':>8}  {'Pred F':>8}  {'Pred k':>8}")
    for fk_true in unique:
        mask = np.all(targets == fk_true, axis=1)
        p = preds[mask].mean(axis=0)
        print(f"  {'F='+str(fk_true[0])+'/k='+str(fk_true[1]):20s}  "
              f"{fk_true[0]:8.4f}  {fk_true[1]:8.4f}  {p[0]:8.4f}  {p[1]:8.4f}")


if __name__ == "__main__":
    main()
