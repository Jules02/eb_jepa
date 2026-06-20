"""Iso-protocol field-space baselines for Gray-Scott VRMSE.

Every baseline predicts the FIELD directly (no encoder/decoder), so none of them carry a
decoder floor — the honest comparison is jepa(+floor) vs these. The VRMSE metric, data,
z-score normalization, context length (C=2), rollout stride and test split all match
examples.gray_scott.eval, so the numbers drop straight into the same table.

Baselines:
  free (no training):
    persistence   x_{t+1} = x_t
    linear        x_{t+1} = x_t + (x_t - x_{t-1})     (stronger trivial bar for slow PDEs)
    climatology   x_{t+1} = mean = 0  (data is z-scored)  -> VRMSE ~1 sanity floor
  trained (iso-protocol, one-step MSE, autoregressive rollout):
    unet          ResUNet on stacked context fields  [B,2*C,H,W] -> [B,2,H,W]
    fno           FNO2d (spectral) — the strong baseline (diffusion is diagonal in Fourier)

Run:  python -m examples.gray_scott.baselines --split test --H 30 [--epochs 20] [--which all]
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.architectures import ResUNet
from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader

C = 2                                              # context length (matches eval.py)
WELL_WINDOWS = {"6:12": (5, 12), "13:30": (12, 30)}  # mirrors eval.py


# --------------------------------------------------------------------------- #
# FNO (2D Fourier Neural Operator)
# --------------------------------------------------------------------------- #
class SpectralConv2d(nn.Module):
    """Multiply the lowest Fourier modes by learnable complex weights."""
    def __init__(self, in_c, out_c, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_c * out_c)
        self.w1 = nn.Parameter(scale * torch.rand(in_c, out_c, modes1, modes2, 2))
        self.w2 = nn.Parameter(scale * torch.rand(in_c, out_c, modes1, modes2, 2))

    def _mul(self, inp, w):
        return torch.einsum("bixy,ioxy->boxy", inp, torch.view_as_complex(w.contiguous()))

    def forward(self, x):
        B, _, H, W = x.shape
        m1, m2 = self.modes1, self.modes2
        xft = torch.fft.rfft2(x)                                    # [B,C,H,W//2+1]
        out = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :m1, :m2] = self._mul(xft[:, :, :m1, :m2], self.w1)
        out[:, :, -m1:, :m2] = self._mul(xft[:, :, -m1:, :m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, in_c, out_c, width=32, modes=16, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_c, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, out_c, 1))

    def forward(self, x):
        x = self.lift(x)
        for s, w in zip(self.spectral, self.w):
            x = F.gelu(s(x) + w(x))
        return self.proj(x)


# --------------------------------------------------------------------------- #
# step functions: [B,2,C,H,W] context -> [B,2,H,W] next field
# --------------------------------------------------------------------------- #
def _stack(ctx):                                   # [B,2,C,H,W] -> [B,2*C,H,W]
    return ctx.flatten(1, 2)


def step_persistence(ctx):
    return ctx[:, :, -1]


def step_linear(ctx):
    return 2 * ctx[:, :, -1] - ctx[:, :, -2]


def step_climatology(ctx):
    return torch.zeros_like(ctx[:, :, -1])


def step_model(model):
    def f(ctx):
        return model(_stack(ctx))
    return f


# --------------------------------------------------------------------------- #
# training (iso-protocol: one-step next-field MSE) + autoregressive VRMSE rollout
# --------------------------------------------------------------------------- #
def train_field_model(model, device, stride, epochs, lr, tag):
    dcfg = GrayScottConfig(split="train", n_frames=C + 1, time_stride=stride,
                           epoch_size=8000, batch_size=16, num_workers=8)
    loader = make_loader(dcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for batch in loader:
            x = batch["video"].to(device)          # [B,2,C+1,H,W]
            ctx, tgt = x[:, :, :C], x[:, :, C]      # [B,2,C,H,W], [B,2,H,W]
            pred = model(_stack(ctx))
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); n += 1
        sched.step()
        print(f"[{tag}] ep{ep:02d} mse={tot / n:.5f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
    model.eval()


def baseline_weights_path(stride, name):
    base = os.environ.get("EBJEPA_CKPTS", os.path.join(os.getcwd(), "_ckpts"))
    d = os.path.join(base, "gray_scott", "baselines")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}_s{stride}.pt")


def load_or_train(name, model, device, stride, epochs, lr=1e-3):
    """Load cached weights for this (name, stride) if present, else train + save them once."""
    path = baseline_weights_path(stride, name)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"[{name}] loaded cached weights from {path}", flush=True)
        return model
    train_field_model(model, device, stride, epochs, lr, name)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)                              # atomic (safe if jobs run in parallel)
    print(f"[{name}] saved weights -> {path}", flush=True)
    return model


@torch.no_grad()
def vrmse_rollout(step_fn, clips, device, H, bs=8, metric="vrmse"):
    """Autoregressive field-space VRMSE over the FIXED clip set — same clips + metric as
    examples.gray_scott.eval.vrmse_fixed (metric: 'vrmse' = The Well paper, or 'pooled')."""
    from examples.gray_scott.eval_common import make_vrmse, iter_batches
    acc = make_vrmse(metric, H)
    for xb in iter_batches(clips, bs):
        x = xb.to(device)                          # [B,2,C+H,H,W]
        ctx = x[:, :, :C].clone()                  # sliding buffer [B,2,C,H,W]
        for h in range(H):
            pred = step_fn(ctx)                     # [B,2,H,W]
            acc.add(h, pred, x[:, :, C + h])
            ctx = torch.cat([ctx[:, :, 1:], pred.unsqueeze(2)], dim=2)
    return acc.scores()


def _window(arr, win):
    s, e = win
    return float(arr[s:min(e, arr.shape[0])].mean())


# --------------------------------------------------------------------------- #
def main():
    a = sys.argv
    def opt(flag, default, cast=str):
        return cast(a[a.index(flag) + 1]) if flag in a else default
    H = opt("--H", 30, int)
    split = opt("--split", "test")
    stride = opt("--stride", 4, int)
    epochs = opt("--epochs", 20, int)
    which = opt("--which", "all")
    n_clips = opt("--n_clips", 400, int)
    seed = opt("--seed", 0, int)
    metric = opt("--metric", "vrmse")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baselines] split={split} H={H} stride={stride} epochs={epochs} "
          f"which={which} n_clips={n_clips} seed={seed} metric={metric}", flush=True)

    # FIXED, shared eval set (same clips as examples.gray_scott.eval -> comparable tables)
    from examples.gray_scott.eval_common import load_or_build_fixed_eval, default_cache_dir
    clips = load_or_build_fixed_eval(split, C + H, stride, n_clips, seed, default_cache_dir())

    scores = {}   # name -> per-horizon array (mean over channels), + _u/_v

    def add(name, res):
        scores[name] = res["all"]
        scores[name + "_u"] = res["u"]; scores[name + "_v"] = res["v"]

    # ---- free baselines ----
    add("persistence", vrmse_rollout(step_persistence, clips, device, H, metric=metric))
    add("linear", vrmse_rollout(step_linear, clips, device, H, metric=metric))
    add("climatology", vrmse_rollout(step_climatology, clips, device, H, metric=metric))

    # ---- trained baselines ----
    sel = ["unet", "fno"] if which == "all" else which.split(",")
    if "unet" in sel:
        unet = ResUNet(in_d=2 * C, h_d=32, out_d=2, norm="group").to(device)
        print(f"[unet] params: {sum(p.numel() for p in unet.parameters())/1e6:.2f}M", flush=True)
        load_or_train("unet", unet, device, stride, epochs)
        add("unet", vrmse_rollout(step_model(unet), clips, device, H, metric=metric))
    if "fno" in sel:
        fno = FNO2d(in_c=2 * C, out_c=2, width=32, modes=16, n_layers=4).to(device)
        print(f"[fno] params: {sum(p.numel() for p in fno.parameters())/1e6:.2f}M", flush=True)
        load_or_train("fno", fno, device, stride, epochs)
        add("fno", vrmse_rollout(step_model(fno), clips, device, H, metric=metric))

    # ---- report ----
    names = [n for n in ["persistence", "linear", "climatology", "unet", "fno"] if n in scores]
    print(f"\n=== Gray-Scott baselines | split={split} H={H} stride={stride} ===", flush=True)
    print(f"{'baseline':14s} {'h1':>8s} {'h'+str(H):>8s}")
    for n in names:
        print(f"{n:14s} {scores[n][0]:8.3f} {scores[n][-1]:8.3f}")
    print("\n--- The Well windows (mean VRMSE over window) ---")
    for wname, win in WELL_WINDOWS.items():
        cells = "  ".join(f"{n}={_window(scores[n], win):.3f}" for n in names)
        print(f"window {wname}: {cells}")
    print("\n(reminder: these are FLOOR-FREE field predictors; the JEPA carries its decoder "
          "floor, so compare jepa-minus-floor against these.)", flush=True)


if __name__ == "__main__":
    main()
