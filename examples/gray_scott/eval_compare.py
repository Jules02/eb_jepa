"""Compare JEPA vs UNet baselines on gray_scott — same metric, same protocol.

Metric: pooled VRMSE = sqrt(Σ_samples MSE_i / Σ_samples var_i) per horizon.
Stable on flat/dissolving samples (down-weights them naturally vs mean-of-ratios).
Predicting the spatial mean gives pooled VRMSE = 1.0 (same interpretation as paper).

Both JEPA and UNet are evaluated AUTOREGRESSIVELY at stride=4 (40-second steps).

Usage (from repo root):
  uv run --with "neuraloperator==0.3.0" --with torch-harmonics \\
         --with timm --with einops \\
         python examples/gray_scott/eval_compare.py --ckpt <jepa_checkpoint>
"""
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf

from examples.gray_scott._well_baselines import stub_heavy_well_models

# Stub heavy unused models so the_well imports without needing timm etc.
stub_heavy_well_models()

from the_well.benchmark.models.unet_classic import UNetClassic
from the_well.benchmark.models.unet_convnext import UNetConvNext

from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader
from examples.gray_scott.eval import C, load_jepa, build_decoder, rollout_latents

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NC = 2
H = 30
C_UNET = 4          # The Well config: n_steps_input=4
HF_SUFFIX = "gray_scott_reaction_diffusion"
WELL_WINDOWS = {"6:12": (5, 12), "13:30": (12, 30)}


# ── Pooled VRMSE accumulator ─────────────────────────────────────────────────

class PooledAccum:
    """Accumulates MSE numerator and variance denominator separately per horizon.

    Final score per horizon h:
        pooled_vrmse(h) = sqrt( Σ_i mse_i(h) / Σ_i var_i(h) )
    where the sums are over all samples i and channels.
    Predicting the spatial mean gives pooled VRMSE = 1.0.
    """
    def __init__(self, H, NC):
        self.num = np.zeros((H, NC))
        self.den = np.zeros((H, NC))

    def add(self, h, pred, true):
        """pred, true: [B, NC, H, W] channels-first."""
        mse = ((pred - true) ** 2).mean(dim=(-2, -1))    # [B, NC]
        var = true.var(dim=(-2, -1))                       # [B, NC]
        self.num[h] += mse.sum(dim=0).cpu().numpy()
        self.den[h] += var.sum(dim=0).cpu().numpy()

    def result(self):
        """Returns [H, NC] pooled VRMSE."""
        return np.sqrt(self.num / np.maximum(self.den, 1e-8))


# ── UNet autoregressive rollout ───────────────────────────────────────────────

@torch.no_grad()
def unet_rollout_preds(model, x_ctx, H):
    """Autoregressive rollout from ground-truth context.

    x_ctx: [B, C_UNET, H, W, NC] channels-last
    Returns: [B, H, NC, Hs, Ws] channels-first predictions
    """
    preds = []
    ctx = x_ctx.clone()
    for _ in range(H):
        inp = ctx.permute(0, 1, 4, 2, 3).flatten(1, 2)   # [B, C*NC, H, W]
        out_cf = model(inp)                                # [B, NC, H, W]
        out_cl = out_cf.permute(0, 2, 3, 1)               # [B, H, W, NC]
        preds.append(out_cf.unsqueeze(1))
        ctx = torch.cat([ctx[:, 1:], out_cl.unsqueeze(1)], dim=1)
    return torch.cat(preds, dim=1)                        # [B, H, NC, Hs, Ws]


# ── Main evaluation loop ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_all(jepa, encoder, decoder, loader):
    unet = UNetClassic.from_pretrained(
        f"polymathic-ai/UNetClassic-{HF_SUFFIX}").to(DEVICE).eval()
    cnext = UNetConvNext.from_pretrained(
        f"polymathic-ai/UNetConvNext-{HF_SUFFIX}").to(DEVICE).eval()
    print("[compare] loaded UNetClassic and UNetConvNext", flush=True)

    accums = {k: PooledAccum(H, NC)
              for k in ("jepa", "unet_classic", "cnext_unet", "persistence")}

    for batch in loader:
        x = batch["video"].to(DEVICE)           # [B, NC, T, H, W]
        B, _, T, Hs, Ws = x.shape

        # JEPA: autoregressive latent rollout from first C=2 frames
        pred_z = rollout_latents(jepa, x, H, DEVICE)      # [B, D, C+H, h, w]
        jepa_preds = decoder(pred_z[:, :, C:])             # [B, NC, H, Hs, Ws]

        # UNet: autoregressive from first C_UNET=4 ground-truth frames (stride=4)
        x_cl = x.permute(0, 2, 3, 4, 1)                   # [B, T, Hs, Ws, NC]
        unet_preds = unet_rollout_preds(unet, x_cl[:, :C_UNET], H)
        cnext_preds = unet_rollout_preds(cnext, x_cl[:, :C_UNET], H)

        last_ctx = x[:, :, C - 1]                          # [B, NC, Hs, Ws]

        for h in range(H):
            # Reference frame: C_UNET + h (so both models predict the same future frames)
            ref_idx = C_UNET + h
            if ref_idx >= T:
                break
            true = x[:, :, ref_idx]                        # [B, NC, Hs, Ws]

            # JEPA predicted frame at same absolute position
            # JEPA context ends at C, UNet at C_UNET — offset to align
            jepa_h = h + (C_UNET - C)
            if 0 <= jepa_h < H:
                accums["jepa"].add(h, jepa_preds[:, :, jepa_h], true)

            accums["unet_classic"].add(h, unet_preds[:, h], true)
            accums["cnext_unet"].add(h, cnext_preds[:, h], true)
            accums["persistence"].add(h, last_ctx, true)

    return {k: v.result() for k, v in accums.items()}     # [H, NC] each


def window(arr, s, e):
    e = min(e, H)
    return float(arr[s:e].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--time-stride", type=int, default=4)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    jepa, encoder = load_jepa(ckpt, DEVICE)
    dstc = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    decoder = build_decoder(dstc, DEVICE, ckpt_path=args.ckpt)
    print(f"[compare] JEPA epoch={ckpt.get('epoch')}, D={dstc}, stride={args.time_stride}", flush=True)

    n_frames = C_UNET + H
    dcfg = GrayScottConfig(split="valid", n_frames=n_frames, time_stride=args.time_stride,
                           epoch_size=400, batch_size=8, num_workers=8)
    loader = make_loader(dcfg, shuffle=False)

    per_ch = eval_all(jepa, encoder, decoder, loader)

    print(f"\nPooled VRMSE — autoregressive, stride={args.time_stride} ({args.time_stride*10}s/step)")
    print(f"{'Method':15s}  {'h=1':>7}  {'[6:12]':>7}  {'[13:30]':>8}")
    print("-" * 52)
    for name, pch in per_ch.items():
        avg = pch.mean(axis=-1)
        print(f"{name:15s}  {avg[0]:7.3f}  {window(avg,5,12):7.3f}  {window(avg,12,30):8.3f}",
              flush=True)
        print(f"   A: h=1={pch[0,0]:.3f}  B: h=1={pch[0,1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
