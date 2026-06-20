"""Gray-Scott — downstream evaluation (The Well's open question, in field space).

The Well asks: does latent prediction give more *stable* long-horizon rollouts
than the field-space neural-operator surrogates (FNO / U-Net)? To answer it we
roll the frozen JEPA predictor forward in LATENT space, DECODE each latent back
to a 2-channel field, and score multi-step VRMSE against ground truth and a
PERSISTENCE baseline (optionally vs FNO / U-Net surrogates).

The rollout-extraction harness is provided. What you implement (``# TODO``) is the
latent->field DECODER and the VRMSE metric that makes the comparison meaningful.

Run:  python -m examples.gray_scott.eval --ckpt <.../latest.pth.tar> --H 10
"""
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader
from examples.gray_scott.main import build_encoder, build_jepa

C = 2            # context_length (StateOnlyPredictor predicts from the previous 2 frames)
# The Well Table 3 evaluation windows (steps after context, 1-indexed)
WELL_WINDOWS = {"6:12": (5, 12), "13:30": (12, 30)}  # (start_idx, end_idx) inclusive, 0-indexed into H


def load_jepa(ckpt, device):
    """Provided: rebuild encoder + JEPA from a training checkpoint and freeze."""
    cfg = OmegaConf.create(ckpt["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    jepa = build_jepa(encoder, cfg.model).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    jepa.load_state_dict(ckpt["jepa"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa, encoder


@torch.no_grad()
def rollout_latents(jepa, x, H, device):
    """Provided: autoregressive latent rollout from C context frames.

    Feeds the first C frames of the clip and rolls the predictor forward H steps
    in latent space (``ctxt_window_time=C`` — the StateOnlyPredictor needs 2
    context frames, else the autoregressive loop yields an empty time axis).
    Returns the predicted latent sequence ``[B, D, C+H, h, w]``."""
    pred, _ = jepa.unroll(x[:, :, :C], actions=None, nsteps=H,
                          unroll_mode="autoregressive", ctxt_window_time=C,
                          compute_loss=False, return_all_steps=False)
    return pred


# --------------------------------------------------------------------------- #
# LATENT -> FIELD DECODER  — # TODO
# --------------------------------------------------------------------------- #
class _FrameDecoder(nn.Module):
    """Per-frame latent->field decoder: [B,D,T,H,W] -> [B,2,T,H,W]."""
    def __init__(self, D, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(D, hid, 3, padding=1), nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GELU(),
            nn.Conv2d(hid, 2, 1),
        )

    def forward(self, z):
        B, D, T, H, W = z.shape
        out = self.net(z.permute(0, 2, 1, 3, 4).reshape(B * T, D, H, W))
        return out.view(B, T, 2, H, W).permute(0, 2, 1, 3, 4)


def _train_decoder(decoder, jepa, encoder, device, epochs=5):
    """Train decoder (frozen JEPA) to minimise MSE(decode(encode(x)), x)."""
    dcfg = GrayScottConfig(split="train", epoch_size=2000, batch_size=8, num_workers=4)
    loader = make_loader(dcfg)
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    decoder.train()
    for ep in range(epochs):
        total, n = 0.0, 0
        for batch in loader:
            x = batch["video"].to(device)          # [B,2,T,H,W]
            with torch.no_grad():
                z = encoder(x)                     # [B,D,T,H,W]
            recon = decoder(z)                     # [B,2,T,H,W]
            loss = nn.functional.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item(); n += 1
        print(f"[decoder] ep{ep} mse={total/n:.4f}", flush=True)
    decoder.eval()


def build_decoder(dstc, device, ckpt_path=None):
    """Build (and optionally train) a latent->field decoder.

    If ``ckpt_path`` points to a file that contains a ``'decoder'`` key the
    weights are loaded directly (no training). Otherwise the decoder is trained
    from scratch against the frozen JEPA loaded from ``ckpt_path``."""
    decoder = _FrameDecoder(D=dstc).to(device)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "decoder" in ckpt:
            decoder.load_state_dict(ckpt["decoder"])
            print(f"[decoder] loaded weights from {ckpt_path}", flush=True)
            return decoder
        # No saved decoder weights — train from the checkpoint's frozen JEPA
        jepa, encoder = load_jepa(ckpt, device)
        _train_decoder(decoder, jepa, encoder, device)
        # Save decoder weights back into the checkpoint for next time
        ckpt["decoder"] = decoder.state_dict()
        torch.save(ckpt, ckpt_path)
        print(f"[decoder] weights saved to {ckpt_path}", flush=True)
    return decoder


# --------------------------------------------------------------------------- #
# METRIC  — # TODO
# --------------------------------------------------------------------------- #
_HEADLINE_KEYS = ("jepa", "persistence", "floor")


@torch.no_grad()
def vrmse_per_horizon(jepa, encoder, decoder, loader, device, H):
    """Per-horizon VRMSE using the paper's exact formula (mean-of-ratios).

    Per sample and channel: sqrt(mean_space((pred-true)²) / (var_space(true) + 1e-7)).
    Final score = mean over samples, then averaged over channels.
    Also returns per-channel '_u' and '_v' diagnostic keys."""
    NC = 2
    psum = {k: np.zeros((H, NC)) for k in _HEADLINE_KEYS}
    pcnt = np.zeros(H)

    for batch in loader:
        x = batch["video"].to(device)                            # [B,2,C+H,H,W]
        last_ctx = x[:, :, C - 1]                               # [B,2,H,W]

        pred_z = rollout_latents(jepa, x, H, device)            # [B,D,C+H,h,w]
        pred_fields = decoder(pred_z[:, :, C:])                  # [B,2,H,H,W]

        for h in range(H):
            true = x[:, :, C + h]                               # [B,2,H,W]
            true_var = true.var(dim=(-2, -1))                    # [B,2]

            def _accum(name, pred_hw):
                mse = ((pred_hw - true) ** 2).mean(dim=(-2, -1))     # [B,2]
                pv = torch.sqrt(mse / (true_var + 1e-7))              # [B,2]
                psum[name][h] += pv.sum(dim=0).cpu().numpy()

            _accum("jepa", pred_fields[:, :, h])
            _accum("persistence", last_ctx)

            z_true = encoder(true.unsqueeze(2))                  # [B,D,1,H,W]
            floor_field = decoder(z_true).squeeze(2)             # [B,2,H,W]
            _accum("floor", floor_field)
            pcnt[h] += true.shape[0]

    per_ch = {k: psum[k] / np.maximum(pcnt[:, None], 1) for k in _HEADLINE_KEYS}
    result = {k: per_ch[k].mean(axis=-1) for k in _HEADLINE_KEYS}
    for k in _HEADLINE_KEYS:
        result[f"{k}_u"] = per_ch[k][:, 0]
        result[f"{k}_v"] = per_ch[k][:, 1]
    return result


def window_vrmse(scores, window_name):
    """Average VRMSE over a named window. Returns all keys (headline + _u/_v)."""
    start, end = WELL_WINDOWS[window_name]
    H = scores[_HEADLINE_KEYS[0]].shape[0]
    end = min(end, H)
    return {k: float(scores[k][start:end].mean()) for k in scores}


def main():
    ckpt_path = sys.argv[sys.argv.index("--ckpt") + 1]
    H = int(sys.argv[sys.argv.index("--H") + 1]) if "--H" in sys.argv else 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    jepa, encoder = load_jepa(ckpt, device)
    dstc = int(OmegaConf.create(ckpt["cfg"]).model.dstc)
    decoder = build_decoder(dstc, device, ckpt_path=ckpt_path)
    print(f"[gs-eval] loaded (epoch {ckpt.get('epoch')}), H={H}", flush=True)

    dcfg = GrayScottConfig(split="valid", n_frames=C + H, time_stride=4,
                           epoch_size=400, batch_size=8, num_workers=8)
    loader = make_loader(dcfg, shuffle=False)
    scores = vrmse_per_horizon(jepa, encoder, decoder, loader, device, H)

    # Per-horizon headlines
    for name in _HEADLINE_KEYS:
        arr = scores[name]
        print(f"   {name:14s} h1={arr[0]:.3f} h{H}={arr[-1]:.3f} | {np.round(arr, 3).tolist()}", flush=True)
    # Per-channel diagnostics (jepa and floor only)
    print("   --- per channel ---", flush=True)
    for name in ("jepa", "floor"):
        for ch in ("u", "v"):
            arr = scores[f"{name}_{ch}"]
            print(f"   {name}_{ch:11s} h1={arr[0]:.3f} h{H}={arr[-1]:.3f}", flush=True)

    # The Well Table 3 windows
    print("\n   === The Well Table 3 comparison ===", flush=True)
    for wname in WELL_WINDOWS:
        start, end = WELL_WINDOWS[wname]
        if end <= H:
            w = window_vrmse(scores, wname)
            headline = "  ".join(f"{k}={w[k]:.3f}" for k in _HEADLINE_KEYS)
            print(f"   window {wname}: {headline}", flush=True)
            print(f"      jepa_u={w['jepa_u']:.3f}  jepa_v={w['jepa_v']:.3f}  "
                  f"floor_u={w['floor_u']:.3f}  floor_v={w['floor_v']:.3f}", flush=True)


if __name__ == "__main__":
    main()
