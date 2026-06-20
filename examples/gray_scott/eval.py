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


class _FrameDecoderV2(nn.Module):
    """Deeper residual decoder (stem + nblocks residual blocks + head).

    Matches abenmanso's decoder architecture saved with keys stem/blocks/head.
    """
    def __init__(self, D, hid=128, nblocks=6):
        super().__init__()
        self.stem = nn.Conv2d(D, hid, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.GELU(),
                nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid),
            ) for _ in range(nblocks)
        ])
        self.act = nn.GELU()
        self.head = nn.Conv2d(hid, 2, 1)

    def forward(self, z):
        B, D, T, H, W = z.shape
        h = self.stem(z.permute(0, 2, 1, 3, 4).reshape(B * T, D, H, W))
        for blk in self.blocks:
            h = self.act(h + blk(h))
        out = self.head(h)
        return out.view(B, T, 2, H, W).permute(0, 2, 1, 3, 4)


def _train_decoder(decoder, jepa, encoder, device, epochs=40):
    """Train decoder (frozen JEPA) to minimise MSE(decode(encode(x)), x).

    Trains until the MSE plateaus (early stop) rather than a fixed 5 passes — the
    decoder's reconstruction error IS the VRMSE *floor*, so it must converge well
    below the persistence baseline for the rollout metric to mean anything.
    """
    dcfg = GrayScottConfig(split="train", epoch_size=4000, batch_size=8, num_workers=4)
    loader = make_loader(dcfg)
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    decoder.train()
    prev, patience = float("inf"), 0
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
        sched.step()
        mse = total / n
        print(f"[decoder] ep{ep:02d} mse={mse:.5f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
        # early stop: < 1% relative improvement for 4 consecutive epochs
        if (prev - mse) / (prev + 1e-8) < 0.01:
            patience += 1
            if patience >= 4:
                print(f"[decoder] converged at ep{ep}, stopping early", flush=True)
                break
        else:
            patience = 0
        prev = mse
    decoder.eval()


def build_decoder(dstc, device, ckpt_path=None):
    """Build (and optionally train) a latent->field decoder.

    If ``ckpt_path`` points to a file that contains a ``'decoder'`` key the
    weights are loaded directly (no training). Otherwise the decoder is trained
    from scratch against the frozen JEPA loaded from ``ckpt_path``."""
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "decoder" in ckpt:
            # Auto-detect architecture from state dict keys
            if any(k.startswith("stem") for k in ckpt["decoder"]):
                decoder = _FrameDecoderV2(D=dstc).to(device)
            else:
                decoder = _FrameDecoder(D=dstc).to(device)
            decoder.load_state_dict(ckpt["decoder"])
            print(f"[decoder] loaded weights from {ckpt_path}", flush=True)
            return decoder
    decoder = _FrameDecoder(D=dstc).to(device)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
def vrmse_per_horizon(jepa, encoder, decoder, loader, device, H, metric="vrmse"):
    """Per-horizon VRMSE, either The Well paper metric or the pooled diagnostic.

    metric='vrmse' (default, The Well App. E.3): per sample/channel
        sqrt(mean_space((pred-true)²) / (var_space(true) + 1e-7)), then mean over
        samples. Mean-of-ratios -> matches the paper, but a single near-uniform
        Gray-Scott frame (var≈0) blows the average up to 10²-10³ on low-F regimes.
    metric='pooled': aggregate sum_samples(MSE) and sum_samples(var) per
        horizon/channel, THEN ratio -> sqrt(ΣMSE/Σvar). Not the paper metric, but
        denominator-stable (no per-frame blow-up). Good for ranking regimes.

    Both average the two channels and also return per-channel '_u'/'_v' keys."""
    if metric not in ("vrmse", "pooled"):
        raise ValueError(f"metric must be 'vrmse' or 'pooled', got {metric!r}")
    NC = 2
    psum = {k: np.zeros((H, NC)) for k in _HEADLINE_KEYS}      # vrmse: Σ per-sample ratio
    pcnt = np.zeros(H)
    num = {k: np.zeros((H, NC)) for k in _HEADLINE_KEYS}       # pooled: Σ mse
    den = np.zeros((H, NC))                                     # pooled: Σ var (shared)

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
                if metric == "vrmse":
                    pv = torch.sqrt(mse / (true_var + 1e-7))          # [B,2]
                    psum[name][h] += pv.sum(dim=0).cpu().numpy()
                else:
                    num[name][h] += mse.sum(dim=0).double().cpu().numpy()

            _accum("jepa", pred_fields[:, :, h])
            _accum("persistence", last_ctx)

            z_true = encoder(true.unsqueeze(2))                  # [B,D,1,H,W]
            floor_field = decoder(z_true).squeeze(2)             # [B,2,H,W]
            _accum("floor", floor_field)
            den[h] += true_var.sum(dim=0).double().cpu().numpy()
            pcnt[h] += true.shape[0]

    if metric == "vrmse":
        per_ch = {k: psum[k] / np.maximum(pcnt[:, None], 1) for k in _HEADLINE_KEYS}
    else:
        per_ch = {k: np.sqrt(num[k] / np.maximum(den, 1e-12)) for k in _HEADLINE_KEYS}
    result = {k: per_ch[k].mean(axis=-1) for k in _HEADLINE_KEYS}
    for k in _HEADLINE_KEYS:
        result[f"{k}_u"] = per_ch[k][:, 0]
        result[f"{k}_v"] = per_ch[k][:, 1]
    return result


@torch.no_grad()
def vrmse_fixed(jepa, encoder, decoder, clips, device, H, bs=8, metric="vrmse"):
    """VRMSE over a FIXED clip set. metric='vrmse' = The Well paper (mean-of-ratios),
    'pooled' = stable diagnostic. Same jepa/persistence/floor, reproducible."""
    from examples.gray_scott.eval_common import make_vrmse, iter_batches
    accs = {k: make_vrmse(metric, H) for k in _HEADLINE_KEYS}   # jepa, persistence, floor
    for xb in iter_batches(clips, bs):
        x = xb.to(device)                                  # [B,2,C+H,H,W]
        last_ctx = x[:, :, C - 1]
        pred_z = rollout_latents(jepa, x, H, device)
        pred_fields = decoder(pred_z[:, :, C:])            # [B,2,H,Hs,Ws]
        for h in range(H):
            true = x[:, :, C + h]
            accs["jepa"].add(h, pred_fields[:, :, h], true)
            accs["persistence"].add(h, last_ctx, true)
            floor_field = decoder(encoder(true.unsqueeze(2))).squeeze(2)
            accs["floor"].add(h, floor_field, true)
    result = {}
    for k in _HEADLINE_KEYS:
        s = accs[k].scores()
        result[k] = s["all"]; result[f"{k}_u"] = s["u"]; result[f"{k}_v"] = s["v"]
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
    # split: "valid" for model selection, "test" for the final report (no leakage).
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "valid"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    jepa, encoder = load_jepa(ckpt, device)
    cfg = OmegaConf.create(ckpt["cfg"])
    dstc = int(cfg.model.dstc)
    # stride MUST match the model's training stride for the autoregressive rollout to be
    # physically meaningful (one predictor step = `stride` native steps). Default to the
    # checkpoint's own training stride; override with --stride only if you know what you do.
    stride = int(sys.argv[sys.argv.index("--stride") + 1]) if "--stride" in sys.argv \
        else int(cfg.data.get("time_stride", 4))
    decoder = build_decoder(dstc, device, ckpt_path=ckpt_path)
    n_clips = int(sys.argv[sys.argv.index("--n_clips") + 1]) if "--n_clips" in sys.argv else 400
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    metric = sys.argv[sys.argv.index("--metric") + 1] if "--metric" in sys.argv else "vrmse"
    print(f"[gs-eval] loaded (epoch {ckpt.get('epoch')}), split={split}, H={H}, "
          f"stride={stride}, n_clips={n_clips}, seed={seed}, metric={metric}", flush=True)

    # FIXED, shared eval set + chosen metric (default = The Well VRMSE) -> comparable
    from examples.gray_scott.eval_common import load_or_build_fixed_eval, default_cache_dir
    clips = load_or_build_fixed_eval(split, C + H, stride, n_clips, seed, default_cache_dir())
    scores = vrmse_fixed(jepa, encoder, decoder, clips, device, H, metric=metric)

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
