"""Distributional + perceptual field metrics for Gray-Scott rollouts, per phase / horizon.

Three metrics alongside VRMSE, on the decoded physical fields (JEPA via decoder + baselines):
  - w2_spatial : sliced-Wasserstein over 2D pixel positions (mass = concentration), per
                 channel A/B, per frame -> measures spatial layout / drift. Reported per horizon.
  - w2_values  : sliced-Wasserstein in the (A,B) VALUE space, pooled over the whole
                 space-time cube -> concentration statistics, INVARIANT to spatial/temporal
                 phase & frequency (no position/time in the ground metric -> no boundary issue).
  - vgg_style  : Gatys style distance. Decode -> RGB (A->R, B->B, green=(A+B)/2) -> ImageNet
                 norm -> VGG19 Gram matrices (conv1_1..conv5_1) -> MSE. Per frame, per horizon.

Run: python -m examples.gray_scott.field_metrics --ckpt <jepa.pth.tar> [--per_regime 20] [--H 60]
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torchvision

from examples.gray_scott.eval import load_jepa, build_decoder, rollout_latents, C
from examples.gray_scott.eval_common import build_regime_clips, iter_batches
from examples.gray_scott.baselines import (
    FNO2d, load_or_train, step_model, step_persistence, step_linear)
from eb_jepa.architectures import ResUNet
from eb_jepa.datasets.gray_scott.dataset import MEAN, STD

HZ = [1, 15, 30, 60]
MODELS = ["jepa", "unet", "fno", "persistence", "linear"]   # climatology dropped
PLOT_MODELS = ["jepa", "unet", "fno"]        # persistence/linear dropped from field_*.png
_mean_t = torch.tensor(MEAN).view(1, 2, 1, 1)
_std_t = torch.tensor(STD).view(1, 2, 1, 1)


def unz(x):                                    # z-scored [B,2,...] -> physical, clipped >=0
    return (x.cpu() * _std_t.unsqueeze(-1) + _mean_t.unsqueeze(-1)).clamp(min=0)


@torch.no_grad()
def sliced_w2(X, Y, n_proj=128):
    """Sliced-W2 between two point clouds X,Y [N,d] (same N)."""
    d = X.shape[1]
    th = torch.randn(d, n_proj, device=X.device); th /= th.norm(dim=0, keepdim=True)
    px = (X @ th).sort(dim=0).values
    py = (Y @ th).sort(dim=0).values
    return float(torch.sqrt(((px - py) ** 2).mean()))


def _sample_spatial(field, N, device):
    """field [Hs,Ws] >=0 -> [N,2] sampled pixel coords in [0,1] (prob ∝ mass)."""
    Hs, Ws = field.shape
    f = field.flatten().to(device) + 1e-8
    idx = torch.multinomial(f, N, replacement=True)
    return torch.stack([(idx % Ws).float() / Ws, (idx // Ws).float() / Hs], 1)


def w2_spatial(pred_phys, true_phys, device, N=2048):
    """pred/true [2,Hs,Ws] physical -> mean over channels of spatial sliced-W2."""
    vals = []
    for c in range(2):
        X = _sample_spatial(pred_phys[c], N, device)
        Y = _sample_spatial(true_phys[c], N, device)
        vals.append(sliced_w2(X, Y))
    return float(np.mean(vals))


def w2_values(pred_cube, true_cube, device, N=8192):
    """pred/true [2,T,Hs,Ws] physical -> sliced-W2 of (A,B) clouds pooled over space-time."""
    P = pred_cube.permute(1, 2, 3, 0).reshape(-1, 2)         # [T*Hs*Ws, 2]
    Q = true_cube.permute(1, 2, 3, 0).reshape(-1, 2)
    ip = torch.randint(0, P.shape[0], (N,)); iq = torch.randint(0, Q.shape[0], (N,))
    return sliced_w2(P[ip].to(device), Q[iq].to(device))


def to_rgb(phys):                              # [B,2,Hs,Ws] physical -> [B,3] : R=A, G=B, B=0
    A, B = phys[:, 0].clamp(0, 1), phys[:, 1].clamp(0, 1)
    return torch.stack([A, B, torch.zeros_like(A)], 1)


class VGGStyle:
    def __init__(self, device):
        w = torchvision.models.VGG19_Weights.IMAGENET1K_V1
        self.vgg = torchvision.models.vgg19(weights=w).features.eval().to(device)
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.layers = {1, 6, 11, 20, 29}       # conv1_1..conv5_1 (post-ReLU)
        self.m = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.s = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _gram(self, f):
        B, Ch, H, W = f.shape
        f = f.reshape(B, Ch, H * W)
        return (f @ f.transpose(1, 2)) / (Ch * H * W)

    @torch.no_grad()
    def dist(self, rgb_pred, rgb_true):        # [B,3,H,W] in [0,1] -> [B] style distance
        a = (rgb_pred - self.m) / self.s
        b = (rgb_true - self.m) / self.s
        d = torch.zeros(a.shape[0], device=a.device)
        for i, layer in enumerate(self.vgg):
            a = layer(a); b = layer(b)
            if i in self.layers:
                d = d + ((self._gram(a) - self._gram(b)) ** 2).mean(dim=(-2, -1))
            if i >= max(self.layers):
                break
        return d


def main():
    a = sys.argv
    def opt(f, d, c=str):
        return c(a[a.index(f) + 1]) if f in a else d
    ckpt_path = a[a.index("--ckpt") + 1]
    per_regime = opt("--per_regime", 20, int)
    H = opt("--H", 60, int)
    split = opt("--split", "test")
    out_dir = opt("--out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"]); stride = int(cfg.data.get("time_stride", 4))
    jepa, encoder = load_jepa(ckpt, device)
    decoder = build_decoder(int(cfg.model.dstc), device, ckpt_path=ckpt_path)
    clips, tags, titles = build_regime_clips(split, C + H, stride, per_regime)
    phases = np.array([t.split()[0] for t in titles])
    PHASES = list(dict.fromkeys(phases.tolist()))
    unet = ResUNet(in_d=2 * C, h_d=32, out_d=2, norm="group").to(device); load_or_train("unet", unet, device, stride, 20)
    fno = FNO2d(in_c=2 * C, out_c=2, width=32, modes=16, n_layers=4).to(device); load_or_train("fno", fno, device, stride, 20)
    vgg = VGGStyle(device)
    print(f"[fld] {len(clips)} clips x H={H} over {PHASES}", flush=True)

    @torch.no_grad()
    def predict_fields(model, x):
        if model == "jepa":
            return decoder(rollout_latents(jepa, x, H, device)[:, :, C:])      # [B,2,H,Hs,Ws]
        step = {"unet": step_model(unet), "fno": step_model(fno),
                "persistence": step_persistence, "linear": step_linear}[model]
        ctx = x[:, :, :C].clone(); out = []
        for _ in range(H):
            p = step(ctx); out.append(p); ctx = torch.cat([ctx[:, :, 1:], p.unsqueeze(2)], 2)
        return torch.stack(out, dim=2)

    # metric storage: metric -> model -> list over clips (w2v: scalar; w2s/vgg: [len(HZ)])
    W2S = {m: [] for m in MODELS}; W2V = {m: [] for m in MODELS}; VGG = {m: [] for m in MODELS}
    clip_phase = []
    bs = 8
    for bi, xb in enumerate(iter_batches(clips, bs)):
        x = xb.to(device)
        tf = unz(x[:, :, C:])                                                  # [b,2,H,Hs,Ws] phys
        for m in MODELS:
            pf = unz(predict_fields(m, x))
            for i in range(x.shape[0]):
                W2V[m].append(w2_values(pf[i], tf[i], device))
                ws, vg = [], []
                # VGG batched over the HZ frames of this clip
                hp = torch.stack([pf[i, :, h - 1] for h in HZ]).to(device)     # [4,2,Hs,Ws]
                ht = torch.stack([tf[i, :, h - 1] for h in HZ]).to(device)
                vd = vgg.dist(to_rgb(hp), to_rgb(ht)).cpu().numpy()            # [4]
                for j, h in enumerate(HZ):
                    ws.append(w2_spatial(pf[i, :, h - 1], tf[i, :, h - 1], device))
                W2S[m].append(ws); VGG[m].append(vd.tolist())
        clip_phase.extend([phases[bi * bs + i] for i in range(x.shape[0])])
        print(f"[fld] batch {bi+1} done", flush=True)
    clip_phase = np.array(clip_phase)
    for d in (W2S, W2V, VGG):
        for m in MODELS:
            d[m] = np.array(d[m])

    np.savez(os.path.join(out_dir, "field_metrics_data.npz"), phases=clip_phase, HZ=np.array(HZ),
             **{f"w2s_{m}": W2S[m] for m in MODELS},
             **{f"w2v_{m}": W2V[m] for m in MODELS},
             **{f"vgg_{m}": VGG[m] for m in MODELS})
    print(f"[fld] saved {out_dir}/field_metrics_data.npz", flush=True)

    # ---- tables ----
    def tbl(name, D, hcol):
        print(f"\n=== {name} (median over clips) ===", flush=True)
        if hcol:
            print(f"{'phase':10s} {'model':12s} " + "  ".join(f"h{h:>3}" for h in HZ))
            for p in PHASES:
                for m in MODELS:
                    v = np.median(D[m][clip_phase == p], 0)
                    print(f"{p:10s} {m:12s} " + "  ".join(f"{x:6.3f}" for x in v), flush=True)
        else:
            print(f"{'phase':10s} " + "  ".join(f"{m:>11s}" for m in MODELS))
            for p in PHASES:
                print(f"{p:10s} " + "  ".join(f"{np.median(D[m][clip_phase==p]):11.3f}" for m in MODELS), flush=True)
    tbl("W2 spatial", W2S, True)
    tbl("VGG style", VGG, True)
    tbl("W2 values (A,B), cube-pooled", W2V, False)

    # ---- bar figures (median per phase per model) ----
    col = {m: c for m, c in zip(MODELS, plt.get_cmap("tab10").colors)}
    def bars(name, getval, fname):
        fig, ax = plt.subplots(figsize=(13, 5.5))
        x = np.arange(len(PHASES)); w = 0.8 / len(PLOT_MODELS)
        for j, m in enumerate(PLOT_MODELS):
            vals = [np.median(getval(m, p)) for p in PHASES]
            ax.bar(x + j * w, vals, w, color=col[m], label=m)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(PHASES, fontsize=11)
        ax.set_ylabel(name); ax.set_title(f"{name} per phase per model  (epoch {ckpt.get('epoch')}, test)")
        ax.legend(ncol=len(PLOT_MODELS), fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname), dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"[fld] wrote {fname}", flush=True)
    for hi, h in enumerate(HZ):
        bars(f"W2 spatial @ h{h}", lambda m, p, hi=hi: W2S[m][clip_phase == p][:, hi], f"field_w2spatial_h{h}.png")
        bars(f"VGG style @ h{h}", lambda m, p, hi=hi: VGG[m][clip_phase == p][:, hi], f"field_vgg_h{h}.png")
    bars("W2 values (A,B), cube", lambda m, p: W2V[m][clip_phase == p], "field_w2values.png")
    print("[fld] DONE", flush=True)


if __name__ == "__main__":
    main()
