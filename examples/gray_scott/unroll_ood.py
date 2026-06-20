"""Gray-Scott — unroll JEPA from an UNSEEN (F,k) ("real physics" eyeball test).

The model is STATE-ONLY: it never sees (F,k), it only continues whatever local
dynamics live in its C context frames. So "test on a new (F,k)" means: hand it
context frames that a *new* (F,k) genuinely produced, then watch the latent
rollout. The 6 training regimes are the only data we have, so the new-(F,k)
context frames have to be *simulated*.

Design (so the rollout itself stays solver-free, which is the whole point):

  1. SEED from a mature real frame of a source regime (default: spirals). Because
     A=u (mean~0.73) and B=v (mean~0.10) are already physical [0,1] fields, a
     denormalised real frame is a ready-made Gray-Scott state.
  2. CALIBRATE solver-steps-per-JEPA-frame ``S`` by matching the SOURCE regime's
     OWN real data across one JEPA stride (time_stride saved frames). This pins
     the solver's timescale to The Well's empirically — we never need The Well's
     internal dt / save cadence.
  3. Evolve the seed under the NEW (F,k) for a short warmup, then by ``S`` steps
     to get the 2 context frames JEPA needs. Both frames are genuine new-(F,k)
     states spaced by exactly one JEPA frame.
  4. Unroll the frozen JEPA in latent space, decode to fields. Since the solver
     is already running, also continue it H frames to get a FREE "true new-regime
     physics" reference panel (solver-generated, clearly labelled) — turning the
     eyeball test into a quantitative one too.

The standard Pearson/Karl-Sims model reproduces The Well's named regimes at their
filename (F,k): Du=0.16, Dv=0.08, dt=1, periodic BC, 9-point Laplacian.

Run (on a GPU node, via uv):
  python -m examples.gray_scott.unroll_ood --ckpt <.../latest.pth.tar> \
      --F 0.020 --k 0.0515 --source-regime spirals --H 60
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as TF
from omegaconf import OmegaConf

try:
    import h5py
except ImportError:
    h5py = None

from eb_jepa.datasets.gray_scott.dataset import MEAN, STD, ROOT, parse_regime
from examples.gray_scott.eval import C, load_jepa, build_decoder, rollout_latents
from examples.gray_scott.visualize import _rgb, make_compare_gif

# Standard Gray-Scott constants (reproduce The Well's named regimes at their F,k).
DU, DV, DT = 0.16, 0.08, 1.0
# 9-point Laplacian (Karl Sims / Pearson) — the kernel these (F,k) were tuned for.
_LAP = torch.tensor([[0.05, 0.2, 0.05],
                     [0.20, -1.0, 0.20],
                     [0.05, 0.2, 0.05]], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Gray-Scott solver (torch, periodic BC, runs on the JEPA device)
# --------------------------------------------------------------------------- #
def _lap(x, kernel):
    """Periodic 9-point Laplacian. x: [1,1,H,W]."""
    xp = TF.pad(x, (1, 1, 1, 1), mode="circular")
    return TF.conv2d(xp, kernel)


def gs_step(u, v, Fd, kd, kernel):
    uvv = u * v * v
    u = u + (DU * _lap(u, kernel) - uvv + Fd * (1.0 - u)) * DT
    v = v + (DV * _lap(v, kernel) + uvv - (Fd + kd) * v) * DT
    # positivity / blow-up guard (a no-op for in-range states)
    return u.clamp_(0.0, 2.0), v.clamp_(0.0, 2.0)


def simulate(u, v, Fd, kd, steps, kernel):
    for _ in range(int(steps)):
        u, v = gs_step(u, v, Fd, kd, kernel)
    return u, v


# --------------------------------------------------------------------------- #
# Real-data access (physical units, for seeding + calibration)
# --------------------------------------------------------------------------- #
def _regime_file(regime, split, data_root):
    files = glob.glob(os.path.join(data_root, "data", split, f"*_{regime}_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no hdf5 for regime '{regime}' in {split}")
    return sorted(files)[0]


def load_phys_frames(regime, split, idxs, traj, data_root):
    """Return (A, B) real fields [n,128,128] in physical units, + parsed (F,k)."""
    if h5py is None:
        raise ImportError("h5py required")
    path = _regime_file(regime, split, data_root)
    with h5py.File(path, "r") as f:
        A = f["t0_fields/A"][traj, list(idxs)].astype(np.float32)
        B = f["t0_fields/B"][traj, list(idxs)].astype(np.float32)
    _, Fsrc, ksrc = parse_regime(path)
    return A, B, (Fsrc, ksrc)


def _to_uv(A, B, device):
    """[H,W] physical fields -> ([1,1,H,W], [1,1,H,W]) torch on device."""
    u = torch.from_numpy(np.asarray(A))[None, None].to(device)
    v = torch.from_numpy(np.asarray(B))[None, None].to(device)
    return u, v


# --------------------------------------------------------------------------- #
# Calibration: how many dt=1 solver steps == one JEPA frame (time_stride saved
# frames), measured on the SOURCE regime's own real data.
# --------------------------------------------------------------------------- #
def calibrate_steps(regime, split, time_stride, traj, data_root, kernel, device,
                    anchors=(300, 500, 700), s_max=400):
    """Scan S; pick the value that best maps frame t -> frame t+time_stride."""
    idxs = []
    for t0 in anchors:
        idxs += [t0, t0 + time_stride]
    A, B, (Fsrc, ksrc) = load_phys_frames(regime, split, idxs, traj, data_root)

    def total_err(S):
        err = 0.0
        for i in range(len(anchors)):
            u0, v0 = _to_uv(A[2 * i], B[2 * i], device)
            ut, vt = simulate(u0, v0, Fsrc, ksrc, S, kernel)
            tu, tv = _to_uv(A[2 * i + 1], B[2 * i + 1], device)
            e = TF.mse_loss(ut, tu) + TF.mse_loss(vt, tv)
            err += float(e)
        return err

    # coarse scan then refine around the minimum
    coarse = list(range(2, s_max + 1, 8))
    best = min(coarse, key=total_err)
    fine = [s for s in range(max(1, best - 8), best + 9) if s >= 1]
    best = min(fine, key=total_err)
    print(f"[ood] calibrated S = {best} solver steps / JEPA frame "
          f"(source={regime} F={Fsrc} k={ksrc}, err={total_err(best):.4e})", flush=True)
    return best, (Fsrc, ksrc)


# --------------------------------------------------------------------------- #
# VRMSE (paper formula) of JEPA decoded fields vs the solver "truth"
# --------------------------------------------------------------------------- #
def vrmse(pred, true):
    """pred,true: [2,H,W] physical. Returns scalar (mean over channels)."""
    mse = ((pred - true) ** 2).mean(axis=(-2, -1))
    var = true.var(axis=(-2, -1))
    return float(np.sqrt(mse / (var + 1e-7)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to *.pth.tar (jepa + optional decoder)")
    ap.add_argument("--F", type=float, required=True, help="feed rate of the UNSEEN regime")
    ap.add_argument("--k", type=float, required=True, help="kill rate of the UNSEEN regime")
    ap.add_argument("--source-regime", default="spirals",
                    help="regime to seed from + calibrate against (default: spirals)")
    ap.add_argument("--H", type=int, default=60, help="rollout horizon (JEPA frames)")
    ap.add_argument("--time-stride", type=int, default=4, help="saved frames per JEPA frame")
    ap.add_argument("--traj", type=int, default=0, help="trajectory index for seed/calibration")
    ap.add_argument("--seed-frame", type=int, default=700,
                    help="saved-frame index of the (mature) seed state")
    ap.add_argument("--steps-per-frame", type=int, default=None,
                    help="override calibrated S (skip calibration)")
    ap.add_argument("--warmup-frames", type=float, default=2.0,
                    help="JEPA-frames of new-(F,k) warmup before taking context (relax onto new manifold)")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--outdir", default="examples/gray_scott/viz")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--tag", default=None, help="suffix for output filename")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    data_root = ROOT
    kernel = _LAP.to(device).view(1, 1, 3, 3)

    # --- model ---
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    data_root = cfg.data.get("data_root", ROOT)
    jepa, encoder = load_jepa(ckpt, device)
    dstc = int(cfg.model.dstc)
    decoder = build_decoder(dstc, device, ckpt_path=args.ckpt)
    print(f"[ood] ckpt epoch {ckpt.get('epoch')}, NEW (F,k)=({args.F},{args.k}), "
          f"source={args.source_regime}, H={args.H}, device={device}", flush=True)

    # --- calibrate timescale on the source regime ---
    if args.steps_per_frame is not None:
        S = args.steps_per_frame
        _, _, (Fsrc, ksrc) = load_phys_frames(
            args.source_regime, args.split, [args.seed_frame], args.traj, data_root)
        print(f"[ood] using override S={S} (source F={Fsrc} k={ksrc})", flush=True)
    else:
        S, (Fsrc, ksrc) = calibrate_steps(
            args.source_regime, args.split, args.time_stride, args.traj,
            data_root, kernel, device)

    dF = args.F - Fsrc
    dk = args.k - ksrc
    print(f"[ood] offset from {args.source_regime}: dF={dF:+.4f} dk={dk:+.4f}", flush=True)

    # --- seed a mature state, evolve it under the NEW (F,k) ---
    A0, B0, _ = load_phys_frames(
        args.source_regime, args.split, [args.seed_frame], args.traj, data_root)
    u, v = _to_uv(A0[0], B0[0], device)

    warmup = int(round(args.warmup_frames * S))
    u, v = simulate(u, v, args.F, args.k, warmup, kernel)  # relax onto new-(F,k) manifold

    # 2 context frames (C=2) at the new (F,k), spaced by one JEPA frame (S steps)
    ctx = [(u.clone(), v.clone())]
    for _ in range(C - 1):
        u, v = simulate(u, v, args.F, args.k, S, kernel)
        ctx.append((u.clone(), v.clone()))

    # solver "truth": continue H more frames from the last context frame
    truth = []
    for _ in range(args.H):
        u, v = simulate(u, v, args.F, args.k, S, kernel)
        truth.append((u.clone(), v.clone()))

    def stack_phys(frames):
        """list of (u,v)[1,1,H,W] -> np [2, T, H, W] physical."""
        A = np.concatenate([f[0].cpu().numpy().reshape(1, 1, *f[0].shape[-2:]) for f in frames], axis=1)
        B = np.concatenate([f[1].cpu().numpy().reshape(1, 1, *f[1].shape[-2:]) for f in frames], axis=1)
        return np.concatenate([A, B], axis=0)  # [2, T, H, W]

    ctx_phys = stack_phys(ctx)          # [2, C, H, W]
    truth_phys = stack_phys(truth)      # [2, H, H, W]

    # --- JEPA latent rollout from the simulated context ---
    mean = MEAN[:, None, None, None]
    std = STD[:, None, None, None]
    x_ctx = (ctx_phys - mean) / std                      # z-score, [2, C, H, W]
    x = torch.from_numpy(x_ctx[None].astype(np.float32)).to(device)  # [1,2,C,H,W]
    with torch.no_grad():
        pred_z = rollout_latents(jepa, x, args.H, device)     # [1,D,C+H,h,w]
        jepa_future = decoder(pred_z[:, :, C:])               # [1,2,H,H,W]
    jepa_phys = jepa_future[0].cpu().numpy() * std + mean     # [2, H, H, W]

    # --- VRMSE: JEPA decoded vs solver truth (per horizon) ---
    vr = np.array([vrmse(jepa_phys[:, h], truth_phys[:, h]) for h in range(args.H)])
    print(f"[ood] VRMSE(JEPA vs solver-truth)  h1={vr[0]:.3f}  "
          f"h{args.H}={vr[-1]:.3f}  mean={vr.mean():.3f}", flush=True)

    # --- side-by-side GIF: solver truth | JEPA, context spliced in front ---
    truth_full = np.concatenate([ctx_phys, truth_phys], axis=1)   # [2, C+H, H, W]
    jepa_full = np.concatenate([ctx_phys, jepa_phys], axis=1)     # [2, C+H, H, W]
    scale = [(float(truth_full[0].min()), float(truth_full[0].max())),
             (float(truth_full[1].min()), float(truth_full[1].max()))]
    panels = [
        (f"Truth (solver) F={args.F} k={args.k}", _rgb(truth_full, scale), {}),
        ("JEPA unroll", _rgb(jepa_full, scale), {}),
    ]
    tag = args.tag or f"{args.source_regime}_F{args.F}_k{args.k}"
    gif = os.path.join(args.outdir, f"ood_{tag}.gif")
    title = (f"OOD (F={args.F}, k={args.k})  src={args.source_regime}  "
             f"S={S}/frame  VRMSE_mean={vr.mean():.3f}")
    make_compare_gif(panels, gif, title, fps=args.fps)
    print(f"[ood] wrote {gif}", flush=True)


if __name__ == "__main__":
    main()
