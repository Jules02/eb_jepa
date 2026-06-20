"""Gray-Scott — temporal-JEPA pretraining entrypoint (PDE reaction-diffusion).

Research question: can a JEPA learn the *dynamics* of a PDE by predicting the
*latent* of the future (not the pixels)? Each simulation is a 2D physical video
``[2, T, 128, 128]`` (chemical fields A, B). This is a PREDICTIVE / temporal
JEPA (video-style), NOT a two-view objective:

  context  z[:, :context_length]  --predictor-->  z_hat  (future latent)
  target   z_target = target_encoder(future frames)      (EMA, no grad)
  loss     = || z_hat - z_target ||  (latent prediction) + VC(z) (anti-collapse)

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement are
marked ``# TODO`` below — that is the whole point of the track:
  1. the 2D encoder over a frame  ``[B, 2, H, W] -> [B, D, h, w]``
  2. the temporal-JEPA assembly (encoder + EMA target + predictor + VCLoss)

Run:  python -m examples.gray_scott.main --fname examples/gray_scott/cfgs/train.yaml
"""
import os
import sys
import time

import torch
from omegaconf import OmegaConf

from eb_jepa.architectures import Projector, ResNet5, ResUNet, StateOnlyPredictor
from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VCLoss
from eb_jepa.training_utils import setup_wandb


# --------------------------------------------------------------------------- #
# 1) ENCODER  — # TODO
# --------------------------------------------------------------------------- #
def build_encoder(cfg):
    # stride-1 everywhere (default), no avg-pool -> latent stays 128x128
    # TemporalBatchMixin on ResNet5 folds T into batch for 5D inputs
    norm = cfg.get("norm", "batch") if hasattr(cfg, "get") else getattr(cfg, "norm", "batch")
    return ResNet5(in_d=cfg.dobs, h_d=cfg.henc, out_d=cfg.dstc, norm=norm)


# --------------------------------------------------------------------------- #
# 2) TEMPORAL-JEPA ASSEMBLY  — # TODO
# --------------------------------------------------------------------------- #
def build_jepa(encoder, cfg):
    D = cfg.dstc
    norm = cfg.get("norm", "batch") if hasattr(cfg, "get") else getattr(cfg, "norm", "batch")
    predictor = StateOnlyPredictor(
        ResUNet(in_d=2 * D, h_d=cfg.hpre, out_d=D, norm=norm),
        context_length=2,
    )
    regularizer = VCLoss(
        cfg.std_coeff, cfg.cov_coeff,
        proj=Projector(f"{D}-{4*D}-{4*D}"),
    )
    predcost = SquareLossSeq()
    # actions=None throughout, so the action encoder (arg 2) is never called
    return JEPA(encoder, None, predictor, regularizer, predcost)


# --------------------------------------------------------------------------- #
# INLINE EVAL HELPER
# --------------------------------------------------------------------------- #
def _run_inline_eval(jepa, encoder, cfg, device, wandb_run, epoch, gstep,
                     decoder, decoder_opt, eval_loader, H=30):
    """Run VRMSE eval with a persistent warm-started decoder (fast, every epoch).

    Reports per-horizon VRMSE and The Well Table 3 windows [6:12] and [13:30]."""
    from examples.gray_scott.eval import vrmse_per_horizon, window_vrmse, WELL_WINDOWS
    jepa.eval()
    # Fine-tune decoder for 1 pass (warm start — fast since encoder barely changed)
    decoder.train()
    for batch in eval_loader:
        x = batch["video"].to(device)
        with torch.no_grad():
            z = encoder(x)
        recon = decoder(z)
        loss = torch.nn.functional.mse_loss(recon, x)
        decoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        decoder_opt.step()
    decoder.eval()
    scores = vrmse_per_horizon(jepa, encoder, decoder, eval_loader, device, H)
    from examples.gray_scott.eval import _HEADLINE_KEYS
    for name in _HEADLINE_KEYS:
        arr = scores[name]
        print(f"[eval-e{epoch}] {name:14s} h1={arr[0]:.3f} h{H}={arr[-1]:.3f}", flush=True)
    for name in ("jepa", "floor"):
        for ch in ("u", "v"):
            arr = scores[f"{name}_{ch}"]
            print(f"[eval-e{epoch}] {name}_{ch:11s} h1={arr[0]:.3f} h{H}={arr[-1]:.3f}", flush=True)
    # The Well Table 3 windowed summary
    for wname, (start, end) in WELL_WINDOWS.items():
        if end <= H:
            w = window_vrmse(scores, wname)
            headline = "  ".join(f"{k}={w[k]:.3f}" for k in _HEADLINE_KEYS)
            print(f"[eval-e{epoch}] window {wname}: {headline}", flush=True)
            print(f"[eval-e{epoch}]   jepa_u={w['jepa_u']:.3f}  jepa_v={w['jepa_v']:.3f}", flush=True)
    if wandb_run:
        import wandb
        log_dict = {}
        for h in range(H):
            for name in _HEADLINE_KEYS:
                log_dict[f"eval/vrmse_{name}_h{h+1}"] = scores[name][h]
            for name in ("jepa", "floor"):
                for ch in ("u", "v"):
                    log_dict[f"eval/vrmse_{name}_{ch}_h{h+1}"] = scores[f"{name}_{ch}"][h]
        for wname, (start, end) in WELL_WINDOWS.items():
            if end <= H:
                w = window_vrmse(scores, wname)
                for name in _HEADLINE_KEYS:
                    log_dict[f"eval/vrmse_{name}_w{wname.replace(':','_')}"] = w[name]
                for name in ("jepa", "floor"):
                    for ch in ("u", "v"):
                        log_dict[f"eval/vrmse_{name}_{ch}_w{wname.replace(':','_')}"] = w[f"{name}_{ch}"]
        wandb.log(log_dict, step=gstep)
    jepa.train()


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/gray_scott/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = GrayScottConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    train_loader = make_loader(dcfg)
    val_loader = make_loader(GrayScottConfig(**{**dcfg.__dict__, "split": "valid",
                                                "epoch_size": dcfg.batch_size * 10}), shuffle=False)
    print(f"[gs] {len(train_loader.dataset.files)} train hdf5 | "
          f"clip=[{dcfg.channels},{dcfg.n_frames},{dcfg.img_size},{dcfg.img_size}] "
          f"stride={dcfg.time_stride} | {len(train_loader)} steps/epoch", flush=True)

    encoder = build_encoder(cfg.model).to(device)
    jepa = build_jepa(encoder, cfg.model).to(device)
    print(f"[gs] params: {sum(p.numel() for p in jepa.parameters()) / 1e6:.2f}M", flush=True)

    opt = torch.optim.Adam(jepa.parameters(), lr=cfg.optim.lr)
    use_amp = bool(cfg.training.use_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.training.get("dtype", "bfloat16") == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_run = setup_wandb(
        project=cfg.logging.get("wandb_project", "eb_jepa"),
        config={"example": "gray_scott", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=ckpt_dir,
        run_name=cfg.logging.get("wandb_run_name", "gray_scott"),
        tags=["gray_scott", f"seed_{cfg.meta.seed}"],
        enabled=bool(cfg.logging.get("log_wandb", False)),
    )

    def _save(name):
        torch.save({"epoch": epoch,
                    "encoder": encoder.state_dict(),
                    "jepa": jepa.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, name))

    # persistent decoder for per-epoch VRMSE eval (warm-started each epoch)
    eval_every = cfg.logging.get("eval_every", 0)
    if eval_every > 0:
        from examples.gray_scott.eval import _FrameDecoder  # noqa: F401
        H_eval = 30    # The Well Table 3: 30-step rollout, windows [6:12] and [13:30]
        C_eval = 2
        _decoder = _FrameDecoder(D=cfg.model.dstc).to(device)
        _decoder_opt = torch.optim.Adam(_decoder.parameters(), lr=1e-3)
        _eval_loader = make_loader(GrayScottConfig(
            **{**dcfg.__dict__, "split": "valid",
               "n_frames": C_eval + H_eval, "epoch_size": 400}), shuffle=False)
        print(f"[gs] inline eval enabled every {eval_every} epoch(s), H={H_eval}", flush=True)
    else:
        _decoder = _decoder_opt = _eval_loader = None

    gstep = 0
    for epoch in range(cfg.optim.epochs):
        jepa.train()
        t0 = time.time()
        for batch in train_loader:
            x = batch["video"].to(device, non_blocking=True)        # [B,2,T,H,W]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                _, (jepa_loss, regl, _, _, pl) = jepa.unroll(
                    x, actions=None, nsteps=cfg.model.steps,
                    unroll_mode="parallel", compute_loss=True, return_all_steps=False)
            if scaler.is_enabled():
                scaler.scale(jepa_loss).backward(); scaler.step(opt); scaler.update()
            else:
                jepa_loss.backward(); opt.step()
            gstep += 1
            if gstep % cfg.logging.log_every == 0:
                print(f"e{epoch} s{gstep} loss={jepa_loss.item():.4f} "
                      f"vc={regl.item():.4f} pred={pl.item():.4f}", flush=True)
                if wandb_run:
                    import wandb
                    wandb.log({"train/loss": jepa_loss.item(),
                               "train/vc_loss": regl.item(),
                               "train/pred_loss": pl.item()}, step=gstep)

        # val
        jepa.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["video"].to(device)
                with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                    _, (jl, _, _, _, _) = jepa.unroll(x, actions=None, nsteps=cfg.model.steps,
                                                      unroll_mode="parallel", compute_loss=True)
                vl += jl.item(); nb += 1
        val_loss = vl / max(nb, 1)
        elapsed = time.time() - t0
        print(f"[epoch {epoch}] {elapsed:.0f}s | val_loss={val_loss:.4f}", flush=True)
        if wandb_run:
            import wandb
            wandb.log({"val/loss": val_loss, "epoch": epoch,
                       "train/loss_last": jepa_loss.item()}, step=gstep)

        # always save latest; save numbered checkpoint every save_every epochs
        _save("latest.pth.tar")
        save_every = cfg.logging.get("save_every", 5)
        if (epoch + 1) % save_every == 0:
            _save(f"epoch_{epoch}.pth.tar")

        # optional inline VRMSE eval (logged to W&B if enabled)
        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            _run_inline_eval(jepa, encoder, cfg, device, wandb_run, epoch, gstep,
                             _decoder, _decoder_opt, _eval_loader, H=H_eval)

    if wandb_run:
        import wandb
        wandb.finish()
    print(f"[gs] done -> {ckpt_dir}/latest.pth.tar", flush=True)


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/gray_scott/cfgs/train.yaml"
    run(fname=fname)
