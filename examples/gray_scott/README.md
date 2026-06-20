# Gray-Scott — temporal JEPA on a PDE reaction-diffusion field (The Well)

**Question.** Can a JEPA learn the *dynamics* of a PDE by predicting the *latent*
of the future (not the pixels), and how does latent-space prediction compare —
in The Well's field-space VRMSE — to neural-operator surrogates (FNO / U-Net)?

## Data
`polymathic-ai/gray_scott_reaction_diffusion` (The Well,
[Ohana et al. 2024, arXiv:2412.00568](https://arxiv.org/abs/2412.00568)). Two
chemical fields **A** and **B** diffuse and react on a 128x128 grid; each
trajectory is **1001 timesteps**, stored as HDF5 under `t0_fields/{A,B}`. Feed/
kill parameters (F, k) give **6 visually distinct regimes** (spots, worms,
maze-like, ...) — one regime per HDF5 file. A training item is a clip of
`n_frames=16` with `time_stride=4`, the two fields stacked as channels into a
`[2, T, 128, 128]` tensor, z-scored per channel. The train/valid/test splits are
the dataset's own trajectory folders, so any probe is trajectory-disjoint.

## Layout
```
eb_jepa/datasets/gray_scott/   dataset.py (provided HDF5 loader) + data_config.yaml
examples/gray_scott/
  main.py             temporal-JEPA pretraining (encoder + JEPA assembly)
  eval.py             field-space per-horizon VRMSE rollout (JEPA vs persistence)
  eval_regimes.py     same protocol, broken out per (F,k) regime + plots
  eval_compare.py     JEPA vs The Well U-Net baselines, pooled VRMSE, autoregressive
  eval_baselines.py   pretrained The Well baselines, paper VRMSE, teacher-forced
  train_decoder.py    continue-train the latent->field decoder from a checkpoint
  visualize.py        truth-vs-rollout filmstrip PNG + GIF (optional baseline panels)
  unroll_ood.py       roll JEPA from an UNSEEN (F,k) — out-of-distribution probe
  viz_regimes_gif.py  presentation GIF: the 6 regimes beside the F-k phase diagram
  _well_baselines.py  shared helper: stub heavy unused The Well models on import
  cfgs/               train.yaml, train_large.yaml, eval.yaml
  DESIGN.md, DESIGN_large.md   design notes for the base / large runs
```

## The model — temporal / predictive JEPA (not two-view)
```
context  z[:, :context_length=2]  --predictor(ResUNet)-->  z_hat (future latent)
target   z_target = target_encoder(future frames)        (EMA, no grad)
loss     = || z_hat - z_target ||  (SquareLossSeq) + VCLoss(std, cov)  (anti-collapse)
```
There is **no pixel loss in pretraining** — the model predicts a *representation*
of the future. A latent->field decoder is added only at eval to score VRMSE.

## The four key pieces (implemented)
The example started as a template with four `# TODO`s; they are now implemented.
What each one is and where it lives:
1. `main.py:build_encoder` — a 2D frame encoder `[B, 2, H, W] -> [B, D, h, w]`
   (point at `eb_jepa.architectures.ResNet5` / `ImpalaEncoder`; stride-1 keeps the
   latent full-resolution so a decoder can map it back to a field).
2. `main.py:build_jepa` — the temporal-JEPA assembly: `eb_jepa.jepa.JEPA` with the
   shared encoder + EMA target, a `StateOnlyPredictor(ResUNet(2D, hpre, D))` that
   rolls latents forward, `VCLoss` (anti-collapse) and `SquareLossSeq` (prediction).
3. `eval.py:build_decoder` — a frozen-JEPA latent->field decoder (train it to
   minimise `MSE(decode(encode(field)), field)`); its error is JEPA's irreducible
   field floor.
4. `eval.py:vrmse_per_horizon` — multi-step **VRMSE** (variance-scaled RMSE,
   aggregated num/den) for JEPA vs **persistence** (and optionally FNO / U-Net
   surrogates, trained iso-protocol) over horizons `1..H`.

Everything else (HDF5 loading, training loop, autoregressive latent rollout
extraction) is provided. Reuse the eb_jepa core (`ResNet5`, `ResUNet`,
`StateOnlyPredictor`, `Projector`, `VCLoss`, `SquareLossSeq`, `JEPA`) — do not
duplicate it.

## Run
Train, then score field-space VRMSE of the latent rollout:
```bash
python -m examples.gray_scott.main --fname examples/gray_scott/cfgs/train.yaml
python -m examples.gray_scott.eval --ckpt <.../latest.pth.tar> --H 10
```

## Scripts
All are run from the repo root. The eval/visualize scripts that touch The Well
baselines need the neural-operator extras, easiest via `uv run --with`:
```
uv run --with "neuraloperator==0.3.0" --with torch-harmonics --with timm \
       --with einops python examples/gray_scott/<script>.py ...
```

| Script | What it does | Run |
|---|---|---|
| `main.py` | Temporal-JEPA pretraining (W&B + optional per-epoch inline VRMSE). | `python -m examples.gray_scott.main --fname examples/gray_scott/cfgs/train.yaml` |
| `eval.py` | Headline VRMSE: roll JEPA in latent space, decode, score per horizon `1..H` vs persistence; at large enough `H` also prints The Well Table-3 windows `[6:12]`, `[13:30]`. | `python -m examples.gray_scott.eval --ckpt <ckpt> --H 30` |
| `eval_regimes.py` | Same rollout/metric **per (F,k) regime** (the default loader mixes all 6); prints a sorted table, saves `viz/regime_vrmse{,_bars,_curves}.*`. | `python -m examples.gray_scott.eval_regimes --ckpt <ckpt> --H 30 --n-per-regime 80` |
| `eval_compare.py` | JEPA **vs The Well U-Net baselines**, both autoregressive at stride=4, pooled VRMSE. | `uv run ... python examples/gray_scott/eval_compare.py --ckpt <ckpt>` |
| `eval_baselines.py` | Pretrained The Well baselines (FNO/TFNO/UNet/CNextUNet) alone, the paper's exact metric/protocol. | `uv run ... python examples/gray_scott/eval_baselines.py` |
| `train_decoder.py` | Continue-train the latent->field decoder stored in a checkpoint for more epochs (sharpens the VRMSE floor). | `python -m examples.gray_scott.train_decoder --ckpt <ckpt> --epochs 30` |
| `visualize.py` | Truth-vs-rollout filmstrip PNG + GIF per channel; `--baselines` adds U-Net panels. | `python -m examples.gray_scott.visualize --ckpt <ckpt> --H 60 [--baselines]` |
| `unroll_ood.py` | Out-of-distribution probe: seed from a real frame, evolve under an **unseen (F,k)**, then watch the (state-only) JEPA rollout. | `python examples/gray_scott/unroll_ood.py --ckpt <ckpt> --F 0.020 --k 0.0515 --source-regime spirals --H 60` |
| `viz_regimes_gif.py` | Presentation GIF: the 6 regimes animating beside the F-k phase diagram (h5py only, no GPU). | `python examples/gray_scott/viz_regimes_gif.py` |

### A note on the VRMSE variants
The eval scripts deliberately use **different** VRMSE definitions — compare like
with like, don't mix numbers across scripts:
- **Per-horizon, aggregated num/den** (`eval.py`, `eval_regimes.py`): per-channel
  variance-scaled RMSE summed over the batch then ratioed. The headline metric.
- **Pooled** `sqrt(Σ_i MSE_i / Σ_i var_i)` (`eval_compare.py`): robust on
  flat/dissolving samples; predicting the spatial mean gives exactly 1.0.
- **Mean-of-ratios, per sample, teacher-forced** (`eval_baselines.py`): The Well
  paper's exact Table-3 protocol (C_hist=4 ground-truth context, stride=1), so the
  baseline numbers are directly comparable to the paper.

The eval/compare protocols also differ: `eval*.py` roll **autoregressively** from
the model's own outputs (the hard, long-horizon stability test), while
`eval_baselines.py` is **teacher-forced** to match the paper.
