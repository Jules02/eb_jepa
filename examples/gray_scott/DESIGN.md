# Gray-Scott Track 4 — Method Design

## Research question

Can a JEPA learn the *dynamics* of a PDE by predicting future *latents* (not pixels)?
And how does latent-space prediction compare — in field-space VRMSE — to a persistence
baseline and neural-operator surrogates (FNO / U-Net)?

---

## Data

`polymathic-ai/gray_scott_reaction_diffusion` (The Well, Ohana et al. 2024).

- Two chemical fields **A** and **B** diffuse and react on a 128×128 grid
- 6 visually distinct regimes (spots, worms, maze, spirals, gliders, bubbles)
- 1001 timesteps per trajectory; each training item is a clip of `n_frames=10`
  frames sampled at stride 4 → covers 40 real timesteps per clip
- Fields are z-scored per channel using dataset statistics
- Train / valid / test splits are trajectory-disjoint (no leakage)

---

## Model: temporal / predictive JEPA

```
context frames  x[:, :, :2]
       │
   [encoder]  ResNet5(in_d=2, h=32, out_d=16)  [stride-1, full 128×128]
       │
   z[:, :, :2]  ──[predictor]──►  ẑ[:, :, 2:]   (predicted future latents)
                  StateOnlyPredictor(
                    ResUNet(in_d=2D, h=32, out_d=D)
                    context_length=2
                  )
       │
   z[:, :, 2:]  ──[target]──►  (ground-truth future latents from same encoder)
       │
   Loss = SquareLossSeq(ẑ, z_target)   [prediction]
        + VCLoss(z, proj=Projector(D→4D→4D))  [anti-collapse]
```

This is a **predictive JEPA**, not a two-view contrastive JEPA:
- The predictor rolls latents forward in time (no pixel reconstruction)
- The "target" is the online encoder applied to ground-truth future frames
  (no separate EMA network — VCLoss handles collapse prevention instead)

---

## Design choices

### Encoder: `ResNet5` with stride-1

**Choice:** `ResNet5(in_d=2, h_d=32, out_d=16)` with all strides=1 and no average-pool.

**Why:** The encoder must preserve spatial structure (128×128) so a lightweight
decoder can map latents back to fields for VRMSE evaluation. `ImpalaEncoder`
was rejected because it flattens spatial dimensions entirely (output `[B,D,1,1]`),
making field decoding impossible without a large upsampling network.
`TemporalBatchMixin` on ResNet5 handles 5D `[B,C,T,H,W]` inputs automatically
by folding T into the batch dim.

### Predictor: `StateOnlyPredictor` + `ResUNet`

**Choice:** `StateOnlyPredictor(ResUNet(in_d=2*D, h_d=32, out_d=D), context_length=2)`.

**Why:** The predictor must map latent spatial fields forward in time. `ResUNet`
is the natural choice: its encoder-decoder with skip connections preserves
spatial detail while the bottleneck captures global dynamics. It takes two
consecutive latent frames concatenated on the channel axis (`in_d=2*D`) and
predicts the next one. `context_length=2` matches this: the JEPA unroll
re-feeds the 2 most recent ground-truth latents as context at each step.

An RNN predictor was not chosen because it would collapse spatial structure
to a vector and lose the 128×128 layout that the decoder needs.

### Anti-collapse: `VCLoss` with projector

**Choice:** `VCLoss(std_coeff=10.0, cov_coeff=100.0, proj=Projector("16-64-64"))`.

**Why:** Without a separate EMA target encoder, the predictor can collapse to
a trivial constant representation. VCLoss prevents this by:
- **Variance term** (std_coeff=10): penalises features whose std falls below 1,
  forcing each latent dimension to be informative.
- **Covariance term** (cov_coeff=100): penalises off-diagonal covariance,
  decorrelating features and preventing redundant dimensions.
The projector (D→4D→4D) follows the VICReg convention of projecting before
computing variance/covariance so the penalty does not distort the raw latents.

### Prediction loss: `SquareLossSeq`

**Choice:** `SquareLossSeq()` — plain MSE between predicted and ground-truth latents.

**Why:** The latents are already in a meaningful metric space (z-scored inputs,
stride-1 encoder). A simple L2 loss is appropriate; no additional projector
is needed on the prediction side since VCLoss already regularises the space.

### Decoder (eval only): 3-layer conv stack

**Choice:** `Conv2d(D,64,3,pad=1) → GELU → Conv2d(64,64,3,pad=1) → GELU → Conv2d(64,2,1)`.

**Why:** Because the encoder is stride-1 the latent is already at 128×128 —
no upsampling needed. A 3-layer conv stack is sufficient to reconstruct the
2-channel field from D=16 latent channels. It is trained separately with the
JEPA frozen (MSE on fields), and its reconstruction error gives the
**irreducible floor**: the minimum VRMSE achievable by any predictor using
this encoder, since `VRMSE_floor = VRMSE(decode(encode(true)), true)`.

---

## VRMSE metric

VRMSE (The Well) = `sqrt( Σ_space (pred − true)² / Σ_space (true − μ_true)² )`

Numerators and denominators are **aggregated across the full validation set**
before dividing. This is critical: per-sample ratios blow up on near-uniform
frames (channel B has near-zero spatial variance in some regimes), whereas
the aggregated ratio stays stable.

Three curves are reported per horizon h=1..H:
| Curve | Description |
|---|---|
| `jepa` | decode(rollout_latents(context)) |
| `persistence` | repeat last context frame |
| `floor` | decode(encode(ground-truth frame)) |

A well-trained JEPA should satisfy `floor < jepa < persistence` at all horizons,
with JEPA approaching the floor at short horizons.

---

## Training

| Hyperparameter | Value |
|---|---|
| Encoder hidden | 32 |
| Latent dim D | 16 |
| Predictor hidden | 32 |
| JEPA steps | 4 |
| VCLoss std coeff | 10.0 |
| VCLoss cov coeff | 100.0 |
| Optimizer | Adam, lr=1e-3 |
| Epochs | 20 |
| Clips/epoch | 8000 |
| Batch size | 8 |
| AMP | bfloat16 |

Checkpoints: `latest.pth.tar` every epoch, `epoch_N.pth.tar` every 5 epochs.
W&B project: `eb_jepa`, run: `gray_scott_dev`.
