# Gray-Scott Large Model — Design Corrections & Upgrades

This document records what changed from the initial toy run (`gray_scott_dev`) to the
scaled run (`gray_scott_large`), and why. Read alongside `DESIGN.md`.

---

## What the toy model revealed

The small model (run `gray_scott_dev`) showed a floor VRMSE of **0.542** at h=1,
meaning that even decode(encode(ground-truth)) loses 54% of spatial variance. This
set a hard ceiling: no predictor can beat 0.54 VRMSE with a D=16 encoder. The
FNO target is 0.1365, so the toy model was architecturally incapable of competing.

| Toy model VRMSE (epoch 9) | h=1 | h=8 |
|---|---|---|
| floor | 0.542 | 0.526 |
| jepa | 0.535 | 0.627 |
| persistence | 0.251 | 0.768 |

JEPA beats persistence at h≥3 (long-horizon advantage), but floor >> FNO target.

---

## Architecture changes in the large model

### Latent dimension: D=16 → D=64

**Why:** With D=16 at 128×128, the encoder compresses 2×128×128=32,768 input values
into 16×128×128=262,144 latent values (8x expansion), yet the 3-layer decoder
couldn't reconstruct faithfully — floor=0.54. Increasing to D=64 gives 4× more
latent channels, giving the encoder and decoder far more capacity to preserve fine
spatial structure. Expected floor: well below 0.15.

### Encoder hidden: h=32 → h=128

**Why:** Wider intermediate features give the encoder more capacity to extract
spatially coherent representations at each scale. Pairs with dstc=64.

### Predictor hidden: h=32 → h=64 (NOT the 256 originally attempted)

**Why — OOM lesson:** The first scaled attempt used hpre=256 with batch=64.
The ResUNet predictor operates at full 128×128 spatial resolution with all frames
folded into the batch dim (effective batch = 64×10 = 640). At h=256 this used
171 GB of the 184 GB available, crashing with OOM. The fix is a *lighter* predictor
(hpre=64), since the predictor's job is dynamics modelling in latent space, not
feature extraction — D=64 latent channels already carry the spatial signal, and
the predictor just needs to learn the temporal transition.

### n_frames: 10 → 16 (README correction)

**Why:** The README specifies `n_frames=16` for Gray-Scott. Using 10 frames
underused each clip and reduced the number of prediction targets per forward pass.
With n_frames=16 and context_length=2, we predict 10 future frames per clip
(steps=10) instead of 4 — 2.5× more supervision signal per clip.

### JEPA steps: 4 → 10

**Why:** With n_frames=16 and context_length=2, all 16 frames can be used:
2 context + 10 predicted = 12 ≤ 16. More steps = more prediction targets =
stronger self-supervised signal per forward pass.

### Batch size: 8 → 32

**Why:** The toy run with batch=8 used <5% of the 184 GB GPU memory. With the
larger model (dstc=64, hpre=64) and batch=32, the effective batch in the ResUNet
is 32×16=512 frames at 64×128×128, which should use ~20–40 GB — much better
utilization while staying safely within limits.

### epoch_size: 8000 → 28800 (full dataset)

**Why:** The toy run sampled only 28% of the 28,800 available clips per epoch.
The large run sees the full dataset each epoch — no information left on the table.
Steps per epoch: 28800/32 = 900 (vs 1000 in the toy run; similar wall time per epoch).

### Learning rate: 1e-3 → 3e-4

**Why:** Larger model → sharper loss landscape → lower stable LR. Standard
practice for scaled-up variants.

### Epochs: 20 → 50

**Why:** More parameters need more epochs to converge. VRMSE eval runs every
5 epochs so we get 10 data points across training.

---

## Updated hyperparameter table

| Hyperparameter | Toy (dev) | Large |
|---|---|---|
| Encoder hidden henc | 32 | 128 |
| Latent dim D | 16 | 64 |
| Predictor hidden hpre | 32 | 64 |
| n_frames | 10 | 16 |
| JEPA steps | 4 | 10 |
| Batch size | 8 | 32 |
| epoch_size | 8000 | 28800 |
| Optimizer | Adam, lr=1e-3 | Adam, lr=3e-4 |
| Epochs | 20 | 50 |
| VRMSE eval every | — (manual) | 5 epochs |

---

## Expected floor with D=64

With D=64 and henc=128, the latent has 4× more channels than D=16. The
encoder/decoder reconstruction MSE should drop well below 0.10, giving
floor VRMSE well below 0.30. Whether the predictor can then approach the
floor at short horizons (competing with FNO's 0.1365) depends on training.
The first data point is at epoch 4 (~1 hour into training).
