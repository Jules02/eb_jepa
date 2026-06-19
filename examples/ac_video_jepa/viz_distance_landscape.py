"""
Visualize the latent distance landscape d(s_0, s) = || E(s_0) - E(s) ||_2 over the
Two-Rooms environment, for a trained AC-Video-JEPA encoder.

We fix the wall/door layout, pick an initial ball position s_0, then sweep the ball
over every pixel position s, encode each rendered frame with the trained encoder, and
plot the L2 distance between each embedding and the embedding of s_0 as a heatmap.

Usage (run on a node that can import torch; the forward pass is tiny so CPU is fine):

    python -m examples.ac_video_jepa.viz_distance_landscape \
        --model_folder /lustre/work/.../impala_cov8_std16_simt12_idm1_seed1 \
        --s0 24,32

Args:
    model_folder : run directory containing `config.yaml` + `latest.pth.tar`.
    checkpoint   : checkpoint filename inside model_folder (default latest.pth.tar).
    s0           : "x,y" pixel coordinates of the initial state (default = grid center).
    fix_wall     : True to use the fixed wall/door layout (default), False to sample one.
    stride       : sub-sample the position grid by this step (1 = every pixel).
    out          : output PNG path (default: <model_folder>/distance_landscape.png).
"""

from pathlib import Path

import fire
import matplotlib
matplotlib.use("Agg")  # headless compute node: render to file, no display
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.architectures import ImpalaEncoder
from eb_jepa.datasets.two_rooms.env import DotWall
from eb_jepa.datasets.two_rooms.normalizer import Normalizer
from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.utils import _resolve_env, load_env_data_config


def _build_encoder(cfg, img_size, device):
    """Rebuild the ImpalaEncoder exactly as in examples/ac_video_jepa/main.py."""
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=cfg.model.dobs,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(cfg.model.dobs, img_size, img_size),
    )
    return encoder.to(device).eval()


def _load_encoder_weights(encoder, ckpt_path, device):
    """Load only the `encoder.*` submodule weights from a JEPA training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    # strip torch.compile prefix, then keep the encoder submodule keys
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    enc_sd = {
        k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")
    }
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=True)
    return ckpt.get("epoch", "?")


@torch.no_grad()
def _encode(encoder, obs_4d, normalizer, device, batch=1024):
    """obs_4d: (N, 2, H, W) uint8/float -> embeddings (N, F) flattened."""
    obs = obs_4d.to(device).float()
    obs = normalizer.normalize_state(obs)          # per-image min-max + standardize
    obs = obs.unsqueeze(2)                          # (N, C, T=1, H, W)
    outs = []
    for i in range(0, obs.shape[0], batch):
        z = encoder(obs[i : i + batch])             # (b, F, 1, h, w)
        outs.append(z.flatten(1).cpu())
    return torch.cat(outs, dim=0)                    # (N, F)


def run(
    model_folder: str,
    checkpoint: str = "latest.pth.tar",
    s0: str = None,
    fix_wall: bool = True,
    stride: int = 1,
    out: str = None,
    device: str = None,
):
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    folder = Path(model_folder)
    cfg = OmegaConf.load(folder / "config.yaml")
    img_size = int(cfg.data.get("img_size", 65))

    # --- build the rendering env with a FIXED wall (so the landscape is meaningful) ---
    _, ConfigClass, _ = _resolve_env("two_rooms")
    data_overrides = OmegaConf.to_container(cfg.data, resolve=True)
    data_overrides.pop("pipeline", None)            # no data generation, just geometry
    data_overrides["fix_wall"] = fix_wall
    data_overrides["device"] = str(device)
    env_config = update_config_from_yaml(
        ConfigClass, load_env_data_config("two_rooms", data_overrides)
    )
    # fix_wall is read from the config (env_config.fix_wall), not a constructor kwarg
    env = DotWall(config=env_config, normalize=False)
    img_size = env.img_size

    # s_0 (pixel coords x,y). Default = center of the image.
    if s0 is None:
        s0_xy = [img_size / 2.0, img_size / 2.0]
    elif isinstance(s0, (tuple, list)):
        s0_xy = [float(v) for v in s0]            # fire parses "18,32" into a tuple
    else:
        s0_xy = [float(v) for v in str(s0).strip("()[]").split(",")]
    s0_t = torch.tensor(s0_xy, device=device, dtype=torch.float32)

    # Set up the (fixed) wall layout directly. We avoid env.reset(location=...) because
    # that code path never sets self.target_position and then crashes in _build_info.
    env.wall_x, env.hole_y = env._generate_wall()
    env.left_wall_x = env.wall_x - env.wall_width // 2
    env.right_wall_x = env.wall_x + env.wall_width // 2
    env.wall_img = env._render_walls(env.wall_x, env.hole_y)
    env.dot_position = s0_t
    print(f"[viz] device={device} img_size={img_size} "
          f"wall_x={float(env.wall_x):.1f} door_y={float(env.hole_y):.1f} s0={s0_xy}")

    # --- build encoder + load weights ---
    encoder = _build_encoder(cfg, img_size, device)
    epoch = _load_encoder_weights(encoder, folder / checkpoint, device)
    normalizer = Normalizer()

    # --- grid of ball positions s ---
    coords = torch.arange(0, img_size, stride, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")   # rows=y, cols=x
    locs = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (N, 2) = (x, y)
    gh, gw = gx.shape

    # render dot at every grid position, share the (fixed) wall channel
    dots = env._render_dot(locs)                                # (N, H, W) uint8
    wall = env.wall_img.unsqueeze(0).expand(dots.shape[0], -1, -1)
    obs_grid = torch.stack([dots, wall], dim=1)                 # (N, 2, H, W)

    # render s_0 the same way
    dot0 = env._render_dot(s0_t)                                # (H, W)
    obs0 = torch.stack([dot0, env.wall_img], dim=0).unsqueeze(0)  # (1, 2, H, W)

    # --- encode and compute distances ---
    z_grid = _encode(encoder, obs_grid, normalizer, device)     # (N, F)
    z0 = _encode(encoder, obs0, normalizer, device)             # (1, F)
    dist = (z_grid - z0).norm(dim=1).reshape(gh, gw).numpy()    # (gh, gw)

    # --- plot ---
    out_path = Path(out) if out else folder / "distance_landscape.png"
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(
        dist,
        origin="upper",
        extent=[0, img_size, img_size, 0],   # x: 0..img, y: 0..img (top-down)
        cmap="viridis",
    )
    # overlay the wall (channel 1) as faint white
    wall_np = env.wall_img.cpu().numpy()
    ax.imshow(
        np.ma.masked_where(wall_np == 0, wall_np),
        origin="upper",
        extent=[0, img_size, img_size, 0],
        cmap="gray_r",
        alpha=0.35,
    )
    ax.scatter([s0_xy[0]], [s0_xy[1]], c="red", marker="*", s=220,
               edgecolors="white", linewidths=1.0, label="$s_0$", zorder=5)
    ax.set_title(f"latent distance  $d(s_0, s) = \\|E(s_0)-E(s)\\|_2$\n"
                 f"epoch={epoch}  (fix_wall={fix_wall})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="latent L2 distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"[viz] saved -> {out_path}")


if __name__ == "__main__":
    fire.Fire(run)
