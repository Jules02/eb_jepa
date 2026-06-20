"""Train (or fine-tune) the latent->field decoder stored in a checkpoint.

Usage:
    python -m examples.gray_scott.train_decoder --ckpt <path> [--epochs 30] [--init_from <other_ckpt>]

--init_from warm-starts the decoder from another checkpoint's decoder weights (e.g. train
epoch_25's decoder from epoch_10's — same run, similar latent -> far fewer epochs to converge).
"""
import os
import sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader
from examples.gray_scott.eval import _FrameDecoder, load_jepa


def main():
    ckpt_path = sys.argv[sys.argv.index("--ckpt") + 1]
    extra_epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 30
    init_from = sys.argv[sys.argv.index("--init_from") + 1] if "--init_from" in sys.argv else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    _, encoder = load_jepa(ckpt, device)
    dstc = int(OmegaConf.create(ckpt["cfg"]).model.dstc)

    decoder = _FrameDecoder(D=dstc).to(device)
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
        print(f"[decoder] resumed from this checkpoint's own decoder", flush=True)
    elif init_from is not None:
        src = torch.load(init_from, map_location=device, weights_only=False)
        if "decoder" not in src:
            raise KeyError(f"--init_from {init_from} has no 'decoder' key")
        decoder.load_state_dict(src["decoder"])
        print(f"[decoder] WARM-START from {init_from}", flush=True)
    else:
        print("[decoder] no saved weights, starting from scratch", flush=True)

    opt = torch.optim.Adam(decoder.parameters(), lr=3e-4)
    dcfg = GrayScottConfig(split="train", epoch_size=4000, batch_size=8, num_workers=4)
    loader = make_loader(dcfg)

    prev_mse = float("inf")
    patience = 0
    for ep in range(extra_epochs):
        decoder.train()
        total, n = 0.0, 0
        for batch in loader:
            x = batch["video"].to(device)
            with torch.no_grad():
                z = encoder(x)
            recon = decoder(z)
            loss = nn.functional.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item(); n += 1
        mse = total / n
        print(f"[decoder] ep{ep:02d} mse={mse:.5f}", flush=True)

        # early stopping: < 0.5% relative improvement for 3 consecutive epochs
        if (prev_mse - mse) / (prev_mse + 1e-8) < 0.005:
            patience += 1
            if patience >= 3:
                print(f"[decoder] converged at ep{ep}, stopping early", flush=True)
                break
        else:
            patience = 0
        prev_mse = mse

    decoder.eval()
    ckpt["decoder"] = decoder.state_dict()
    tmp = f"{ckpt_path}.tmp.{os.getpid()}"
    torch.save(ckpt, tmp)
    os.replace(tmp, ckpt_path)                          # atomic
    print(f"[decoder] saved to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
