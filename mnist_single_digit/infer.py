#
# infer.py
#

import argparse
from pathlib import Path
import torch
from torchvision.utils import make_grid, save_image
from model import Denoiser, precompute_schedules, generate_images

def main():
    parser = argparse.ArgumentParser("DDPM MNIST inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="samples_infer.png")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", None)
    if ckpt_args is None:
        raise RuntimeError("Checkpoint missing 'args' dict; cannot rebuild schedule. Re-train with newer train.py.")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate model & schedules exactly as trained
    net = Denoiser(base_ch=64, time_dim=256).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    T = ckpt_args["T"]; beta_start = ckpt_args["beta_start"]; beta_end = ckpt_args["beta_end"]
    sched = precompute_schedules(T, beta_start, beta_end, device)

    with torch.no_grad():
        imgs = generate_images(net, sched, n=args.n, device=device).cpu()
    grid = make_grid(imgs, nrow=int(args.n ** 0.5), padding=2)
    save_image(grid, args.out)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
