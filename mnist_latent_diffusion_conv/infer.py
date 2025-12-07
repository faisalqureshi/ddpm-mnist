#
# infer.py - Generate images using trained latent diffusion model
#

from pathlib import Path
import sys

# Add paths for dependencies
_single_digit_path = str(Path(__file__).parent / "../mnist_single_digit")
_ae_path = str(Path(__file__).parent / "../mnist_ae")
sys.path.insert(0, _single_digit_path)
sys.path.insert(0, _ae_path)

import argparse
import torch
from torchvision import utils as tvutils

# Import AE model module directly (Conv version)
import importlib.util
spec = importlib.util.spec_from_file_location("ae_module", str(Path(__file__).parent / "../mnist_ae/model_conv.py"))
ae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae_module)

# Import local model
from model import LatentDenoiser, precompute_schedules, generate_images

def load_autoencoder(ckpt_path: Path, device):
    """Load pretrained autoencoder decoder"""
    ae = ae_module.Autoencoder(latent_dim=16, use_sigmoid=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    return ae.decoder

def main():
    parser = argparse.ArgumentParser("Generate images with latent diffusion")
    parser.add_argument("--ckpt", type=str, required=True, help="Diffusion model checkpoint")
    parser.add_argument("--ae-ckpt", type=str, required=True, help="Autoencoder checkpoint")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--output", type=str, default="generated.png")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load autoencoder decoder
    print(f"Loading autoencoder from {args.ae_ckpt}")
    decoder = load_autoencoder(Path(args.ae_ckpt), device)

    # Load diffusion model
    print(f"Loading diffusion model from {args.ckpt}")
    net = LatentDenoiser(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        time_dim=256
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()

    # Precompute schedule
    sched = precompute_schedules(args.T, args.beta_start, args.beta_end, device)

    # Generate images
    print(f"Generating {args.num_images} images...")
    with torch.no_grad():
        imgs = generate_images(
            net, decoder, sched,
            n=args.num_images,
            latent_dim=args.latent_dim,
            device=device
        ).cpu()

    # Save grid
    nrow = int(args.num_images ** 0.5)
    grid = tvutils.make_grid(imgs, nrow=nrow, padding=2)
    tvutils.save_image(grid, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
