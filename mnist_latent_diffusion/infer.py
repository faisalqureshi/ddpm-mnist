#
# infer.py - Generate images using trained latent diffusion model
# Automatically detects and loads the correct autoencoder (MLP or Conv) from checkpoint
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

# Import BOTH AE model modules
import importlib.util

# Import MLP autoencoder
spec_mlp = importlib.util.spec_from_file_location("ae_module_mlp",
                                                    str(Path(__file__).parent / "../mnist_ae/model.py"))
ae_module_mlp = importlib.util.module_from_spec(spec_mlp)
spec_mlp.loader.exec_module(ae_module_mlp)

# Import Conv autoencoder
spec_conv = importlib.util.spec_from_file_location("ae_module_conv",
                                                     str(Path(__file__).parent / "../mnist_ae/model_conv.py"))
ae_module_conv = importlib.util.module_from_spec(spec_conv)
spec_conv.loader.exec_module(ae_module_conv)

# Import local model
from model import LatentDenoiser, precompute_schedules, generate_images

def detect_ae_model_type(ae_ckpt_path: Path) -> str:
    """Detect autoencoder model type from checkpoint"""
    ckpt = torch.load(ae_ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt["args"], dict):
        return ckpt["args"]["model"]
    else:
        return ckpt["args"].model

def load_autoencoder(ae_ckpt_path: Path, device, latent_dim: int, model_type: str):
    """Load the correct autoencoder based on model type"""

    if model_type == "ae-conv":
        ae_module = ae_module_conv
    elif model_type == "ae-mlp":
        ae_module = ae_module_mlp
    else:
        raise ValueError(f"Unknown autoencoder model type: {model_type}")

    ae = ae_module.Autoencoder(latent_dim=latent_dim, use_sigmoid=True).to(device)
    ckpt = torch.load(ae_ckpt_path, map_location=device, weights_only=False)
    ae.load_state_dict(ckpt["model"])
    ae.eval()

    print(f"Loaded {model_type} autoencoder from {ae_ckpt_path}")
    return ae.decoder

def main():
    parser = argparse.ArgumentParser("Generate images with latent diffusion")
    parser.add_argument("--ckpt", type=str, required=True, help="Diffusion model checkpoint")
    parser.add_argument("--ae-ckpt", type=str, default=None, help="Autoencoder checkpoint (auto-detected if not provided)")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--output", type=str, default="generated.png")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load diffusion checkpoint to extract architecture parameters
    print(f"Loading diffusion model from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # Extract architecture parameters from checkpoint
    latent_dim = ckpt_args.get("latent_dim", 16)
    hidden_dim = ckpt_args.get("hidden_dim", 256)
    T = ckpt_args.get("T", 1000)
    beta_start = ckpt_args.get("beta_start", 1e-4)
    beta_end = ckpt_args.get("beta_end", 2e-2)

    print(f"Architecture from checkpoint: latent_dim={latent_dim}, hidden_dim={hidden_dim}, T={T}")

    # Determine AE checkpoint path
    ae_ckpt_path = args.ae_ckpt
    if not ae_ckpt_path and "ae_ckpt_path" in ckpt_args:
        ae_ckpt_path = ckpt_args["ae_ckpt_path"]
        print(f"Auto-detected autoencoder checkpoint from diffusion checkpoint: {ae_ckpt_path}")
    elif not ae_ckpt_path:
        print("Error: --ae-ckpt not provided and not found in diffusion checkpoint metadata")
        exit(-1)

    # Detect autoencoder type and load decoder
    ae_model_type = detect_ae_model_type(Path(ae_ckpt_path))
    print(f"Detected autoencoder type: {ae_model_type}")

    print(f"Loading autoencoder from {ae_ckpt_path}")
    decoder = load_autoencoder(Path(ae_ckpt_path), device, latent_dim, ae_model_type)

    # Build diffusion model with checkpoint parameters
    net = LatentDenoiser(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        time_dim=256
    ).to(device)

    net.load_state_dict(ckpt["model"])
    net.eval()

    # Precompute schedule
    sched = precompute_schedules(T, beta_start, beta_end, device)

    # Generate images
    print(f"Generating {args.num_images} images...")
    with torch.no_grad():
        imgs = generate_images(
            net, decoder, sched,
            n=args.num_images,
            latent_dim=latent_dim,
            device=device
        ).cpu()

    # Save grid
    nrow = int(args.num_images ** 0.5)
    grid = tvutils.make_grid(imgs, nrow=nrow, padding=2)
    tvutils.save_image(grid, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
