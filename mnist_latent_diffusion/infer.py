#
# infer.py - Generate images using trained latent diffusion model
# Automatically detects and loads the correct autoencoder (MLP or Conv) from checkpoint
#

from pathlib import Path
import sys

# Add parent directory to path for common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import logging
from torchvision import utils as tvutils

# Import common utilities
from common.device import get_device

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

# Import latent diffusion model (local)
spec_local = importlib.util.spec_from_file_location("local_model", str(Path(__file__).parent / "model.py"))
local_model = importlib.util.module_from_spec(spec_local)
spec_local.loader.exec_module(local_model)

LatentDenoiser = local_model.LatentDenoiser
precompute_schedules = local_model.precompute_schedules
generate_images = local_model.generate_images
generate_images_with_process = local_model.generate_images_with_process

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
    parser.add_argument("--digit", type=int, default=None, help="Generate specific digit (0-9), or None for all digits")
    parser.add_argument("--output", type=str, default="generated.png")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--show-process", action="store_true", help="Show denoising process (only works with single digit and num-images=1)")
    parser.add_argument("--process-steps", type=int, default=20, help="Number of intermediate steps to capture")
    args = parser.parse_args()

    # Setup simple logger for device detection
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    # Auto-detect device (CUDA > MPS > CPU)
    device = get_device(logger, args.device)

    # Load diffusion checkpoint to extract architecture parameters
    print(f"Loading diffusion model from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # Extract architecture parameters from checkpoint
    hidden_dim = ckpt_args.get("hidden_dim", 256)
    T = ckpt_args.get("T", 1000)
    beta_start = ckpt_args.get("beta_start", 1e-4)
    beta_end = ckpt_args.get("beta_end", 2e-2)

    # Extract autoencoder metadata from diffusion checkpoint
    ae_latent_dim = ckpt_args.get("ae_latent_dim", 16)
    ae_model_type = ckpt_args.get("ae_model", None)

    print(f"Architecture from checkpoint: ae_latent_dim={ae_latent_dim}, hidden_dim={hidden_dim}, T={T}")

    # Determine AE checkpoint path
    ae_ckpt_path = args.ae_ckpt
    if not ae_ckpt_path and "ae_ckpt" in ckpt_args:
        ae_ckpt_path = ckpt_args["ae_ckpt"]
        print(f"Auto-detected autoencoder checkpoint from diffusion checkpoint: {ae_ckpt_path}")
    elif not ae_ckpt_path:
        print("Error: --ae-ckpt not provided and not found in diffusion checkpoint metadata")
        exit(-1)

    # Detect autoencoder type if not in diffusion checkpoint metadata
    if not ae_model_type:
        ae_model_type = detect_ae_model_type(Path(ae_ckpt_path))
        print(f"Detected autoencoder type from AE checkpoint: {ae_model_type}")
    else:
        print(f"Using autoencoder type from diffusion checkpoint: {ae_model_type}")

    print(f"Loading autoencoder from {ae_ckpt_path}")
    decoder = load_autoencoder(Path(ae_ckpt_path), device, ae_latent_dim, ae_model_type)

    # Build diffusion model with checkpoint parameters
    net = LatentDenoiser(
        latent_dim=ae_latent_dim,
        hidden_dim=hidden_dim,
        time_dim=256
    ).to(device)

    net.load_state_dict(ckpt["model"])
    net.eval()

    # Precompute schedule
    sched = precompute_schedules(T, beta_start, beta_end, device)

    # Generate images
    if args.digit is not None:
        print(f"Generating {args.num_images} images of digit {args.digit}...")
        labels = torch.full((args.num_images,), args.digit, dtype=torch.long, device=device)
    else:
        print(f"Generating {args.num_images} images with mixed digits...")
        labels = torch.arange(args.num_images, device=device) % 10

    with torch.no_grad():
        if args.show_process:
            # Generate with intermediate steps
            print(f"Generating with {args.process_steps} intermediate steps...")
            intermediates = generate_images_with_process(
                net, decoder, sched,
                n=args.num_images,
                latent_dim=ae_latent_dim,
                labels=labels,
                device=device,
                num_steps=args.process_steps
            ).cpu()  # Shape: (num_steps, n, 1, 28, 28)

            # Save each intermediate frame
            output_path = Path(args.output)
            output_dir = output_path.parent
            output_stem = output_path.stem
            output_ext = output_path.suffix

            frame_paths = []
            nrow = int(args.num_images ** 0.5)

            for step_idx in range(intermediates.shape[0]):
                frame = intermediates[step_idx]  # (n, 1, 28, 28)
                grid = tvutils.make_grid(frame, nrow=nrow, padding=2)
                frame_path = output_dir / f"{output_stem}_frame_{step_idx:03d}{output_ext}"
                tvutils.save_image(grid, str(frame_path))
                frame_paths.append(str(frame_path))

            # Output JSON for server to parse
            import json
            result = {
                "type": "process",
                "num_steps": args.process_steps,
                "frames": frame_paths,
                "final_image": frame_paths[-1]
            }
            print(f"PROCESS_RESULT: {json.dumps(result)}")
        else:
            # Standard generation
            imgs = generate_images(
                net, decoder, sched,
                n=args.num_images,
                latent_dim=ae_latent_dim,
                labels=labels,
                device=device
            ).cpu()

            # Save grid
            nrow = int(args.num_images ** 0.5)
            grid = tvutils.make_grid(imgs, nrow=nrow, padding=2)
            tvutils.save_image(grid, args.output)
            print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
