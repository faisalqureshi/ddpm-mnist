# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains experiments with **Denoising Diffusion Probabilistic Models (DDPM)** on MNIST. It includes pixel-space diffusion, autoencoders, and latent diffusion implementations.

## Project Structure

The repository is organized into independent experiment directories that share common utilities:

- **`common/`** - Shared utilities (checkpoint management, data loading, device setup, logging, SLURM support)
- **`mnist_single_digit/`** - Pixel-space DDPM with U-Net architecture
- **`mnist_ae/`** - MLP and Convolutional autoencoders for latent diffusion
- **`mnist_latent_diffusion_mlp/`** - Latent diffusion using MLP autoencoder (16D latent space)
- **`mnist_latent_diffusion_conv/`** - Latent diffusion using Convolutional autoencoder (16D latent space)

### Import Pattern

All projects use this pattern to import common utilities:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.ckpt import load_checkpoint, save_checkpoint_and_link_latest
from common.device import get_device, seed_everything
from common.data import HF_MNIST, make_mnist_loader
from common.logging import setup_component_logger
from common.slurm import install_signal_handlers, stop_requested
```

**Never use relative imports or sys.path hacks to import from sibling directories.**

## Training Commands

### Pixel-Space Diffusion (mnist_single_digit)

```bash
cd mnist_single_digit
python train.py \
    --epochs 30 \
    --batch-size 128 \
    --lr 2e-4 \
    --T 1000 \
    --only-digit 5 \
    --generate-images \
    --auto-resume
```

### Autoencoder Training (mnist_ae)

MLP autoencoder:
```bash
cd mnist_ae
python train.py --model mlp --epochs 25 --generate-images
```

Convolutional autoencoder:
```bash
cd mnist_ae
python train.py --model conv --epochs 100 --generate-images
```

### Latent Diffusion Training

MLP version:
```bash
cd mnist_latent_diffusion_mlp
python train.py \
    --ae-ckpt /path/to/mlp-autoencoder/checkpoint_epoch_000025.pt \
    --epochs 30 \
    --T 1000 \
    --batch-size 128 \
    --generate-images
```

Conv version:
```bash
cd mnist_latent_diffusion_conv
python train.py \
    --ae-ckpt /path/to/conv-autoencoder/checkpoint_epoch_000100.pt \
    --epochs 30 \
    --T 1000 \
    --batch-size 128 \
    --generate-images
```

### Image Generation (Inference)

```bash
# Pixel-space diffusion
cd mnist_single_digit
python infer.py --ckpt /path/to/checkpoint.pt --num-images 64

# Latent diffusion
cd mnist_latent_diffusion_mlp
python infer.py \
    --ckpt /path/to/diffusion/checkpoint.pt \
    --ae-ckpt /path/to/autoencoder/checkpoint.pt \
    --num-images 64
```

## Key Architectural Patterns

### Checkpoint Management

All training scripts use the common checkpoint utilities:

- Checkpoints saved to `$SCRATCH/checkpoints/` (SLURM) or `./outputs/checkpoints/` (local)
- Format: `checkpoint_epoch_{epoch:06d}.pt`
- Includes: model, optimizer, scaler, epoch, global_step, args, exp_name, timestamp
- Automatic `latest.pt` symlink creation via `save_checkpoint_and_link_latest()`
- Resume with `--auto-resume` (finds latest) or `--resume /path/to/checkpoint.pt`

### Experiment Naming Convention

Experiment names are auto-generated with this pattern:
```
{model}-d{digit}-bs{batch_size}-lr{lr}-seed{seed}-{timestamp}
```

Examples:
- `mlp-dall-bs128-lr0.0002-seed42-20251207-103045`
- `latent-diffusion-d5-bs128-lr0.0002-T1000-seed42-20251207-103045`

### Time Conditioning (FiLM)

Both pixel-space and latent-space models use **FiLM (Feature-wise Linear Modulation)** for time conditioning:

1. Sinusoidal timestep embeddings → TimeMLP → time_emb (256D)
2. FiLM conditioning: `h = (1 + γ) * norm(x) + β` where γ, β = Linear(time_emb)
3. Pixel-space: FiLMCond uses GroupNorm + Conv2d
4. Latent-space: FiLMCondMLP uses LayerNorm + Linear

### Diffusion Schedules

Standard DDPM setup across all implementations:
- **T = 1000** timesteps
- **Linear noise schedule**: β ∈ [1e-4, 2e-2]
- Precomputed in `precompute_schedules()`: βₜ, αₜ, ᾱₜ, √ᾱₜ, √(1-ᾱₜ)
- Training: `q_sample()` adds noise, model predicts ε
- Inference: `generate_images()` does DDPM sampling (not DDIM)

### Latent Diffusion Architecture

Latent diffusion requires pretrained autoencoder:

1. **Training phase**:
   - Encoder (frozen) maps images → 16D latents
   - Add noise to latents: z₀ → zₜ
   - LatentDenoiser predicts noise from zₜ
   - Loss: MSE(predicted_noise, actual_noise)

2. **Generation phase**:
   - Sample z_T ~ N(0, I) in 16D space
   - Iteratively denoise T steps → z₀
   - Decoder (frozen) maps z₀ → image

**Important**: Latent diffusion imports autoencoder using `importlib.util` to avoid path conflicts:
```python
spec = importlib.util.spec_from_file_location("ae_module", str(Path(__file__).parent / "../mnist_ae/model.py"))
ae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae_module)
```

## SLURM Environment

The codebase is designed to run on SLURM clusters (Alliance Canada/SHARCNET):

### Environment Setup

```bash
# On SLURM cluster
salloc --time=0:10:0 --mem=4000M --gpus-per-node=1 --ntasks=1 --account=def-fqureshi
source mnist_single_digit/setup.sh  # Loads modules and activates venv
```

Modules loaded:
- python/3.12
- python-build-bundle/2024a
- scipy-stack/2025a
- opencv/4.12.0
- arrow/21.0.0

### SLURM Integration

- Uses `$SLURM_TMPDIR` for dataset cache (HuggingFace datasets)
- Uses `$SCRATCH` for outputs (checkpoints, samples, logs, tensorboard runs)
- Signal handlers via `common.slurm.install_signal_handlers()` for graceful SLURM timeout handling
- Auto-checkpointing every 300 seconds (configurable via `--ckpt-every-sec`)
- Check `stop_requested()` to save checkpoint before SLURM timeout

### Job Submission Example

See `mnist_single_digit/job.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=ddpm-train
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.12 python-build-bundle/2024a scipy-stack/2025a opencv/4.12.0 arrow/21.0.0
source "$HOME/venv/ddpm/bin/activate"

python train.py --data-root="$SLURM_TMPDIR" --outdir="$SCRATCH/ddpm-output" --epochs=100 --auto-resume --amp
```

## Output Directory Structure

```
$SCRATCH/  (or ./outputs/)
├── checkpoints/
│   ├── {exp_name}/
│   │   ├── checkpoint_epoch_000001.pt
│   │   ├── checkpoint_epoch_000002.pt
│   │   └── latest.pt -> checkpoint_epoch_000002.pt
├── samples/
│   ├── {exp_name}/
│   │   ├── epoch_000001.png
│   │   └── epoch_000002.png
├── runs/
│   └── {exp_name}/  # TensorBoard logs
└── logs/
    └── {exp_name}.log
```

## Common Utilities Reference

### `common.ckpt`
- `save_checkpoint()` - Atomic checkpoint save
- `load_checkpoint()` - Load and restore state
- `find_latest_checkpoint()` - Find latest checkpoint in directory
- `find_latest_autoencoder_checkpoint()` - Find latest AE checkpoint by model prefix
- `save_checkpoint_and_link_latest()` - Save + create `latest.pt` symlink
- `find_latest_experiment()` - Find most recent experiment directory

### `common.device`
- `get_device()` - Auto-detect CUDA/MPS/CPU
- `seed_everything()` - Seed all RNGs (torch, numpy, random)
- `seed_worker()` - DataLoader worker init function
- `set_seed()` - Set global seed
- `make_output_directories()` - Create output directory structure

### `common.data`
- `HF_MNIST` - HuggingFace MNIST wrapper (can filter by digit)
- `make_mnist_loader()` - Create DataLoader with proper seeding
- `split_train_val()` - Split dataset into train/val

### `common.logging`
- `setup_component_logger()` - Create logger for component
- `log_args_rich()` - Pretty-print argparse arguments

### `common.slurm`
- `install_signal_handlers()` - Handle SLURM timeout signals
- `stop_requested()` - Check if graceful stop was requested

## Common Training Arguments

All training scripts support these arguments:

- `--data-root` - HuggingFace cache dir (default: `$SLURM_TMPDIR` or `./hf_cache`)
- `--outdir` - Output directory (default: `$SCRATCH` or `./outputs`)
- `--only-digit` - Train on single digit (0-9, or None for all digits)
- `--epochs` - Number of training epochs
- `--batch-size` - Batch size (default: 128)
- `--lr` - Learning rate (default: 2e-4)
- `--seed` - Random seed (default: 42)
- `--device` - Force device (cuda/mps/cpu, or None for auto-detect)
- `--amp` - Enable automatic mixed precision (CUDA only)
- `--resume` - Resume from specific checkpoint path
- `--auto-resume` - Resume from latest checkpoint automatically
- `--generate-images` - Generate sample images each epoch
- `--ckpt-every-sec` - Auto-checkpoint interval in seconds (default: 300)
- `--ckpt-save-every` - Save checkpoint every N epochs (default: 1)

Diffusion-specific:
- `--T` - Number of diffusion timesteps (default: 1000)
- `--beta-start` - Noise schedule start (default: 1e-4)
- `--beta-end` - Noise schedule end (default: 2e-2)

Latent diffusion-specific:
- `--ae-ckpt` - Path to pretrained autoencoder checkpoint (required)
- `--latent-dim` - Latent dimension (default: 16)
- `--hidden-dim` - Denoiser hidden dimension (default: 256)

## Development Notes

- All training scripts use **AdamW** optimizer by default
- TensorBoard logging is automatic (writes to `{outdir}/runs/{exp_name}/`)
- Images are saved as grids using `torchvision.utils.save_image()`
- Latent diffusion requires training autoencoder first
- Encoder is **always frozen** during latent diffusion training
- Use `--debug` flag to enable debug-level logging
