# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains experiments with **Denoising Diffusion Probabilistic Models (DDPM)** on MNIST. It includes pixel-space diffusion, autoencoders, and latent diffusion implementations.

## Project Structure

The repository is organized into independent experiment directories that share common utilities:

- **`common/`** - Shared utilities (checkpoint management, data loading, device setup, logging, SLURM support)
- **`mnist_single_digit/`** - Pixel-space DDPM with U-Net architecture
- **`mnist_ae/`** - MLP and Convolutional autoencoders for latent diffusion
- **`mnist_latent_diffusion/`** - **[RECOMMENDED]** Unified latent diffusion (automatically detects and works with both MLP and Conv autoencoders)
- **`mnist_latent_diffusion_mlp/`** - Legacy: MLP-specific latent diffusion (use `mnist_latent_diffusion/` instead)
- **`mnist_latent_diffusion_conv/`** - Legacy: Conv-specific latent diffusion (use `mnist_latent_diffusion/` instead)

### Import Pattern

All projects use this pattern to import common utilities:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.ckpt import load_checkpoint, save_checkpoint_and_link_latest, resolve_resume_path
from common.device import get_device, seed_everything
from common.data import HF_MNIST, make_mnist_loader
from common.logger_utils import setup_component_logger
from common.slurm import install_signal_handlers, stop_requested
from common import emoji, error_codes
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

**IMPORTANT**: The `--model` flag is **required** and must be either `mlp` or `conv`.

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

The `--model` flag determines:
- Which architecture to use (MLP vs Convolutional)
- Experiment naming (used in checkpoint/sample directory names)
- Checkpoint metadata (stored in checkpoint file for auto-detection)

### Latent Diffusion Training

The unified implementation automatically detects whether to use MLP or Conv autoencoder from the checkpoint:

```bash
cd mnist_latent_diffusion
python train.py \
    --ae-ckpt /path/to/autoencoder/checkpoint.pt \
    --epochs 30 \
    --T 1000 \
    --batch-size 128 \
    --generate-images
```

**Resume from checkpoint** (ae_ckpt_path automatically extracted):
```bash
cd mnist_latent_diffusion
python train.py \
    --resume /path/to/latent-diffusion/checkpoint.pt \
    --epochs 50
```

When resuming, the autoencoder checkpoint path is automatically extracted from the diffusion checkpoint metadata, so you don't need to specify `--ae-ckpt` again.

**Auto-resume** (finds latest latent-diffusion experiment):
```bash
cd mnist_latent_diffusion
python train.py \
    --auto-resume \
    --epochs 50
```

The script can also auto-select the latest autoencoder if neither `--ae-ckpt` nor `--resume` is provided:

```bash
cd mnist_latent_diffusion
python train.py \
    --epochs 30 \
    --T 1000 \
    --batch-size 128 \
    --generate-images
```

### Image Generation (Inference)

Pixel-space diffusion:
```bash
cd mnist_single_digit
python infer.py \
    --checkpoint /path/to/checkpoint.pt \
    --out samples.png \
    --n 64
```

Latent diffusion (automatically detects autoencoder type):
```bash
cd mnist_latent_diffusion
python infer.py \
    --ckpt /path/to/diffusion/checkpoint.pt \
    --num-images 64 \
    --output generated.png
```

The `--ae-ckpt` parameter is **optional** - if not provided, the script automatically extracts the autoencoder checkpoint path from the diffusion checkpoint metadata. You can override this by specifying `--ae-ckpt` explicitly:

```bash
cd mnist_latent_diffusion
python infer.py \
    --ckpt /path/to/diffusion/checkpoint.pt \
    --ae-ckpt /path/to/different/autoencoder.pt \
    --num-images 64
```

**Note**: Inference scripts extract all necessary parameters (T, beta_start, beta_end, latent_dim, hidden_dim, ae_ckpt_path) from checkpoint metadata, so you don't need to specify them manually.

### Utility Scripts

**Latent Space Visualization** (mnist_ae/run_tsne.py):

Visualizes autoencoder latent space using t-SNE dimensionality reduction:

```bash
cd mnist_ae
python run_tsne.py \
    --checkpoint /path/to/autoencoder/checkpoint.pt \
    --output tsne_plot.png \
    --n-samples 5000
```

The script:
- Automatically detects autoencoder type (MLP or Conv) from checkpoint
- Encodes MNIST images to latent space
- Applies t-SNE to reduce from 16D to 2D
- Colors points by digit class
- Computes clustering metrics (silhouette score, Calinski-Harabasz score)

**Checkpoint Inspection** (common/inspect_ckpt.py):

Inspect contents of a checkpoint file:

```bash
python -m common.inspect_ckpt /path/to/checkpoint.pt
```

**Find Best Checkpoint** (common/find_best_ckpt.py):

Find the best checkpoint based on metrics (val_loss, train_loss, epoch, etc.):

```bash
# Find checkpoint with lowest validation loss
python -m common.find_best_ckpt /path/to/checkpoints --metric val_loss --mode min

# Find checkpoint with highest epoch number (latest)
python -m common.find_best_ckpt /path/to/checkpoints --metric epoch --mode max

# Copy best checkpoint to a specific location
python -m common.find_best_ckpt /path/to/checkpoints --metric val_loss --copy-to best_model.pt

# Print only the path (for scripting)
python -m common.find_best_ckpt /path/to/checkpoints --metric val_loss --print-path-only
```

The script can extract losses from log files if available, or use checkpoint metadata.

The `run_tsne.py` script generates three output files in the specified output directory:
- `tsne_latent_space.png` - All digits colored by class on a single plot
- `tsne_by_digit.png` - 10 subplots, one per digit
- `reconstructions.png` - Sample original vs reconstructed images

**Interactive Latent Space Visualization** (mnist_ae/tsne_visualization.ipynb):

For interactive exploration, use the Jupyter notebook:

```bash
cd mnist_ae
jupyter notebook tsne_visualization.ipynb
```

The notebook automatically detects autoencoder type (MLP or Conv) from the checkpoint and provides:
- Interactive t-SNE visualization with matplotlib
- Individual digit plots
- Cluster quality metrics (silhouette score, Calinski-Harabasz score)
- Sample reconstructions

## Key Architectural Patterns

### Checkpoint Management

All training scripts use the common checkpoint utilities:

- Checkpoints saved to `$SCRATCH/checkpoints/` (SLURM) or `./outputs/checkpoints/` (local)
- Format: `checkpoint_epoch_{epoch:06d}.pt`
- Includes: model, optimizer, scaler, epoch, global_step, args, exp_name, timestamp
- Automatic `latest.pt` symlink creation via `save_checkpoint_and_link_latest()`
- Resume with `--auto-resume` (finds latest) or `--resume /path/to/checkpoint.pt`

### Experiment Naming Convention

Experiment names are auto-generated with patterns specific to each model type:

**Autoencoder:**
```
ae-{model}-d{digit}-bs{batch_size}-lr{lr}-D{latent_dim}-seed{seed}-{timestamp}
```
Example: `ae-conv-dall-bs128-lr0.0002-D16-seed42-20251207-103045`

**Latent Diffusion:**
```
lddpm-E{ae_model}-d{digit}-bs{batch_size}-lr{lr}-D{latent_dim}-T{timesteps}-seed{seed}-{timestamp}
```
Example: `lddpm-Eae-conv-dall-bs128-lr0.0002-D16-T1000-seed42-20251207-103045`

The experiment names include:
- **model**: `ae-conv`, `ae-mlp`, or `lddpm` (latent DDPM)
- **ae_model**: For latent diffusion, the encoder type used (e.g., `ae-conv`)
- **digit**: `dall` for all digits, or specific digit `0-9`
- **latent_dim**: Dimension of the latent space (typically 16)

**Important checkpoint metadata** stored for reproducibility:
- Autoencoder: `model` (ae-conv/ae-mlp), `latent_dim`, `only_digit`
- Latent diffusion: `ae_model`, `ae_latent_dim`, `ae_only_digit`, `ae_ckpt`, `ae_ckpt_data`

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

**Important**: Latent diffusion imports autoencoder modules using `importlib.util` to avoid path conflicts:
```python
# Import both MLP and Conv autoencoder modules
spec_mlp = importlib.util.spec_from_file_location("ae_module_mlp",
                                                    str(Path(__file__).parent / "../mnist_ae/model.py"))
ae_module_mlp = importlib.util.module_from_spec(spec_mlp)
spec_mlp.loader.exec_module(ae_module_mlp)

spec_conv = importlib.util.spec_from_file_location("ae_module_conv",
                                                     str(Path(__file__).parent / "../mnist_ae/model_conv.py"))
ae_module_conv = importlib.util.module_from_spec(spec_conv)
spec_conv.loader.exec_module(ae_module_conv)
```

The correct module is selected at runtime by reading the `model` field from the autoencoder checkpoint:
```python
# Load checkpoint and detect type
ckpt = torch.load(ae_ckpt_path, map_location="cpu", weights_only=False)
model_type = ckpt["args"]["model"]  # "mlp" or "conv"

# Select appropriate module
ae_module = ae_module_conv if model_type == "conv" else ae_module_mlp
ae = ae_module.Autoencoder(latent_dim=latent_dim, use_sigmoid=True).to(device)
```

This pattern is used in:
- `mnist_latent_diffusion/train.py` - Training latent diffusion
- `mnist_latent_diffusion/infer.py` - Generating images
- `mnist_ae/run_tsne.py` - Visualizing latent space

## SLURM Environment

The codebase is designed to run on SLURM clusters (Alliance Canada/SHARCNET):

### Environment Setup

```bash
# On SLURM cluster
salloc --time=0:10:0 --mem=4000M --gpus-per-node=1 --ntasks=1 --account=def-fqureshi
source mnist_single_digit/setup.sh  # Loads modules and activates venv
```

The `setup.sh` script loads required modules and activates the virtual environment at `~/venv/ddpm/`:

Modules loaded:
- python/3.12
- python-build-bundle/2024a
- scipy-stack/2025a
- opencv/4.12.0
- arrow/21.0.0

**Note**: Ensure you have created the virtual environment at `~/venv/ddpm/` with PyTorch, torchvision, datasets (HuggingFace), tqdm, tensorboard, scikit-learn, and matplotlib installed.

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

python train.py --cache-dir="$SLURM_TMPDIR" --outdir="$SCRATCH/ddpm-output" --epochs=100 --auto-resume --amp
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
- `find_latest_checkpoint()` - Find latest checkpoint in directory by model prefix
- `resolve_resume_path()` - Unified resume logic handling --resume and --auto-resume
- `infer_model_from_checkpoint()` - Extract model type from checkpoint metadata
- `save_checkpoint_and_link_latest()` - Save + create `latest.pt` symlink
- `inspect_checkpoint()` - Display checkpoint metadata and structure

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

### `common.logger_utils`
- `setup_component_logger()` - Create logger for component with file and console handlers
- `log_args_rich()` - Pretty-print argparse arguments

### `common.emoji`
- UI constants for consistent visual feedback: `step`, `error`, `warning`, `ok`, `info`

### `common.error_codes`
- Standard exit codes: `MODEL_MISMATCH`, `NO_MODEL_SPECIFIED`, `NO_AE_CHECKPOINT`

### `common.slurm`
- `install_signal_handlers()` - Handle SLURM timeout signals
- `stop_requested()` - Check if graceful stop was requested

## Command-Line Arguments Reference

### Common Training Arguments

Arguments supported by most/all training scripts:

- `--cache-dir` - HuggingFace cache directory (default: `$SLURM_TMPDIR` or `./hf_cache`)
- `--outdir` - Output directory (default: `$SCRATCH` or `./outputs`)
- `--logdir` - Log directory (default: `{outdir}/logs`)
- `--only-digit` - Train on single digit (0-9, or None for all digits)
- `--epochs` - Number of training epochs
- `--batch-size` - Batch size (default: 128)
- `--num-workers` - DataLoader workers (default: `$SLURM_CPUS_PER_TASK` or 1/4)
- `--lr` - Learning rate (default: 2e-4)
- `--seed` - Random seed (default: 42)
- `--device` - Force device (cuda/mps/cpu, or None for auto-detect)
- `--amp` - Enable automatic mixed precision (CUDA only)
- `--resume` - Resume from specific checkpoint path
- `--auto-resume` - Resume from latest checkpoint automatically
- `--generate-images` - Generate sample images each epoch
- `--ckpt-every-sec` - Auto-checkpoint interval in seconds (default: 300)
- `--ckpt-save-every` - Save checkpoint every N epochs (default: 1)
- `--debug` - Enable debug-level logging

### Autoencoder-Specific Arguments (mnist_ae/train.py)

- `--model` - **REQUIRED**: `mlp` or `conv` (chooses architecture)
- `--latent-dim` - Latent dimension (default: 16)
- `--lr-scheduler` - Learning rate scheduler: `none`, `cosine`, or `step` (default: cosine)
- `--early-stopping-patience` - Early stopping patience in epochs (default: None)
- `--grad-clip` - Gradient clipping max norm (default: 1.0, 0 to disable)

### Diffusion-Specific Arguments (mnist_single_digit/train.py, mnist_latent_diffusion/train.py)

- `--T` - Number of diffusion timesteps (default: 1000)
- `--beta-start` - Noise schedule start (default: 1e-4)
- `--beta-end` - Noise schedule end (default: 2e-2)
- `--samples-per-epoch` - Images to generate per epoch (default: 16)

### Latent Diffusion-Specific Arguments (mnist_latent_diffusion/train.py)

- `--ae-ckpt` - Path to pretrained autoencoder checkpoint
- `--ae-ckpt-dir` - Directory with autoencoder checkpoints (uses latest if `--ae-ckpt` not provided)
- `--hidden-dim` - Denoiser hidden dimension (default: 256)
- `--early-stopping-patience` - Early stopping patience in epochs (default: None)

**Note**: If neither `--ae-ckpt` nor `--ae-ckpt-dir` is provided, the script attempts to auto-find the latest autoencoder checkpoint (tries Conv first, then MLP).

### Inference Arguments

**Pixel-space diffusion** (mnist_single_digit/infer.py):
- `--checkpoint` - **REQUIRED**: Path to diffusion model checkpoint
- `--out` - Output image path (default: `samples_infer.png`)
- `--n` - Number of images to generate (default: 16)
- `--device` - Force device (cuda/mps/cpu, or None for auto-detect)

**Latent diffusion** (mnist_latent_diffusion/infer.py):
- `--ckpt` - **REQUIRED**: Path to diffusion model checkpoint
- `--ae-ckpt` - **REQUIRED**: Path to autoencoder checkpoint
- `--num-images` - Number of images to generate (default: 64)
- `--output` - Output image path (default: `generated.png`)
- `--device` - Force device (cuda/mps/cpu, or None for auto-detect)

## Development Notes

- All training scripts use **AdamW** optimizer by default
- TensorBoard logging is automatic (writes to `{outdir}/runs/{exp_name}/`)
- Images are saved as grids using `torchvision.utils.save_image()`
- Latent diffusion requires training autoencoder first
- Encoder is **always frozen** during latent diffusion training
- Use `--debug` flag to enable debug-level logging
