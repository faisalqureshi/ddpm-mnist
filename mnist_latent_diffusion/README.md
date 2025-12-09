# Unified Latent Diffusion for MNIST

This directory contains a **unified implementation** of latent diffusion that automatically detects and works with both MLP and Convolutional autoencoders.

## Key Feature: Automatic Autoencoder Detection

The training and inference scripts automatically detect which autoencoder type (MLP or Conv) to use based on the checkpoint's metadata. No manual specification needed!

## Architecture

### Latent Space Diffusion
- **Latent dimension**: 16D (compressed from 784D pixel space)
- **Denoiser**: MLP-based with FiLM time conditioning
- **Diffusion steps**: 1000 (T=1000)
- **Schedule**: Linear β ∈ [1e-4, 2e-2]

### Training Pipeline
1. **Encoding**: Pretrained autoencoder encoder (frozen) maps images → 16D latents
2. **Noise**: Forward diffusion adds noise to latents: z₀ → z_t
3. **Denoising**: LatentDenoiser predicts noise from z_t
4. **Loss**: MSE(predicted_noise, actual_noise)

### Generation Pipeline
1. **Sample**: z_T ~ N(0, I) in 16D latent space
2. **Denoise**: Iteratively remove noise over T steps → z₀
3. **Decode**: Pretrained autoencoder decoder (frozen) maps z₀ → image

## Supported Autoencoder Types

### MLP Autoencoder
- Encoder: 784 → 512 → 256 → 128 → 16
- Decoder: 16 → 128 → 256 → 512 → 784
- Fully connected layers with ReLU activation

### Convolutional Autoencoder
- Encoder: Conv layers with 2x striding (28×28 → 14×14 → 7×7 → 4×4) + FC → 16D
- Decoder: FC + upsampling (4×4 → 7×7 → 14×14 → 28×28) + Conv layers
- Uses explicit interpolation to avoid artifacts

## Usage

### Training

The script automatically detects the autoencoder type from the checkpoint:

```bash
# With MLP autoencoder
python train.py \
    --ae-ckpt /path/to/mlp-autoencoder/checkpoint.pt \
    --epochs 30 \
    --batch-size 128 \
    --T 1000 \
    --generate-images

# With Conv autoencoder
python train.py \
    --ae-ckpt /path/to/conv-autoencoder/checkpoint.pt \
    --epochs 30 \
    --batch-size 128 \
    --T 1000 \
    --generate-images
```

**Auto-checkpoint detection** (looks for latest autoencoder):
```bash
python train.py \
    --epochs 30 \
    --batch-size 128 \
    --T 1000 \
    --generate-images
```

The script will automatically find and use the latest autoencoder checkpoint (tries Conv first, then MLP).

### Inference

Generate images using trained latent diffusion model:

```bash
python infer.py \
    --ckpt /path/to/latent-diffusion/checkpoint.pt \
    --ae-ckpt /path/to/autoencoder/checkpoint.pt \
    --num-images 64 \
    --output generated.png
```

The script automatically detects which autoencoder type to use from the `--ae-ckpt` checkpoint.

## Training Arguments

- `--cache-dir` - HuggingFace cache directory (default: `$SLURM_TMPDIR` or `./hf_cache`)
- `--outdir` - Output directory (default: `$SCRATCH` or `./outputs`)
- `--ae-ckpt` - Path to pretrained autoencoder checkpoint
- `--ae-ckpt-dir` - Directory with autoencoder checkpoints (uses latest)
- `--only-digit` - Train on single digit (0-9, or None for all)
- `--epochs` - Number of training epochs (default: 20)
- `--batch-size` - Batch size (default: 128)
- `--lr` - Learning rate (default: 2e-4)
- `--T` - Diffusion timesteps (default: 1000)
- `--beta-start` - Noise schedule start (default: 1e-4)
- `--beta-end` - Noise schedule end (default: 2e-2)
- `--hidden-dim` - Denoiser hidden dimension (default: 256)
- `--samples-per-epoch` - Images to generate per epoch (default: 16)
- `--seed` - Random seed (default: 42)
- `--device` - Force device (cuda/mps/cpu)
- `--amp` - Enable automatic mixed precision
- `--generate-images` - Generate sample images each epoch
- `--auto-resume` - Resume from latest checkpoint
- `--early-stopping-patience` - Early stopping patience (epochs)

## Implementation Details

### Automatic Type Detection

The implementation uses Python's `importlib` to dynamically load the correct autoencoder module:

```python
# Both modules are imported
ae_module_mlp = ...  # from mnist_ae/model.py
ae_module_conv = ... # from mnist_ae/model_conv.py

# Checkpoint is read to detect type
model_type = ckpt["args"]["model"]  # "mlp" or "conv"

# Correct module is selected
if model_type == "conv":
    ae_module = ae_module_conv
elif model_type == "mlp":
    ae_module = ae_module_mlp
```

This pattern is also used in `mnist_ae/run_tsne.py`.

### FiLM Time Conditioning

The LatentDenoiser uses Feature-wise Linear Modulation (FiLM) to incorporate timestep information:

1. Sinusoidal timestep embeddings → TimeMLP → time_emb (256D)
2. FiLM conditioning: `h = (1 + γ) * norm(x) + β` where γ, β = Linear(time_emb)
3. Applied after LayerNorm in the MLP denoiser

### Output Directory Structure

```
$SCRATCH/  (or ./outputs/)
├── checkpoints/
│   └── latent-diffusion-dall-bs128-lr0.0002-T1000-seed42-YYYYMMDD-HHMMSS/
│       ├── checkpoint_epoch_000001.pt
│       └── latest.pt
├── samples/
│   └── latent-diffusion-dall-bs128-lr0.0002-T1000-seed42-YYYYMMDD-HHMMSS/
│       └── epoch_000001.png
├── runs/
│   └── latent-diffusion-dall-bs128-lr0.0002-T1000-seed42-YYYYMMDD-HHMMSS/
└── logs/
    └── latent-diffusion-dall-bs128-lr0.0002-T1000-seed42-YYYYMMDD-HHMMSS.log
```

## Benefits of Unified Implementation

1. **DRY Principle**: Single codebase for both autoencoder types
2. **Automatic**: No manual specification of autoencoder type needed
3. **Maintainable**: One place to fix bugs and add features
4. **Flexible**: Easy to add new autoencoder architectures
5. **Backward Compatible**: Works with existing MLP and Conv checkpoints

## Comparison to Pixel-Space Diffusion

| Aspect | Pixel-Space (mnist_single_digit) | Latent Diffusion (this) |
|--------|----------------------------------|-------------------------|
| **Diffusion Space** | 784D (28×28 pixels) | 16D (latent space) |
| **Denoiser** | U-Net with Conv layers | MLP with FiLM conditioning |
| **Training Speed** | Slower (high-dim) | Faster (low-dim) |
| **Memory** | Higher | Lower |
| **Quality** | Direct pixel generation | Decoder-dependent |
| **Requires** | None | Pretrained autoencoder |

## Notes

- The LatentDenoiser architecture is **identical** regardless of autoencoder type (MLP vs Conv)
- Both autoencoders compress to the same 16D latent space
- The encoder is **always frozen** during latent diffusion training
- Use `--generate-images` to visualize progress during training
