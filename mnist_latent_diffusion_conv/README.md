# MNIST Latent Diffusion (Convolutional Autoencoder)

Latent Diffusion Model for MNIST digit generation using the **Convolutional autoencoder**. Unlike the pixel-space diffusion in `mnist_single_digit`, this implementation operates in the latent space of a pretrained convolutional autoencoder.

**Note**: This is the Conv version. For the MLP autoencoder version, see `mnist_latent_diffusion_mlp/`.

## Architecture

### Latent Space Diffusion
- **Input**: 16-dimensional latent vectors (from autoencoder)
- **Model**: MLP-based denoiser with FiLM conditioning
- **Output**: Denoised latent vectors
- **Decoder**: Pretrained autoencoder decoder converts latents → images

### Key Components

1. **LatentDenoiser** (`model.py`):
   - MLP architecture with 4 residual blocks
   - FiLM conditioning using time embeddings
   - Operates on (B, 16) latent vectors
   - Hidden dimension: 256

2. **Pretrained Autoencoder**:
   - From `../mnist_ae/model_conv.py`
   - Conv Encoder: (1, 28, 28) → (16,) via downsampling (28→14→7→4)
   - Conv Decoder: (16,) → (1, 28, 28) via upsampling (4→7→14→28)
   - Trained with BCE loss

3. **Diffusion Process**:
   - T = 1000 timesteps
   - Linear noise schedule: β ∈ [1e-4, 2e-2]
   - DDPM sampling (not DDIM)

## Advantages over Pixel-Space Diffusion

1. **Efficiency**: 16D latents vs 784D pixels (49x reduction)
2. **Faster training**: Smaller space means faster forward/backward passes
3. **Better scaling**: Can handle larger images by increasing latent dimensions
4. **Semantic structure**: Latent space may capture semantic features better

## Usage

### 1. Train Conv Autoencoder (if not done yet)

```bash
cd ../mnist_ae
python train.py --model conv --epochs 100 --generate-images
```

This will save checkpoints to `$SCRATCH/checkpoints/` (or `./outputs/checkpoints/`).

### 2. Train Latent Diffusion

```bash
cd ../mnist_latent_diffusion_conv
python train.py \
    --ae-ckpt /path/to/conv-autoencoder/checkpoint_epoch_000100.pt \
    --epochs 30 \
    --generate-images \
    --T 1000 \
    --batch-size 128 \
    --lr 2e-4
```

**Key arguments**:
- `--ae-ckpt`: Path to pretrained autoencoder checkpoint (required)
- `--only-digit`: Train on single digit (e.g., `--only-digit 5`)
- `--latent-dim`: Latent dimension (default: 16)
- `--hidden-dim`: Hidden dimension for denoiser MLP (default: 256)
- `--T`: Number of diffusion timesteps (default: 1000)
- `--generate-images`: Generate samples each epoch

### 3. Generate Images

```bash
python infer.py \
    --ckpt /path/to/diffusion/checkpoint_epoch_000030.pt \
    --ae-ckpt /path/to/autoencoder/checkpoint_epoch_000025.pt \
    --num-images 64 \
    --output generated_digits.png
```

## Training Process

```
MNIST Images (28×28)
    ↓
Encoder: images → latents (16D)
    ↓
Add noise: z_0 → z_t
    ↓
Denoiser: predict ε_hat from z_t and t
    ↓
Loss: MSE(ε_hat, ε)
    ↓
Backprop (encoder frozen)
```

## Generation Process

```
Random noise z_T ~ N(0, I)  (16D)
    ↓
Iterative denoising (T steps)
    ↓
Clean latent z_0  (16D)
    ↓
Decoder: z_0 → image (28×28)
```

## Model Details

### LatentDenoiser Architecture
```
Input: (B, 16) latent + timestep t
↓
Time Embedding: t → (B, 256)
↓
Input Projection: (B, 16) → (B, 256)
↓
4 Residual Blocks (each):
  - FiLM Conditioning (2 layers)
  - Residual connection
↓
Output Projection: (B, 256) → (B, 16)
↓
Output: predicted noise ε (B, 16)
```

### FiLM Conditioning
```
LayerNorm(h)
↓
γ, β = Linear(time_emb)
↓
h = (1 + γ) * h + β
↓
h = SiLU(h)
↓
h = Linear(h)
```

## Comparison with Pixel-Space Diffusion

| Feature | Pixel-Space | Latent-Space |
|---------|-------------|--------------|
| Input dimension | 784 (28×28) | 16 |
| Architecture | U-Net (Conv) | MLP |
| Parameters | ~150K | ~50K |
| Training speed | Slower | Faster |
| Memory usage | Higher | Lower |
| Image quality | Direct | Via decoder |

## Files

- `model.py`: Latent diffusion model, schedules, sampling
- `train.py`: Training script
- `infer.py`: Inference script for generation
- `README.md`: This file

## Requirements

Same as parent project:
- PyTorch >= 2.0
- torchvision
- datasets (HuggingFace)
- tensorboard

## Notes

- The autoencoder must be pretrained before training the diffusion model
- Encoder is frozen during diffusion training
- Only the diffusion model parameters are optimized
- Generated images inherit any reconstruction artifacts from the autoencoder
