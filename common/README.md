# Common Utilities

Shared utilities for all DDPM experiments.

## Modules

### `ckpt.py` - Checkpoint Management
- `save_checkpoint()` - Atomically save checkpoint to disk
- `load_checkpoint()` - Load checkpoint and restore model/optimizer state
- `find_latest_checkpoint()` - Find latest checkpoint in a directory
- `find_latest_autoencoder_checkpoint()` - Find latest AE checkpoint by model prefix
- `save_checkpoint_and_link_latest()` - Save checkpoint and create `latest.pt` symlink
- `find_latest_experiment()` - Find most recent experiment directory
- `inspect_checkpoint()` - Inspect checkpoint contents

### `data.py` - Dataset Loaders
- `HF_MNIST` - HuggingFace MNIST dataset wrapper
- `make_mnist_loader()` - Create MNIST DataLoader with proper seeding
- `split_train_val()` - Split dataset into train/val sets

### `device.py` - Device & Seeding Utilities
- `get_device()` - Auto-detect best device (CUDA/MPS/CPU)
- `seed_everything()` - Seed all RNGs for reproducibility
- `seed_worker()` - Worker init function for DataLoader seeding
- `set_seed()` - Set global random seed
- `make_output_directories()` - Create output directory structure

### `logging.py` - Logging Setup
- `setup_component_logger()` - Create logger for a component
- `log_args_rich()` - Pretty-print argparse arguments

### `slurm.py` - SLURM Utilities
- `install_signal_handlers()` - Install SLURM timeout signal handlers
- `stop_requested()` - Check if graceful stop was requested

## Usage

All projects should import from `common.*` instead of duplicating code:

```python
from common.ckpt import load_checkpoint, save_checkpoint_and_link_latest
from common.device import get_device, seed_everything
from common.data import HF_MNIST, make_mnist_loader
from common.logging import setup_component_logger
from common.slurm import install_signal_handlers, stop_requested
```

## Projects Using Common Utilities

- `mnist_single_digit/` - Pixel-space DDPM
- `mnist_ae/` - MLP and Conv autoencoders
- `mnist_latent_diffusion_mlp/` - Latent diffusion with MLP AE
- `mnist_latent_diffusion_conv/` - Latent diffusion with Conv AE

## Migration Notes

Previously, utilities were in `mnist_single_digit/` and projects used `sys.path` hacks to import them. Now all projects use clean imports from the `common/` package.

**Before:**
```python
sys.path.insert(0, str(Path(__file__).parent / "../mnist_single_digit"))
from util import get_device
```

**After:**
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.device import get_device
```
