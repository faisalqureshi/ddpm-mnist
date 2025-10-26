#
# util.py
#

import os, random
from typing import Optional
import torch
import numpy as np
from torchvision import utils as tvutils
from pathlib import Path

# def set_seed(seed: int):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

import os
import torch
from typing import Optional

def get_device(logger, device_arg: Optional[str] = None) -> torch.device:
    logger.info("=== Device availability ===")

    # ---- Probe availability ----
    cuda_ok = torch.cuda.is_available()

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend and hasattr(mps_backend, "is_built") and mps_backend.is_built())
    mps_avail = bool(mps_backend and hasattr(mps_backend, "is_available") and mps_backend.is_available())

    # Optional: runtime smoke test (guarded)
    mps_runtime_ok = False
    if mps_built and mps_avail:
        try:
            _ = torch.ones(1, device="mps") * 1  # simple op to ensure runtime works
            mps_runtime_ok = True
        except Exception as e:
            logger.info(f"- [MPS] Runtime test failed: {e}")

    logger.info(f"- PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda}")
    if cuda_ok:
        n = torch.cuda.device_count()
        logger.info(f"- CUDA GPUs visible: {n} (CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', 'unset')})")
        for i in range(n):
            logger.info(f"-   - cuda:{i} -> {torch.cuda.get_device_name(i)}")
        try:
            logger.info(f"- cuDNN: {torch.backends.cudnn.version()}")
        except Exception:
            pass
    else:
        logger.info("- CUDA GPUs visible: 0")

    logger.info(f"- MPS (Apple Silicon): built={mps_built}, available={mps_avail}, runtime_ok={mps_runtime_ok}")
    logger.info("- CPU: available")

    # ---- Choose device (respect user arg with validation) ----
    chosen = None
    if device_arg:
        try:
            req = torch.device(device_arg)
            if req.type == "cuda":
                if not cuda_ok:
                    logger.info(f"- [get_device] Requested '{device_arg}' but CUDA not available -> using CPU.")
                    chosen = torch.device("cpu")
                else:
                    idx = req.index if req.index is not None else 0
                    if not (0 <= idx < torch.cuda.device_count()):
                        logger.info(f"- [get_device] Requested CUDA index {idx} out of range -> using cuda:0.")
                        chosen = torch.device("cuda:0")
                    else:
                        chosen = torch.device(f"- cuda:{idx}")
            elif req.type == "mps":
                if not (mps_built and mps_avail and mps_runtime_ok):
                    logger.info(f"- [get_device] Requested 'mps' but not fully usable -> using CPU.")
                    chosen = torch.device("cpu")
                else:
                    chosen = req
            else:
                chosen = req  # e.g., "cpu"
        except Exception as e:
            logger.info(f"- [get_device] Invalid device '{device_arg}': {e} -> falling back to auto-select.")

    if chosen is None:
        if cuda_ok:
            chosen = torch.device("cuda:0")
        elif mps_built and mps_avail and mps_runtime_ok:
            chosen = torch.device("mps")
        else:
            chosen = torch.device("cpu")

    logger.info(f"- Chosen device: {chosen}")
    if chosen.type == "cuda":
        idx = chosen.index if chosen.index is not None else torch.cuda.current_device()
        logger.info(f"- Using cuda:{idx} -> {torch.cuda.get_device_name(idx)}")
    return chosen

def make_image_grid(imgs, nrow=4):
    """
    imgs: (N, C, H, W) in [0,1] or [-1,1]
    """
    x = imgs
    if x.min() < 0:          # map from [-1,1] -> [0,1] if needed
        x = (x.clamp(-1,1) + 1) / 2

    grid = tvutils.make_grid(x, nrow=nrow, padding=2)       # (C, H*, W*)
    grid_np = grid.permute(1,2,0).cpu().numpy()     # H* x W* x C
    return grid_np

def show_samples_inline(grid_np, title=""):
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        from IPython.display import clear_output, display
        in_ipy = True
    except Exception:
        in_ipy = False
        clear_output = None
        display = None

    plt.figure(figsize=(4, 4))

    arr = np.asarray(grid_np)
    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.shape[-1] == 1:  # grayscale
        plt.imshow(arr[..., 0], cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(arr[..., :3], vmin=0, vmax=1)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if in_ipy and clear_output is not None:
        clear_output(wait=True)
        display(plt.gcf())
        plt.close()
    else:
        # Fallback for non-notebook runs
        plt.show()
        plt.close()

def seed_everything(seed: int, deterministic: bool = False):
    # If util.set_seed already seeds some of these, calling again is harmless.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # For strict determinism; slower but reproducible.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # For CUDA matmul determinism (optional):
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    return torch.Generator(device="cpu").manual_seed(seed)

def seed_worker(worker_id):
    # Ensures each dataloader worker has a distinct but reproducible RNG state.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_output_directories(outdir, exp_name):
    outdir = Path(outdir)    
    log_dir = outdir / "logs" / exp_name 
    ckpt_dir = outdir / "checkpoints" / exp_name
    sample_dir = outdir / "samples" / exp_name
    run_dir = outdir / "runs" / exp_name
    for d in (ckpt_dir, sample_dir, run_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, sample_dir, run_dir, log_dir
