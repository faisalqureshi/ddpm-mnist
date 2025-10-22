#
# util.py
#

import os, random
from typing import Optional
import torch
import numpy as np
from torchvision import utils as tvutils

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import os
import torch
from typing import Optional

def get_device(device_arg: Optional[str] = None) -> torch.device:
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
            print(f"[MPS] Runtime test failed: {e}")

    print("=== Device availability ===")
    print(f"PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda}")
    if cuda_ok:
        n = torch.cuda.device_count()
        print(f"CUDA GPUs visible: {n} (CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', 'unset')})")
        for i in range(n):
            print(f"  - cuda:{i} -> {torch.cuda.get_device_name(i)}")
        try:
            print(f"cuDNN: {torch.backends.cudnn.version()}")
        except Exception:
            pass
    else:
        print("CUDA GPUs visible: 0")

    print(f"MPS (Apple Silicon): built={mps_built}, available={mps_avail}, runtime_ok={mps_runtime_ok}")
    print("CPU: available")

    # ---- Choose device (respect user arg with validation) ----
    chosen = None
    if device_arg:
        try:
            req = torch.device(device_arg)
            if req.type == "cuda":
                if not cuda_ok:
                    print(f"[get_device] Requested '{device_arg}' but CUDA not available -> using CPU.")
                    chosen = torch.device("cpu")
                else:
                    idx = req.index if req.index is not None else 0
                    if not (0 <= idx < torch.cuda.device_count()):
                        print(f"[get_device] Requested CUDA index {idx} out of range -> using cuda:0.")
                        chosen = torch.device("cuda:0")
                    else:
                        chosen = torch.device(f"cuda:{idx}")
            elif req.type == "mps":
                if not (mps_built and mps_avail and mps_runtime_ok):
                    print(f"[get_device] Requested 'mps' but not fully usable -> using CPU.")
                    chosen = torch.device("cpu")
                else:
                    chosen = req
            else:
                chosen = req  # e.g., "cpu"
        except Exception as e:
            print(f"[get_device] Invalid device '{device_arg}': {e} -> falling back to auto-select.")

    if chosen is None:
        if cuda_ok:
            chosen = torch.device("cuda:0")
        elif mps_built and mps_avail and mps_runtime_ok:
            chosen = torch.device("mps")
        else:
            chosen = torch.device("cpu")

    print(f"=== Chosen device: {chosen} ===")
    if chosen.type == "cuda":
        idx = chosen.index if chosen.index is not None else torch.cuda.current_device()
        print(f"Using cuda:{idx} -> {torch.cuda.get_device_name(idx)}")
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




# def make_mnist_loader(data_root: str, batch_size: int, num_workers: int, only_digit: Optional[int]):
#     tfm = tvtf.ToTensor()
#     ds = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
#     if only_digit is not None:
#         idx = (ds.targets == only_digit).nonzero(as_tuple=True)[0]
#         ds = Subset(ds, idx)
#     loader = DataLoader(
#         ds,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=False,
#     )
#     return loader, len(ds)
