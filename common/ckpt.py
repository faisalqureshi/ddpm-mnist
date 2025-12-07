# ckpt.py
import os, tempfile, logging, torch, time, sys
from pathlib import Path
from typing import Optional
import re
from datetime import datetime

def save_checkpoint(path: Path, model, optimizer, scaler, epoch, global_step, args, exp_name):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "exp_name": exp_name,
        "time": time.time()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_ckpt_", suffix=".pt")
    os.close(fd) 
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

# def save_checkpoint_atomic(state: dict, path: str):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     d = os.path.dirname(path)
#     fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_ckpt_", suffix=".pt")
#     os.close(fd)
#     try:
#         torch.save(state, tmp)
#         os.replace(tmp, path)  # atomic on POSIX
#         logging.info(f"Saved checkpoint: {path}")
#     finally:
#         if os.path.exists(tmp):
#             try: os.remove(tmp)
#             except Exception: pass

def resolve_ckpt_path(p: Path) -> Optional[Path]:
    """If p is a dir, try p/'latest.pt' then newest *.pt. If p is a file, return it."""
    if p.is_file():
        return p
    if p.is_dir():
        latest = p / "latest.pt"
        if latest.is_file():
            return latest
        pts = sorted(p.glob("*.pt"), key=lambda q: q.stat().st_mtime, reverse=True)
        return pts[0] if pts else None
    return None

def inspect_checkpoint(ckpt_path: Path) -> None:
    """Minimal checkpoint inspection (PyTorch .pt)."""
    try:
        import torch
    except ImportError:
        print("PyTorch not installed; cannot read checkpoint.", file=sys.stderr)
        sys.exit(2)

    try:
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load {ckpt_path}: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"\n=== {ckpt_path} ===")
    if isinstance(obj, dict):
        print("Top-level keys:", list(obj.keys()))
        # Common fields people store
        for k in ("epoch", "global_step"):
            if k in obj:
                print(f"- {k}: {obj[k]}")
        if "args" in obj:
            args = obj["args"]
            if isinstance(args, dict):
                print("- args")
                for k in args:
                    print(f"  - {k}: {args[k]}")
            else:
                print(f"- args: {type(obj["args"]).__name__}")
        for k in ("meta", "amp_scaler_state"):
            if k in obj:
                print(f"- {k}: {type(obj[k]).__name__}")
        # Model params summary
        state = obj.get("state_dict") or obj.get("model") or obj.get("net")
        if isinstance(state, dict):
            n_params = sum(v.numel() for v in state.values())
            print(f"- state_dict: {len(state)} tensors, total params: {n_params}")
            # show a couple of names
            names = list(state.keys())[:5]
            if names:
                print(f"  sample keys: {names}")
    else:
        print(f"Object type: {type(obj).__name__}")

def load_checkpoint(path: Path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0), ckpt.get("args", None)

def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    return cands[-1] if cands else None

PAT = re.compile(r"^(?P<model>[^-]+).*?(?P<ts>\d{8}-\d{6})$")
def find_latest_experiment(outdir: Path, model) -> Optional[Path]:
    ckpt_dir = outdir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    paths = sorted(ckpt_dir.glob("*"))

    def _parse_entry(p: Path):
        m = PAT.match(p.name)
        if not (p.is_dir() and m):
            return None
        try:
            ts = datetime.strptime(m["ts"], "%Y%m%d-%H%M%S")
        except ValueError:
            return None
        return m["model"], ts, p

    rows = [_parse_entry(p) for p in paths]
    rows = [r for r in rows if r and r[0] == model]   # filter by model
    if not rows:
        return None

    return max(rows, key=lambda r: r[1])[2]

def find_latest_autoencoder_checkpoint(outdir: Path, model_prefix: str) -> Optional[Path]:
    """
    Find the latest autoencoder checkpoint for a given model prefix (e.g., 'mlp', 'conv').

    Searches in outdir/checkpoints/ for directories matching {model_prefix}-*,
    then returns the latest checkpoint from the most recently modified directory.

    Args:
        outdir: Base output directory (e.g., $SCRATCH or ./outputs)
        model_prefix: Model prefix to search for (e.g., 'mlp', 'conv')

    Returns:
        Path to latest checkpoint, or None if not found
    """
    ckpt_base = Path(outdir) / "checkpoints"
    if not ckpt_base.exists():
        return None

    # Find all directories matching the prefix pattern
    pattern = f"{model_prefix}-*"
    matching_dirs = sorted(ckpt_base.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)

    if not matching_dirs:
        return None

    # Get latest checkpoint from the most recent directory
    return find_latest_checkpoint(matching_dirs[0])

def save_checkpoint_and_link_latest(ckpt_dir: Path, model, optimizer, scaler, epoch, global_step, args, exp_name):
    ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:06d}.pt"
    latest = ckpt_dir / "latest.pt"

    save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, global_step, args, exp_name)
    try:
        if latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_path.name)  # symlink inside same dir
    except Exception:
        # if symlinks not allowed, copy metadata
        torch.save(torch.load(ckpt_path, map_location="cpu"), latest)

    return ckpt_path
