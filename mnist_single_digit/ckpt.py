# ckpt.py
import os, tempfile, logging, torch, time
from pathlib import Path
from typing import Optional

def save_checkpoint(path: Path, model, optimizer, scaler, epoch, global_step, args):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
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

def load_checkpoint(path: Path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
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

def save_checkpoint_and_link_latest(ckpt_dir: Path, model, optimizer, scaler, epoch, global_step, args):
    ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:06d}.pt"
    latest = ckpt_dir / "latest.pt"

    save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, global_step, args)
    try:
        if latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_path.name)  # symlink inside same dir
    except Exception:
        # if symlinks not allowed, copy metadata
        torch.save(torch.load(ckpt_path, map_location="cpu"), latest)

    return ckpt_path
