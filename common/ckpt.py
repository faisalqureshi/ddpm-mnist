# ckpt.py
import os, tempfile, logging, torch, time, sys
from pathlib import Path
from typing import Optional
import re
from datetime import datetime
from . import emoji
from . import error_codes

def resolve_resume_path(args):
    if args.resume:
        ckpt_path = Path(args.resume)
        print(f"{emoji.step} Inspecting checkpoint: {ckpt_path}") 
        ckpt_model = infer_model_from_checkpoint(ckpt_path)
        if args.model and args.model != ckpt_model:
            print(f"{emoji.error} Model mismatch.  Exiting.")
            exit(error_codes.MODEL_MISMATCH)
        else:
            print(f"{emoji.ok} Extracted model: {ckpt_model}")
            args.model = ckpt_model
        if args.auto_resume:
            print(f"{emoji.warning} Ignored auto-resume") 
        return ckpt_path.parent.name, ckpt_path, ckpt_model
    elif args.auto_resume:
        if not args.model:
            print(f"{emoji.error} Model must be specified with auto-resume.  Exiting.")
            exit(error_codes.NO_MODEL_SPECIFIED)
        else:
            return find_latest_checkpoint(Path(args.outdir), args.model)
    else:
        return None, None, None

def infer_model_from_checkpoint(ckpt_path: Path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except:
        print(f"{emoji.error} Error loading checkpoint.  Exiting")
        exit(-1)    
    try:
        if isinstance(ckpt["args"], dict):
            ckpt_model = ckpt["args"]["model"]
        else:
            ckpt_model = ckpt["args"].model
    except:
        print(f"{emoji.error} Error extracting model from ckpt.  Exiting.")
        exit(-1)
    return ckpt_model

def find_latest_checkpoint(outdir: Path, model):
    ckpt_base = outdir / "checkpoints"
    print(f"{emoji.step} Looking for checkpoints in folder: {ckpt_base}")
    if ckpt_base.exists():
        matching_dirs = sorted(
            ckpt_base.glob(model+"*"),
            key = lambda x: x.stat().st_mtime,
            reverse = True
        )
        if matching_dirs:
            print(f"{emoji.ok} Found {len(matching_dirs)} matching folders")
            latest_exp_dir = matching_dirs[0]
            print(f"{emoji.step} Looking for checkpoints in {latest_exp_dir}")
            ckpts = sorted(
                latest_exp_dir.glob("*.pt"),
                key = lambda x: x.stat().st_mtime,
                reverse = True
            )
            resume_ckpt_path = ckpts[0] if ckpts else None
            if resume_ckpt_path:
                print(f"{emoji.ok} Found checkpoint {resume_ckpt_path}")
                ckpt_model = infer_model_from_checkpoint(resume_ckpt_path)
                if model != ckpt_model:
                    print(f"{emoji.error} Model mismatch.  Exiting.")
                    exit(error_codes.MODEL_MISMATCH)
                else:
                    return  latest_exp_dir.name, resume_ckpt_path, ckpt_model
            else:
                print(f"{emoji.warning} No checkpoint found")
        else:
            print(f"{emoji.warning} No matching folders found")
    else:
        print(f"{emoji.warning} Folder not found")

    return None, None, None

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
                print(f"- args: {type(obj['args']).__name__}")
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

def load_checkpoint(path: Path, model=None, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model" in ckpt and ckpt["model"]:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0), ckpt.get("args", None)

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
