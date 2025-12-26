#!/usr/bin/env python
"""
Find the best checkpoint from a directory based on specified criteria.

This script scans all checkpoints in a directory and identifies the best one
based on training or validation loss, epoch number, or other metrics.

Usage:
    python find_best_ckpt.py /path/to/checkpoints --metric val_loss --mode min
    python find_best_ckpt.py /path/to/checkpoints --metric train_loss --mode min
    python find_best_ckpt.py /path/to/checkpoints --metric epoch --mode max
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict


def load_checkpoint_metadata(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint and extract metadata without loading the full model state.

    Returns:
        Dictionary with checkpoint metadata, or None if load fails
    """
    try:
        # Load only metadata, not full model weights
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Extract common metadata fields
        metadata = {
            "path": ckpt_path,
            "epoch": ckpt.get("epoch", None),
            "global_step": ckpt.get("global_step", None),
            "exp_name": ckpt.get("exp_name", None),
            "timestamp": ckpt.get("time", None),
        }

        # Extract args if available
        if "args" in ckpt:
            args = ckpt["args"]
            if isinstance(args, dict):
                metadata["args"] = args
            else:
                # argparse.Namespace
                metadata["args"] = vars(args)

        return metadata
    except Exception as e:
        print(f"Warning: Failed to load {ckpt_path}: {e}", file=sys.stderr)
        return None


def extract_loss_from_log(log_path: Path, checkpoint_epoch: int) -> Optional[Tuple[float, float]]:
    """
    Extract train and val loss for a specific epoch from log file.

    Returns:
        Tuple of (train_loss, val_loss) or (None, None) if not found
    """
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Search for the epoch line
        # Expected format: [Epoch 123/200] train_loss=0.123 val_loss=0.456 ...
        for line in lines:
            if f"[Epoch {checkpoint_epoch}/" in line or f"Epoch {checkpoint_epoch}/" in line:
                # Extract losses using string parsing
                train_loss = None
                val_loss = None

                if "train_loss=" in line:
                    start = line.index("train_loss=") + len("train_loss=")
                    end = line.index(" ", start) if " " in line[start:] else len(line)
                    train_loss = float(line[start:end])

                if "val_loss=" in line:
                    start = line.index("val_loss=") + len("val_loss=")
                    end = line.index(" ", start) if " " in line[start:] else len(line)
                    val_loss = float(line[start:end])

                return train_loss, val_loss

        return None, None
    except Exception as e:
        print(f"Warning: Failed to extract loss from log: {e}", file=sys.stderr)
        return None, None


def find_log_file(ckpt_dir: Path, exp_name: str) -> Optional[Path]:
    """
    Find the log file corresponding to an experiment.

    Searches in common log locations:
    - ../logs/{exp_name}.log
    - ../../logs/{exp_name}.log
    """
    # Try relative to checkpoint directory
    candidates = [
        ckpt_dir.parent.parent / "logs" / f"{exp_name}.log",
        ckpt_dir.parent.parent.parent / "logs" / f"{exp_name}.log",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def find_best_checkpoint(
    ckpt_dir: Path,
    metric: str = "val_loss",
    mode: str = "min",
    use_logs: bool = True,
    verbose: bool = False
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """
    Find the best checkpoint in a directory based on specified metric.

    Args:
        ckpt_dir: Directory containing checkpoint files
        metric: Metric to optimize ('val_loss', 'train_loss', 'epoch', 'global_step')
        mode: 'min' or 'max' (minimize or maximize the metric)
        use_logs: Whether to try extracting losses from log files
        verbose: Print progress information

    Returns:
        Tuple of (best_checkpoint_path, metadata_dict) or None if no checkpoints found
    """
    if not ckpt_dir.exists():
        print(f"Error: Directory does not exist: {ckpt_dir}", file=sys.stderr)
        return None

    # Find all checkpoint files
    ckpt_files = list(ckpt_dir.glob("*.pt"))
    if not ckpt_files:
        print(f"Error: No checkpoint files found in {ckpt_dir}", file=sys.stderr)
        return None

    if verbose:
        print(f"Found {len(ckpt_files)} checkpoint files")

    # Load metadata for all checkpoints
    checkpoints = []
    for ckpt_path in ckpt_files:
        metadata = load_checkpoint_metadata(ckpt_path)
        if metadata is None:
            continue

        # Try to get loss from logs if requested
        if use_logs and metric in ["val_loss", "train_loss"]:
            exp_name = metadata.get("exp_name")
            epoch = metadata.get("epoch")

            if exp_name and epoch:
                log_file = find_log_file(ckpt_dir, exp_name)
                if log_file:
                    train_loss, val_loss = extract_loss_from_log(log_file, epoch)
                    if train_loss is not None:
                        metadata["train_loss"] = train_loss
                    if val_loss is not None:
                        metadata["val_loss"] = val_loss

        checkpoints.append(metadata)

    if not checkpoints:
        print("Error: No valid checkpoints could be loaded", file=sys.stderr)
        return None

    if verbose:
        print(f"Successfully loaded {len(checkpoints)} checkpoints")

    # Filter checkpoints that have the requested metric
    valid_checkpoints = [ckpt for ckpt in checkpoints if metric in ckpt and ckpt[metric] is not None]

    if not valid_checkpoints:
        print(f"Error: No checkpoints have the metric '{metric}'", file=sys.stderr)
        print(f"Available metrics in checkpoints: {list(checkpoints[0].keys())}", file=sys.stderr)
        return None

    # Find best checkpoint
    if mode == "min":
        best_ckpt = min(valid_checkpoints, key=lambda x: x[metric])
    elif mode == "max":
        best_ckpt = max(valid_checkpoints, key=lambda x: x[metric])
    else:
        print(f"Error: Invalid mode '{mode}'. Must be 'min' or 'max'", file=sys.stderr)
        return None

    return best_ckpt["path"], best_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Find the best checkpoint based on specified criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find checkpoint with lowest validation loss
  python find_best_ckpt.py /path/to/checkpoints --metric val_loss --mode min

  # Find checkpoint with highest epoch number (latest)
  python find_best_ckpt.py /path/to/checkpoints --metric epoch --mode max

  # Find checkpoint with lowest training loss
  python find_best_ckpt.py /path/to/checkpoints --metric train_loss --mode min

  # Copy best checkpoint to a specific location
  python find_best_ckpt.py /path/to/checkpoints --metric val_loss --copy-to best_model.pt
        """
    )

    parser.add_argument("checkpoint_dir", type=str, help="Directory containing checkpoint files")
    parser.add_argument("--metric", type=str, default="val_loss",
                       help="Metric to optimize (val_loss, train_loss, epoch, global_step)")
    parser.add_argument("--mode", type=str, default="min", choices=["min", "max"],
                       help="Minimize or maximize the metric")
    parser.add_argument("--no-logs", action="store_true",
                       help="Don't try to extract losses from log files")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print verbose output")
    parser.add_argument("--copy-to", type=str, default=None,
                       help="Copy the best checkpoint to this path")
    parser.add_argument("--symlink-to", type=str, default=None,
                       help="Create a symlink to the best checkpoint at this path")
    parser.add_argument("--print-path-only", action="store_true",
                       help="Only print the path to the best checkpoint (for scripting)")

    args = parser.parse_args()

    # Find best checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    result = find_best_checkpoint(
        ckpt_dir,
        metric=args.metric,
        mode=args.mode,
        use_logs=not args.no_logs,
        verbose=args.verbose
    )

    if result is None:
        sys.exit(1)

    best_path, best_metadata = result

    # Print results
    if args.print_path_only:
        print(best_path)
    else:
        print(f"\n{'='*60}")
        print(f"Best checkpoint based on {args.metric} ({args.mode})")
        print(f"{'='*60}")
        print(f"Path: {best_path}")
        print(f"Epoch: {best_metadata.get('epoch', 'N/A')}")
        print(f"Global Step: {best_metadata.get('global_step', 'N/A')}")

        if "train_loss" in best_metadata:
            print(f"Train Loss: {best_metadata['train_loss']:.6f}")
        if "val_loss" in best_metadata:
            print(f"Val Loss: {best_metadata['val_loss']:.6f}")

        print(f"\nBest {args.metric}: {best_metadata[args.metric]}")
        print(f"{'='*60}\n")

    # Copy or symlink if requested
    if args.copy_to:
        import shutil
        dest = Path(args.copy_to)
        shutil.copy2(best_path, dest)
        print(f"Copied best checkpoint to: {dest}")

    if args.symlink_to:
        dest = Path(args.symlink_to)
        if dest.exists():
            dest.unlink()
        dest.symlink_to(best_path.absolute())
        print(f"Created symlink to best checkpoint: {dest} -> {best_path}")


if __name__ == "__main__":
    main()
