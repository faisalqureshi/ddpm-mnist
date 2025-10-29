#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "../mnist_single_digit" ))
import argparse
from typing import Optional
from ckpt import resolve_ckpt_path, inspect_checkpoint

def main():
    parser = argparse.ArgumentParser("Inspect Checkpoints")
    parser.add_argument("ckpt", type=str, help="Path to a checkpoint file or directory to inspect.")
    args = parser.parse_args()

    candidate = resolve_ckpt_path(Path(args.ckpt))
    if not candidate:
        print(f"Cannot find a checkpoint at: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    inspect_checkpoint(candidate)

if __name__ == "__main__":
    main()
