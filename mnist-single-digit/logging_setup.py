# logging_setup.py
import logging, os, sys
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir, filename="train.log"):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear duplicate handlers if re-imported on resume
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)  # goes to Slurm .out
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    root.addHandler(sh)

    # Make prints flush immediately too
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    logging.info(f"Logging to {log_path}")
