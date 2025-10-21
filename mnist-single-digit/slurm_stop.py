# slurm_stop.py
import os, signal, threading, logging

_stop_event = threading.Event()
STOP_FLAG_FILE = ".stop_soon"  # created by sbatch trap

def _sig_handler(signum, frame):
    logging.warning(f"Received signal {signum}; will stop soon.")
    _stop_event.set()

def install_signal_handlers():
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _sig_handler)
        except Exception:
            pass

def stop_requested(outdir: str) -> bool:
    # either a signal was received, or the file flag exists
    if _stop_event.is_set():
        return True
    try:
        return os.path.exists(os.path.join(outdir, STOP_FLAG_FILE))
    except Exception:
        return _stop_event.is_set()
