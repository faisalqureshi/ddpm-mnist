import sys, logging, io
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.pretty import Pretty

class SkipIfFileOnly(logging.Filter):
    def filter(self, record):
        # drop from this handler if record.file_only is True
        return not getattr(record, "file_only", False)

def make_fmt_string(fmt, level):
    if level == logging.INFO:
        return fmt
    elif level == logging.DEBUG:
        return ("[DEBUG] " + fmt)
    elif level == logging.WARNING:
        return ("[WARNING] " + fmt)
    elif level == logging.ERROR:
        return ("[ERROR] " + fmt)
    elif level == logging.CRITICAL:
        return ("[CRITICAL] " + fmt)

    return fmt

def make_file_handler(path, level=logging.INFO, fmt="%(asctime)s %(levelname)s %(name)s: %(message)s"):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8", delay=True)
    fmt = make_fmt_string(fmt, level)
    h.setFormatter(logging.Formatter(fmt))
    h.setLevel(level)
    return h

def make_stream_handler(stream=sys.stdout, level=logging.INFO, fmt="%(message)s"):
    h = logging.StreamHandler(stream)
    fmt = make_fmt_string(fmt, level)
    h.setFormatter(logging.Formatter(fmt))
    h.setLevel(level)
    h.addFilter(SkipIfFileOnly())
    return h

def setup_console_root(level=logging.INFO):
    # Keep root minimal: console only
    root = logging.getLogger()
    root.setLevel(level)
    # avoid duplicate handlers in notebooks / re-runs
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(make_stream_handler(level=level))

def setup_component_logger(name, logfile, level=logging.INFO, also_console=True):
    """
    Create/return a named logger that logs to its own file (and optionally console).
    It will NOT propagate to root (so files don't get duplicate lines).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # <-- key to prevent double logging via root

    # clear existing handlers if re-running
    logger.handlers.clear()

    logger.addHandler(make_file_handler(logfile, level=level))
    if also_console:
        logger.addHandler(make_stream_handler(level=level))
    return logger

def log_args_rich(logger, args, file_only=False):
    logger.info("Commandline args:", extra={"file_only": file_only})
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    console.print(Pretty(vars(args), expand_all=True))
    logger.info("%s", buf.getvalue(), extra={"file_only": file_only})