import os
import sys
import torch
import logging
import torch.distributed as dist

from typing import Any, Callable, Optional, Union


def get_rank() -> int:
    """Determine the rank of the current process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    
    # Fallback to environment variables
    for env_var in ["RANK", "LOCAL_RANK", "SLURM_PROCID"]:
        if env_var in os.environ:
            try:
                return int(os.environ[env_var])
            except (ValueError, TypeError):
                continue
    
    return 0


def rank_zero_only(fn: Callable) -> Callable:
    """Decorator to only run a function on rank 0."""
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return fn(*args, **kwargs)
        return None
    return wrapper


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    run_name: Optional[str] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging for torch_pharma experiments and libraries."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return get_pylogger("torch_pharma")

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    resolved_log_file: Optional[str] = None
    if run_name is not None:
        from torch_pharma.paths import resolve_log_file

        resolved_log_file = str(resolve_log_file(log_file, run_name))

    root = logging.getLogger()
    if force:
        root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    if resolved_log_file and not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == os.path.abspath(resolved_log_file)
        for h in root.handlers
    ):
        file_handler = logging.FileHandler(resolved_log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(level)
    _CONFIGURED = True
    logger = get_pylogger("torch_pharma")
    if resolved_log_file:
        logger.info("Logging to %s", resolved_log_file)
    return logger


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def rank_zero_debug(*args: Any, **kwargs: Any) -> None:
    """Log a debug-level message only on rank 0 using print."""
    if get_rank() == 0:
        print("[DEBUG]", *args, **kwargs)


def rank_zero_info(*args: Any, **kwargs: Any) -> None:
    """Log an info-level message only on rank 0 using print."""
    if get_rank() == 0:
        print("[INFO]", *args, **kwargs)


def rank_zero_warn(*args: Any, **kwargs: Any) -> None:
    """Log a warning-level message only on rank 0 using print."""
    if get_rank() == 0:
        print("[WARNING]", *args, **kwargs)
