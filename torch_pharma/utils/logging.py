import os
import torch
import logging
import torch.distributed as dist

from typing import Any, Callable, Optional


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
