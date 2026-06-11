"""Checkpoint save/load utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


def save_checkpoint(
    task,
    filepath: str,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "metrics": metrics or {},
        "metadata": metadata or {},
    }
    if hasattr(task, "state_dict"):
        checkpoint["task_state_dict"] = task.state_dict()
    if hasattr(task, "model") and task.model is not None:
        checkpoint["model_state_dict"] = task.model.state_dict()
    if hasattr(task, "optimizer") and task.optimizer is not None:
        checkpoint["optimizer_state_dict"] = task.optimizer.state_dict()
    torch.save(checkpoint, filepath)
    log.info("Wrote checkpoint to %s (epoch=%d)", filepath, epoch)


def load_checkpoint(task, filepath: str, strict: bool = True):
    log.info("Loading checkpoint from %s", filepath)
    checkpoint = torch.load(filepath, map_location="cpu")
    if hasattr(task, "load_state_dict") and "task_state_dict" in checkpoint:
        task.load_state_dict(checkpoint["task_state_dict"], strict=strict)
    elif hasattr(task, "model") and "model_state_dict" in checkpoint:
        task.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if hasattr(task, "optimizer") and task.optimizer and "optimizer_state_dict" in checkpoint:
        task.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return task, checkpoint
