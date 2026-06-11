"""Base task protocol for torch_pharma Trainer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


class BaseTask(ABC, nn.Module):
    """Task encapsulates model, optimizers, and train/val step logic."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.device: torch.device = torch.device("cpu")

    def setup(self, device: torch.device) -> None:
        self.device = device
        if self.model is None:
            log.info("Configuring model for %s", self.__class__.__name__)
            self.model = self.configure_model()
        self.model.to(device)
        if self.optimizer is None:
            opt, sched = self.configure_optimizers()
            self.optimizer = opt
            self.scheduler = sched
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(
            "Task %s ready on %s: %d parameters (%d trainable)",
            self.__class__.__name__,
            device,
            n_params,
            n_trainable,
        )

    @abstractmethod
    def configure_model(self) -> nn.Module:
        ...

    @abstractmethod
    def configure_optimizers(self):
        ...

    @abstractmethod
    def training_step(self, batch) -> Dict[str, torch.Tensor]:
        ...

    def validation_step(self, batch) -> Dict[str, torch.Tensor]:
        return self.training_step(batch)

    def predict_step(self, batch):
        raise NotImplementedError

    def transfer_batch_to_device(self, batch, device: torch.device):
        if isinstance(batch, (list, tuple)):
            return tuple(self.transfer_batch_to_device(b, device) for b in batch)
        if hasattr(batch, "to"):
            return batch.to(device)
        return batch

    def scheduler_step(self, global_step: int) -> None:
        if self.scheduler is None:
            return
        if hasattr(self.scheduler, "step") and hasattr(self.scheduler, "get_lr"):
            self.scheduler.step(global_step)
        elif hasattr(self.scheduler, "step"):
            self.scheduler.step()

    def on_train_epoch_start(self, trainer) -> None:
        pass

    def on_train_epoch_end(self, trainer) -> None:
        pass

    def on_validation_epoch_start(self, trainer) -> None:
        pass

    def on_validation_epoch_end(self, trainer) -> None:
        pass

    def set_trainable_params(self, param_patterns, delta_train: bool = False) -> None:
        """Freeze all params except those matching patterns (NExT-Mol delta training)."""
        self._original_requires_grad = {}
        for name, param in self.named_parameters():
            self._original_requires_grad[name] = param.requires_grad
            match = any(p in name for p in param_patterns)
            param.requires_grad = match if delta_train else self._original_requires_grad[name]

    def restore_trainable_params(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = self._original_requires_grad.get(name, True)


class Task(BaseTask):
    """Backward-compatible alias; subclasses must implement run() or step methods."""

    @abstractmethod
    def run(self, model, data_loader):
        pass
