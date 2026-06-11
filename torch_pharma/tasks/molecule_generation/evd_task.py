"""EVD molecule generation task — proof-of-concept for unified Trainer."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from torch_pharma.tasks.base import BaseTask
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


class EVDMoleculeGenerationTask(BaseTask):
    """Wraps EquivariantVariationalDiffusion for Trainer.fit() integration."""

    def __init__(self, evd_model: nn.Module, lr: float = 1e-4):
        super().__init__()
        self._evd = evd_model
        self.lr = lr

    def configure_model(self) -> nn.Module:
        log.info("Using pre-built EquivariantVariationalDiffusion model (lr=%g)", self.lr)
        return self._evd

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, None]:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr), None

    def training_step(self, batch) -> Dict[str, torch.Tensor]:
        loss_terms = self.model(batch, return_loss_info=True)
        if isinstance(loss_terms, tuple):
            loss = sum(loss_terms)
        else:
            loss = loss_terms
        return {"loss": loss}

    def validation_step(self, batch) -> Dict[str, torch.Tensor]:
        return self.training_step(batch)

    def transfer_batch_to_device(self, batch, device):
        return batch.to(device)
