"""NExT-Mol MoLlama (stage 1) SELFIES language-model training task."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from torch_pharma.config.model.nextmol.config import NextMolTrainingConfig
from torch_pharma.models.llm.mollama import load_mollama
from torch_pharma.tasks.base import BaseTask
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


class NextMolLLMTask(BaseTask):
    def __init__(self, train_cfg: Optional[NextMolTrainingConfig] = None):
        super().__init__()
        self.train_cfg = train_cfg or NextMolTrainingConfig(llm_tune="lora", lm_loss=1.0, diff_loss=0.0)

    def configure_model(self) -> nn.Module:
        log.info("Loading MoLlama for stage-1 training: model=%s tune=%s", self.train_cfg.llm_model, self.train_cfg.llm_tune)
        return load_mollama(
            self.train_cfg.llm_model,
            llm_tune=self.train_cfg.llm_tune,
            lora_r=self.train_cfg.lora_r,
            lora_alpha=self.train_cfg.lora_alpha,
            lora_dropout=self.train_cfg.lora_dropout,
        )

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, None]:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.train_cfg.init_lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        return optimizer, None

    def training_step(self, batch) -> Dict[str, torch.Tensor]:
        targets = batch.input_ids.masked_fill(~batch.attention_mask.bool(), -100)
        outputs = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=targets,
            return_dict=True,
        )
        return {"loss": outputs.loss}

    def validation_step(self, batch) -> Dict[str, torch.Tensor]:
        return self.training_step(batch)

    def transfer_batch_to_device(self, batch, device):
        return batch.to(device)
