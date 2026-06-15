"""NExT-Mol DMT (stage 2/3) training task."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_batch

from torch_pharma.features.kabsch import get_align_noise
from torch_pharma.config.model.nextmol.config import DGTDiffusionConfig, NextMolTrainingConfig
from torch_pharma.models.diffusion.dmt import DGTDiffusion
from torch_pharma.models.diffusion.sde_sampler import reverse_vp_sde_sample
from torch_pharma.models.diffusion.vp_scheduler import NoiseScheduleVPV2
from torch_pharma.models.llm.mollama import MoLlamaConditioning, load_mollama
from torch_pharma.models.llm.projector import LLMProjector
from torch_pharma.tasks.base import BaseTask
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


def to_dense_batch_list_tensor(list_of_tensor, batch, bs, max_num_nodes):
    shapes = [t.shape[1] for t in list_of_tensor]
    combined = torch.cat(list_of_tensor, dim=1)
    dense, batch_mask = to_dense_batch(combined, batch, batch_size=bs, max_num_nodes=max_num_nodes)
    return torch.split(dense, shapes, dim=2), batch_mask


class NextMolDMTModule(nn.Module):
    """Combined LLM conditioning + DGTDiffusion for NExT-Mol."""

    def __init__(self, train_cfg: NextMolTrainingConfig, dmt_cfg: DGTDiffusionConfig):
        super().__init__()
        self.train_cfg = train_cfg
        self.use_llm = train_cfg.use_llm
        self.llm_tune = train_cfg.llm_tune
        self.use_llm_projector = train_cfg.use_llm_projector
        self.llm_jk = train_cfg.llm_jk
        self.pred_noise = dmt_cfg.pred_noise
        self.align_loss = train_cfg.align_loss
        self.translation_correction = train_cfg.translation_correction
        self.align_prediction = train_cfg.align_prediction
        self.reduce_node_mean = train_cfg.reduce_node_mean

        in_dim = None
        if self.use_llm:
            self.llm_model = load_mollama(
                train_cfg.llm_model,
                llm_tune=train_cfg.llm_tune,
                lora_r=train_cfg.lora_r,
                lora_alpha=train_cfg.lora_alpha,
                lora_dropout=train_cfg.lora_dropout,
            )
            self.llm_cond = MoLlamaConditioning(
                self.llm_model, self.llm_model.config.hidden_size
            )
            if self.use_llm_projector:
                self.llm_projector = LLMProjector(
                    self.llm_model.config.hidden_size,
                    dmt_cfg.hidden_size,
                    self.llm_jk,
                    use_self_att_proj=False,
                    llm_num_layers=self.llm_model.config.num_hidden_layers,
                )
                in_dim = dmt_cfg.hidden_size
            else:
                in_dim = self.llm_model.config.hidden_size

        args = SimpleNamespace(**{**dmt_cfg.__dict__, **train_cfg.__dict__})
        self.diffusion_model = DGTDiffusion(args, in_dim=in_dim)
        self.noise_scheduler = NoiseScheduleVPV2(schedule=train_cfg.noise_scheduler)

    def forward_llm(self, data_batch, selfies_batch, context=None):
        hidden_states, lm_loss = self.llm_cond(selfies_batch, context, self.llm_tune)
        if self.use_llm_projector:
            lm_x = self.llm_projector(hidden_states, data_batch.rdmol2selfies, selfies_batch)
            return lm_x, lm_loss
        lm_embeds = hidden_states[-1] if self.llm_jk == "last" else hidden_states[-1]
        lm_x = torch.bmm(data_batch.rdmol2selfies.to(lm_embeds.dtype), lm_embeds)
        norm = torch.clamp(torch.sum(data_batch.rdmol2selfies, dim=-1, keepdim=True), min=1)
        return lm_x / norm, lm_loss

    def forward(self, data_batch, selfies_batch, context=None):
        lm_loss = torch.tensor(0.0, device=data_batch.x.device)
        lm_x = None
        if self.use_llm:
            lm_x, lm_loss = self.forward_llm(data_batch, selfies_batch, context)

        bs = len(data_batch.smiles)
        max_num_nodes = data_batch.max_seqlen
        total_num_nodes = data_batch.x.shape[0]

        if context is not None:
            pred_pos, pred_noise, _ = self.diffusion_model(data_batch, lm_x, context)
        else:
            pred_pos, pred_noise = self.diffusion_model(data_batch, lm_x)

        if self.pred_noise:
            pred_batch = (
                to_dense_batch(pred_pos, data_batch.batch, batch_size=bs, max_num_nodes=max_num_nodes)[0]
                if self.align_prediction
                else None
            )
            tensors, batch_mask = to_dense_batch_list_tensor(
                [data_batch.pos, data_batch.gt_pos, pred_noise, data_batch.noise],
                data_batch.batch,
                bs,
                max_num_nodes,
            )
            pos_t_batch, pos_0_batch, pred_noise_batch, gt_noise_batch = tensors
            diff_loss = self._noise_loss(
                pred_noise_batch,
                gt_noise_batch,
                pos_0_batch,
                pos_t_batch,
                pred_batch,
                data_batch.alpha_t_batch,
                data_batch.sigma_t_batch,
                total_num_nodes,
                batch_mask,
            )
        else:
            pos_t_batch, batch_mask = to_dense_batch(
                data_batch.pos, data_batch.batch, batch_size=bs, max_num_nodes=max_num_nodes
            )
            pos_0_batch, _ = to_dense_batch(
                data_batch.gt_pos, data_batch.batch, batch_size=bs, max_num_nodes=max_num_nodes
            )
            pred_pos_batch, _ = to_dense_batch(pred_pos, data_batch.batch, batch_size=bs)
            diff_loss = F.mse_loss(pred_pos_batch[batch_mask], pos_0_batch[batch_mask])

        loss = lm_loss * self.train_cfg.lm_loss + diff_loss * self.train_cfg.diff_loss
        return loss, lm_loss, diff_loss

    def _noise_loss(
        self,
        pred_noise_batch,
        gt_noise_batch,
        pos_0_batch,
        pos_t_batch,
        pred_batch,
        alpha_t_batch,
        sigma_t_batch,
        total_num_nodes,
        batch_mask,
    ):
        if self.align_loss:
            aligned = get_align_noise(
                pos_t_batch,
                pos_0_batch,
                pred_batch,
                alpha_t_batch.unsqueeze(-1),
                sigma_t_batch.unsqueeze(-1),
                batch_mask,
                self.translation_correction,
                self.align_prediction,
            )
            gt_noise_batch = aligned

        if self.reduce_node_mean:
            diff = (pred_noise_batch - gt_noise_batch) ** 2
            return diff[batch_mask].mean()
        return F.mse_loss(pred_noise_batch[batch_mask], gt_noise_batch[batch_mask])

    @torch.no_grad()
    def sample(self, data_batch, selfies_batch, context=None):
        lm_x = None
        if self.use_llm:
            lm_x, _ = self.forward_llm(data_batch, selfies_batch, context)
        return reverse_vp_sde_sample(
            self.diffusion_model,
            data_batch,
            self.noise_scheduler,
            lm_x=lm_x,
            context=context,
            sampling_steps=self.train_cfg.sampling_steps,
            t_cond=self.train_cfg.t_cond,
            pos_std=self.train_cfg.pos_std,
        )


class NextMolDMTTask(BaseTask):
    def __init__(
        self,
        train_cfg: Optional[NextMolTrainingConfig] = None,
        dmt_cfg: Optional[DGTDiffusionConfig] = None,
    ):
        super().__init__()
        self.train_cfg = train_cfg or NextMolTrainingConfig()
        self.dmt_cfg_obj = dmt_cfg or DGTDiffusionConfig.dmt_b()
        self._module: Optional[NextMolDMTModule] = None

    def configure_model(self) -> nn.Module:
        log.info(
            "Building NextMolDMTModule: use_llm=%s llm_tune=%s noise=%s",
            self.train_cfg.use_llm,
            self.train_cfg.llm_tune,
            self.train_cfg.noise_scheduler,
        )
        self._module = NextMolDMTModule(self.train_cfg, self.dmt_cfg_obj)
        return self._module

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, None]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.init_lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        return optimizer, None

    def training_step(self, batch) -> Dict[str, torch.Tensor]:
        data_batch, selfies_batch = batch
        loss, lm_loss, diff_loss = self.model(data_batch, selfies_batch, getattr(data_batch, "context", None))
        return {"loss": loss, "lm_loss": lm_loss, "diff_loss": diff_loss}

    def validation_step(self, batch) -> Dict[str, torch.Tensor]:
        return self.training_step(batch)

    def predict_step(self, batch):
        data_batch, selfies_batch = batch
        return self.model.sample(data_batch, selfies_batch)

    def transfer_batch_to_device(self, batch, device):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0].to(device), batch[1].to(device)
        return super().transfer_batch_to_device(batch, device)
