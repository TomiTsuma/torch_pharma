"""Unified training engine for torch_pharma."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from torch_pharma.paths import TORCH_PHARMA_CHECKPOINTS, ensure_nextmol_dirs
from torch_pharma.training.callbacks import Callback
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


class Trainer:
    """Lightning-style trainer without Lightning dependency."""

    def __init__(
        self,
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: Union[int, str] = "auto",
        log_every_n_steps: int = 50,
        enable_checkpointing: bool = True,
        default_root_dir: str | None = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[Any] = None,
        accumulate_grad_batches: int = 1,
        precision: str = "32",
        gradient_clip_val: Optional[float] = None,
    ):
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.log_every_n_steps = log_every_n_steps
        self.enable_checkpointing = enable_checkpointing
        ensure_nextmol_dirs()
        self.default_root_dir = str(default_root_dir or TORCH_PHARMA_CHECKPOINTS)
        self.callbacks = callbacks or []
        self.logger = logger
        self.accumulate_grad_batches = accumulate_grad_batches
        self.precision = precision
        self.gradient_clip_val = gradient_clip_val

        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        self.callback_metrics: Dict[str, float] = {}
        self.device = self._resolve_device()
        log.info(
            "Trainer initialized: max_epochs=%d device=%s precision=%s checkpoint_dir=%s",
            self.max_epochs,
            self.device,
            self.precision,
            self.default_root_dir,
        )

    def _resolve_device(self) -> torch.device:
        if self.accelerator == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available() and self.accelerator in ("auto", "gpu", "cuda"):
            if isinstance(self.devices, int) and self.devices > 0:
                return torch.device("cuda:0")
            return torch.device("cuda")
        return torch.device("cpu")

    def _autocast_enabled(self) -> bool:
        return self.precision in ("16", "16-mixed", "bf16", "bf16-mixed")

    def _autocast_dtype(self):
        if "bf16" in self.precision:
            return torch.bfloat16
        if "16" in self.precision:
            return torch.float16
        return torch.float32

    def _call_callbacks(self, method: str, *args, **kwargs) -> None:
        for cb in self.callbacks:
            getattr(cb, method)(*args, **kwargs)

    def fit(
        self,
        task,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self._call_callbacks("on_init_start", self)
        log.info("Setting up task on device %s", self.device)
        task.setup(self.device)
        self._call_callbacks("on_init_end", self)
        self._call_callbacks("on_train_start", self, task)
        log.info("Starting training for %d epochs", self.max_epochs)

        for epoch in range(self.max_epochs):
            if self.should_stop:
                break
            self.current_epoch = epoch
            log.info("Epoch %d/%d", epoch + 1, self.max_epochs)
            self._call_callbacks("on_train_epoch_start", self, task)
            self._train_epoch(task, train_loader)
            self._call_callbacks("on_train_epoch_end", self, task)

            if val_loader is not None:
                self.validate(task, val_loader)

            if self.should_stop:
                break

        self._call_callbacks("on_train_end", self, task)
        log.info("Training finished at epoch %d (global_step=%d)", self.current_epoch + 1, self.global_step)

    def _train_epoch(self, task, train_loader: DataLoader) -> None:
        task.train()
        task.on_train_epoch_start(self)
        optimizer = task.optimizer
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            batch = task.transfer_batch_to_device(batch, self.device)
            with torch.cuda.amp.autocast(
                enabled=self._autocast_enabled() and self.device.type == "cuda",
                dtype=self._autocast_dtype(),
            ):
                outputs = task.training_step(batch)
            loss = outputs["loss"] / self.accumulate_grad_batches
            loss.backward()

            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(task.parameters(), self.gradient_clip_val)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if task.scheduler is not None:
                    task.scheduler_step(self.global_step)
                self.global_step += 1

            self._call_callbacks("on_train_batch_end", self, task, outputs, batch, batch_idx)
            if self.global_step % self.log_every_n_steps == 0:
                self._log_metrics(outputs, prefix="train")
                self._log_to_console(outputs, prefix="train", step=self.global_step)
        task.on_train_epoch_end(self)

    @torch.no_grad()
    def validate(self, task, val_loader: DataLoader) -> Dict[str, float]:
        task.eval()
        self._call_callbacks("on_validation_epoch_start", self, task)
        task.on_validation_epoch_start(self)
        metrics_sum: Dict[str, float] = {}
        count = 0

        for batch_idx, batch in enumerate(val_loader):
            batch = task.transfer_batch_to_device(batch, self.device)
            with torch.cuda.amp.autocast(
                enabled=self._autocast_enabled() and self.device.type == "cuda",
                dtype=self._autocast_dtype(),
            ):
                outputs = task.validation_step(batch)
            self._call_callbacks("on_validation_batch_end", self, task, outputs, batch, batch_idx)
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v.item())
            count += 1

        if count:
            self.callback_metrics = {f"val_{k}": v / count for k, v in metrics_sum.items()}
        task.on_validation_epoch_end(self)
        self._call_callbacks("on_validation_epoch_end", self, task)
        self._log_metrics(self.callback_metrics, prefix="val")
        self._log_to_console(self.callback_metrics, prefix="val", step=self.global_step)
        return self.callback_metrics

    def test(self, task, test_loader: DataLoader) -> Dict[str, float]:
        return self.validate(task, test_loader)

    @torch.no_grad()
    def predict(self, task, dataloader: DataLoader):
        task.eval()
        predictions = []
        for batch in dataloader:
            batch = task.transfer_batch_to_device(batch, self.device)
            predictions.append(task.predict_step(batch))
        return predictions

    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        if self.logger is None:
            return
        log_dict = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = float(v.item())
            if isinstance(v, (int, float)):
                key = k if k.startswith(prefix) else f"{prefix}_{k}" if prefix else k
                log_dict[key] = v
        if hasattr(self.logger, "log"):
            self.logger.log(log_dict, step=self.global_step)

    def _log_to_console(self, metrics: Dict[str, Any], prefix: str = "", step: int = 0) -> None:
        parts = []
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = float(v.item())
            if isinstance(v, (int, float)):
                key = k if not prefix or k.startswith(f"{prefix}_") else f"{prefix}_{k}"
                parts.append(f"{key}={v:.6g}")
        if parts:
            log.info("step=%d %s", step, " ".join(parts))
