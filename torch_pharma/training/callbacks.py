"""Training callbacks for torch_pharma Trainer."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


class Callback:
    """Base callback with Lightning-style lifecycle hooks."""

    def on_init_start(self, trainer) -> None:
        pass

    def on_init_end(self, trainer) -> None:
        pass

    def on_train_start(self, trainer, task) -> None:
        pass

    def on_train_epoch_start(self, trainer, task) -> None:
        pass

    def on_train_batch_end(self, trainer, task, outputs, batch, batch_idx) -> None:
        pass

    def on_train_epoch_end(self, trainer, task) -> None:
        pass

    def on_validation_epoch_start(self, trainer, task) -> None:
        pass

    def on_validation_batch_end(self, trainer, task, outputs, batch, batch_idx) -> None:
        pass

    def on_validation_epoch_end(self, trainer, task) -> None:
        pass

    def on_train_end(self, trainer, task) -> None:
        pass

    def on_save_checkpoint(self, trainer, task, checkpoint: Dict[str, Any]) -> None:
        pass


class ModelCheckpoint(Callback):
    """Save checkpoints based on a monitored validation metric."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        filename: str = "epoch={epoch:02d}-{val_loss:.4f}",
        save_last: bool = True,
        save_weights_only: bool = False,
        dirpath: Optional[str] = None,
    ):
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        self.dirpath = dirpath
        self.best_score: Optional[float] = None
        self.best_paths: List[str] = []

    def _is_better(self, current: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        return current < best if self.mode == "min" else current > best

    def on_validation_epoch_end(self, trainer, task) -> None:
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        current = float(metrics[self.monitor])
        if not self._is_better(current, self.best_score):
            if self.save_last:
                self._save(trainer, task, suffix="last")
            return
        self.best_score = current
        path = self._save(trainer, task, current=current)
        self.best_paths.append(path)
        if len(self.best_paths) > self.save_top_k:
            old = self.best_paths.pop(0)
            if os.path.exists(old):
                os.remove(old)

    def _save(self, trainer, task, current=None, suffix=None) -> str:
        from torch_pharma.training.checkpoint import save_checkpoint

        dirpath = self.dirpath or trainer.default_root_dir
        os.makedirs(dirpath, exist_ok=True)
        if suffix:
            name = f"checkpoint-{suffix}.pt"
        else:
            name = self.filename.format(
                epoch=trainer.current_epoch,
                **{k.replace("val_", ""): v for k, v in trainer.callback_metrics.items()},
                **trainer.callback_metrics,
            )
            if not name.endswith(".pt"):
                name += ".pt"
        path = os.path.join(dirpath, name)
        save_checkpoint(task, path, epoch=trainer.current_epoch, metrics=trainer.callback_metrics)
        return path


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.best_score: Optional[float] = None
        self.wait = 0

    def on_validation_epoch_end(self, trainer, task) -> None:
        if self.monitor not in trainer.callback_metrics:
            return
        current = float(trainer.callback_metrics[self.monitor])
        if self.best_score is None:
            self.best_score = current
            return
        improved = (
            current < self.best_score - self.min_delta
            if self.mode == "min"
            else current > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.should_stop = True
                if self.verbose:
                    log.warning(
                        "EarlyStopping: stopping after %d epochs without improvement on %s",
                        self.patience,
                        self.monitor,
                    )


class DeltaTrainingCallback(Callback):
    """NExT-Mol delta training: warmup projector before full model training."""

    def __init__(self, warmup_epochs: int = 10, param_patterns: Optional[List[str]] = None):
        self.warmup_epochs = warmup_epochs
        self.param_patterns = param_patterns or ["projector", "extended_node_emb", "llm_projector"]
        self._restored = False

    def on_train_epoch_start(self, trainer, task) -> None:
        if not hasattr(task, "set_trainable_params"):
            return
        if trainer.current_epoch < self.warmup_epochs:
            if trainer.current_epoch == 0:
                log.info(
                    "Delta training warmup: training only %s for %d epochs",
                    self.param_patterns,
                    self.warmup_epochs,
                )
            task.set_trainable_params(self.param_patterns, delta_train=True)
        elif not self._restored:
            log.info("Delta training warmup complete; restoring full model training")
            task.set_trainable_params([], delta_train=False)
            task.restore_trainable_params()
            self._restored = True
