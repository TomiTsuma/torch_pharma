# Training API

Training engine and utilities.

---

## Overview

The `training` module provides a unified interface for training molecular ML models.

```python
from torch_pharma.training import (
    Trainer,
    Callback,
    ModelCheckpoint,
    EarlyStopping,
)
```

---

## Trainer

Main training engine.

```python
class Trainer:
    """
    PyTorch Lightning-style trainer for molecular ML.
    
    Handles:
    - Training loops
    - Validation loops
    - Device management (CPU/GPU)
    - Distributed training
    - Checkpointing
    - Logging
    """
```

### Constructor

```python
def __init__(
    self,
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: Union[int, List[int]] = "auto",
    log_every_n_steps: int = 50,
    enable_checkpointing: bool = True,
    default_root_dir: str = "./checkpoints",
    callbacks: Optional[List[Callback]] = None,
    logger: Optional[Any] = None
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `max_epochs` | `int` | Maximum number of epochs |
| `accelerator` | `str` | Device type: "cpu", "gpu", "auto" |
| `devices` | `Union[int, List[int]]` | Number of devices or device IDs |
| `log_every_n_steps` | `int` | Logging frequency |
| `enable_checkpointing` | `bool` | Enable model checkpointing |
| `default_root_dir` | `str` | Directory for checkpoints and logs |
| `callbacks` | `Optional[List[Callback]]` | List of callbacks |
| `logger` | `Optional[Any]` | Logger (WandB, TensorBoard, etc.) |

**Example:**

```python
from torch_pharma.training import Trainer

trainer = Trainer(
    max_epochs=100,
    accelerator="auto",  # Use GPU if available
    log_every_n_steps=10,
    callbacks=[EarlyStopping(monitor="val_loss"), ModelCheckpoint()]
)
```

---

### Methods

#### fit

```python
def fit(
    self,
    task: BaseTask,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None
) -> None:
```

Train a task on the given data.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `task` | `BaseTask` | Task to train |
| `train_loader` | `DataLoader` | Training data loader |
| `val_loader` | `Optional[DataLoader]` | Validation data loader |

**Example:**

```python
trainer.fit(task, train_loader, val_loader)
```

---

#### validate

```python
def validate(
    self,
    task: BaseTask,
    val_loader: DataLoader
) -> Dict[str, float]:
```

Run validation and return metrics.

**Returns:**

- `Dict[str, float]`: Validation metrics

**Example:**

```python
metrics = trainer.validate(task, val_loader)
print(f"Validation loss: {metrics['val_loss']}")
```

---

#### test

```python
def test(
    self,
    task: BaseTask,
    test_loader: DataLoader
) -> Dict[str, float]:
```

Run evaluation on test set.

---

#### predict

```python
def predict(
    self,
    task: BaseTask,
    dataloader: DataLoader
) -> torch.Tensor:
```

Generate predictions.

**Returns:**

- `torch.Tensor`: Model predictions

---

## Callbacks

### Callback

Base class for training callbacks.

```python
class Callback:
    """
    Base class for training callbacks.
    
    Override methods to hook into training lifecycle.
    """
```

#### Methods

```python
def on_init_start(self, trainer):
    """Called when trainer initialization begins."""
    
def on_init_end(self, trainer):
    """Called when trainer initialization ends."""
    
def on_train_start(self, trainer, task):
    """Called when training begins."""
    
def on_train_epoch_start(self, trainer, task):
    """Called at the start of each training epoch."""
    
def on_train_batch_end(self, trainer, task, outputs, batch, batch_idx):
    """Called at the end of each training batch."""
    
def on_validation_epoch_start(self, trainer, task):
    """Called at the start of each validation epoch."""
    
def on_validation_batch_end(self, trainer, task, outputs, batch, batch_idx):
    """Called at the end of each validation batch."""
    
def on_validation_epoch_end(self, trainer, task):
    """Called at the end of each validation epoch."""
    
def on_train_end(self, trainer, task):
    """Called when training ends."""
    
def on_save_checkpoint(self, trainer, task, checkpoint):
    """Called when saving a checkpoint."""
```

---

### ModelCheckpoint

Save model checkpoints during training.

```python
class ModelCheckpoint(Callback):
    """
    Save model checkpoints based on validation metric.
    """
```

#### Constructor

```python
def __init__(
    self,
    monitor: str = "val_loss",
    mode: str = "min",  # or "max"
    save_top_k: int = 1,
    filename: str = "{epoch:02d}-{val_loss:.2f}",
    save_last: bool = True,
    save_weights_only: bool = False
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `monitor` | `str` | Metric to monitor |
| `mode` | `str` | "min" or "max" |
| `save_top_k` | `int` | Number of best checkpoints to keep |
| `filename` | `str` | Checkpoint filename format |

**Example:**

```python
checkpoint = ModelCheckpoint(
    monitor="val_mae",
    mode="min",
    save_top_k=3,
    filename="best-{epoch:02d}-{val_mae:.4f}"
)

trainer = Trainer(callbacks=[checkpoint])
```

---

### EarlyStopping

Stop training when metric stops improving.

```python
class EarlyStopping(Callback):
    """
    Stop training early if validation metric doesn't improve.
    """
```

#### Constructor

```python
def __init__(
    self,
    monitor: str = "val_loss",
    min_delta: float = 0.0,
    patience: int = 10,
    mode: str = "min",
    verbose: bool = True
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `monitor` | `str` | Metric to monitor |
| `min_delta` | `float` | Minimum change to qualify as improvement |
| `patience` | `int` | Epochs to wait before stopping |
| `mode` | `str` | "min" or "max" |
| `verbose` | `bool` | Print messages |

**Example:**

```python
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    mode="min"
)

trainer = Trainer(callbacks=[early_stop])
```

---

### LearningRateMonitor

Monitor and log learning rate.

```python
class LearningRateMonitor(Callback):
    """
    Log learning rate at each step.
    """
    
    def __init__(self, logging_interval: str = "step")
```

---

## Checkpointing

### save_checkpoint

```python
def save_checkpoint(
    task: BaseTask,
    filepath: str,
    metadata: Optional[Dict] = None
) -> None:
```

Save a task checkpoint.

### load_checkpoint

```python
def load_checkpoint(
    task: BaseTask,
    filepath: str,
    strict: bool = True
) -> BaseTask:
```

Load a task from checkpoint.

**Example:**

```python
from torch_pharma.training import save_checkpoint, load_checkpoint

# Save
checkpoint = {
    "epoch": trainer.current_epoch,
    "model_state_dict": task.model.state_dict(),
    "optimizer_state_dict": task.optimizer.state_dict()
}
torch.save(checkpoint, "model.pt")

# Load
ckpt = torch.load("model.pt")
task.model.load_state_dict(ckpt["model_state_dict"])
task.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
```

---

## Logging

### WandbLogger

Weights & Biases integration.

```python
from torch_pharma.integrations import WandbLogger

logger = WandbLogger(
    project="torch-pharma",
    name="experiment-1",
    config=config
)

trainer = Trainer(logger=logger)
```

---

## Distributed Training

The Trainer supports distributed training:

```python
# Single machine, multiple GPUs
trainer = Trainer(
    accelerator="gpu",
    devices=4  # Use 4 GPUs
)

# Multiple machines
# Use torchrun or mpirun
trainer = Trainer(
    accelerator="gpu",
    devices="auto",
    strategy="ddp"
)
```

---

## Custom Callbacks

Creating a custom callback:

```python
from torch_pharma.training import Callback

class MyCallback(Callback):
    def on_train_batch_end(self, trainer, task, outputs, batch, batch_idx):
        # Log custom metric
        loss = outputs["loss"]
        custom_metric = compute_metric(batch, task)
        trainer.logger.log({"custom_metric": custom_metric})
    
    def on_validation_epoch_end(self, trainer, task):
        # Compute per-epoch statistics
        all_preds = trainer.predictions
        mean_pred = torch.mean(all_preds)
        trainer.logger.log({"mean_prediction": mean_pred})
```
