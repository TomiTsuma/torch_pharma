# Unified Trainer Design

## BaseTask Protocol

```python
class BaseTask(ABC, nn.Module):
    def configure_model(self) -> nn.Module: ...
    def configure_optimizers(self) -> Tuple[Optimizer, Optional[Scheduler]]: ...
    def training_step(self, batch) -> Dict[str, Tensor]:  # must include "loss"
    def validation_step(self, batch) -> Dict[str, Tensor]: ...
```

Implemented in `torch_pharma/tasks/base.py`.

## Trainer API

Mirrors `docs/api/training/index.md` without Lightning:

- `fit(task, train_loader, val_loader)`
- `validate(task, val_loader)`
- `test(task, test_loader)`
- `predict(task, dataloader)`

Location: `torch_pharma/training/trainer.py`

Default checkpoint directory: `$TORCH_PHARMA_HOME/checkpoints/` (see `torch_pharma/paths.py`). NExT-Mol example scripts use `$TORCH_PHARMA_HOME/checkpoints/nextmol/{mollama,dmt}/`.

## Callbacks

| Callback | Purpose |
|----------|---------|
| `ModelCheckpoint` | Save best checkpoints |
| `EarlyStopping` | Stop on plateau |
| `DeltaTrainingCallback` | NExT-Mol projector warmup |

## NExT-Mol Task Mapping

| NExT-Mol LightningModule | torch_pharma Task |
|--------------------------|-------------------|
| `LLMPL` | `NextMolLLMTask` |
| `DiffussionPL` | `NextMolDMTTask` |

## Example Usage

Example scripts wire `Trainer.fit` end-to-end. Minimal pattern:

```python
from torch_pharma.training import Trainer, ModelCheckpoint

trainer = Trainer(
    max_epochs=10,
    precision="bf16-mixed",
    callbacks=[ModelCheckpoint(monitor="val_loss")],
)
trainer.fit(task, train_loader, val_loader)
trainer.validate(task, val_loader)
samples = trainer.predict(task, val_loader)
```

See [14-examples-and-runbook.md](14-examples-and-runbook.md) for full training and evaluation runbooks.
