# Training Workflows

Three NExT-Mol workflows are exposed as example scripts under `examples/molecule_generation/next_mol/`. Each uses the unified `Trainer` + `BaseTask` protocol (no Lightning).

See **[14-examples-and-runbook.md](14-examples-and-runbook.md)** for install steps, full CLI tables, programmatic examples, and troubleshooting.

## Workflow A: De Novo 3D Generation

1. `Trainer.fit(NextMolLLMTask)` — train MoLlama on SELFIES
2. `Trainer.fit(NextMolDMTTask)` — train DMT with frozen/LoRA LLM conditioning
3. Sample: generate SELFIES → DMT reverse SDE → 3D coords

```bash
# Prerequisites: data installed via scripts/install_nextmol_data.py

# Stage 1 — MoLlama
python examples/molecule_generation/next_mol/train_mollama.py
python examples/molecule_generation/next_mol/train_mollama.py \
  --batch_size 32 --max_epochs 10 \
  --checkpoint_dir ~/.torch_pharma/checkpoints/nextmol/mollama/exp1

# Stage 2 — DMT with LLM conditioning
python examples/molecule_generation/next_mol/train_dmt_uncond.py
python examples/molecule_generation/next_mol/train_dmt_uncond.py \
  --batch_size 32 --max_epochs 100 \
  --checkpoint_dir ~/.torch_pharma/checkpoints/nextmol/dmt/de_novo
```

| Stage | Data | Default checkpoint dir |
|-------|------|------------------------|
| MoLlama | `$TORCH_PHARMA_HOME/data/nextmol/qm9_lm/` | `.../checkpoints/nextmol/mollama/` |
| DMT | `$TORCH_PHARMA_HOME/data/nextmol/tordf_qm9/` | `.../checkpoints/nextmol/dmt/` |

## Workflow B: Conditional 3D Generation

Property-conditioned LLM + DMT with `context` tensor on QM9 properties:

```bash
python examples/molecule_generation/next_mol/train_dmt_cond.py --property gap
python examples/molecule_generation/next_mol/train_dmt_cond.py --property homo --max_epochs 100
python examples/molecule_generation/next_mol/train_dmt_cond.py --property lumo --batch_size 16
```

Supported `--property` values: `mu`, `alpha`, `homo`, `lumo`, `gap`, `Cv`.

## Workflow C: Conformer Prediction

Train DMT on `QM9TorDFDataset` without the LLM generation step. Set `use_llm=False` in `NextMolTrainingConfig` (programmatic — see runbook):

```python
from torch_pharma.models.diffusion.config import NextMolTrainingConfig

train_cfg = NextMolTrainingConfig(use_llm=False, delta_train_epochs=10)
```

Or point `--data_root` at the TorDF cache and use the DMT example script for the same data layout:

```bash
python examples/molecule_generation/next_mol/train_dmt_uncond.py \
  --data_root ~/.torch_pharma/data/nextmol/tordf_qm9
```

## Example Script CLI Reference

### `train_mollama.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | `qm9_lm` path | QM92014 SELFIES cache |
| `--checkpoint_dir` | `mollama/` checkpoints | `ModelCheckpoint` output |
| `--batch_size` | `32` | |
| `--max_epochs` | `10` | |
| `--llm_model` | `acharkq/MoLlama` | HuggingFace id |
| `--log_level` | `INFO` | |
| `--log_file` | none | Optional log file path |

### `train_dmt_uncond.py` / `train_dmt_cond.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | `tordf_qm9` path | GEOM-QM9 TorDF cache |
| `--checkpoint_dir` | `dmt/` checkpoints | |
| `--batch_size` | `32` | |
| `--max_epochs` | `100` | |
| `--use_llm` | `True` | (`train_dmt_uncond.py` only) |
| `--property` | `gap` | (`train_dmt_cond.py` only) |
| `--log_level` / `--log_file` | | Logging |

## Programmatic Training

Minimal DMT training without the example script:

```python
from torch_pharma.data.datamodules.nextmol_dm import build_nextmol_dataloaders
from torch_pharma.models.diffusion.config import DGTDiffusionConfig, NextMolTrainingConfig
from torch_pharma.models.llm.mollama import init_mollama_tokenizer
from torch_pharma.tasks.molecule_generation.nextmol_dmt_task import NextMolDMTTask
from torch_pharma.training import DeltaTrainingCallback, ModelCheckpoint, Trainer

train_cfg = NextMolTrainingConfig(use_llm=True, delta_train_epochs=10)
dmt_cfg = DGTDiffusionConfig.dmt_b()
tokenizer = init_mollama_tokenizer(train_cfg.llm_model)

train_loader, val_loader = build_nextmol_dataloaders(
    dataset_name="QM9-df",
    selfies_tokenizer=tokenizer,
    batch_size=32,
    mode="dmt",
)

task = NextMolDMTTask(train_cfg, dmt_cfg)
trainer = Trainer(
    max_epochs=100,
    precision="bf16-mixed",
    callbacks=[
        DeltaTrainingCallback(warmup_epochs=10),
        ModelCheckpoint(monitor="val_loss"),
    ],
)
trainer.fit(task, train_loader, val_loader)
```

Post-training validation only:

```python
metrics = trainer.validate(task, val_loader)
# {'val_loss': ..., 'val_lm_loss': ..., 'val_diff_loss': ...}
```

## Key Hyperparameters

| Parameter | QM9 | Geom-DRUGS |
|-----------|-----|------------|
| `pos_std` | 1.7226 | 2.4777 |
| `sampling_steps` | 100 | 100 |
| `noise_scheduler` | cosine | cosine |

Set via `NextMolTrainingConfig(pos_std=2.4777)` for Geom-DRUGS; use `dataset_name="geom-drugs"` in `build_nextmol_dataloaders`.

## Delta Training

`DeltaTrainingCallback` freezes all params except the projector for the first 10 epochs, matching NExT-Mol `set_trainble_params`. Controlled by `NextMolTrainingConfig.delta_train_epochs`.

## Trainer Callbacks Used

| Callback | Scripts | Purpose |
|----------|---------|---------|
| `ModelCheckpoint` | all three | Save best `val_loss` |
| `DeltaTrainingCallback` | DMT scripts | Projector-only warmup |
