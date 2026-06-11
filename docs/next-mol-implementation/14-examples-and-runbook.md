# Examples and Runbook

Practical guide for installing data, running training scripts, evaluating checkpoints, and wiring custom experiments.

## Prerequisites

```bash
# From repo root
pip install -e ".[nextmol,dev]"

# Verify imports
pytest tests/test_nextmol_imports.py tests/test_paths.py -q
```

Requirements:

- Python 3.9+ (3.11 recommended for `scripts/install_nextmol_data.py`)
- CUDA GPU with bf16 support recommended for training scripts (they default to `precision="bf16-mixed"`)
- HuggingFace access for `acharkq/MoLlama` (downloaded on first run)

Optional environment override:

```bash
# Default is ~/.torch_pharma — override if needed
export TORCH_PHARMA_HOME=/path/to/your/torch_pharma_cache
```

On Windows PowerShell:

```powershell
$env:TORCH_PHARMA_HOME = "C:\path\to\torch_pharma_cache"
```

## Step 1: Install NExT-Mol Data

Download the OSF archive from the [NExT-Mol README](https://github.com/acharkq/NExT-Mol) or use an existing local copy, then run:

```bash
# Default source: ~/Downloads/osfstorage-archive/datasets
py -3.11 scripts/install_nextmol_data.py

# Custom OSF path
py -3.11 scripts/install_nextmol_data.py --source "C:/Users/you/Downloads/osfstorage-archive/datasets"

# Skip large Geom-DRUGS extract (~7 GB)
py -3.11 scripts/install_nextmol_data.py --skip-geom

# Re-extract geom_drugs_jodo.zip
py -3.11 scripts/install_nextmol_data.py --force-extract-geom
```

Expected layout after install:

```
~/.torch_pharma/data/nextmol/
├── qm9_lm/processed/data_qm9.pt          # MoLlama stage 1
├── qm9_lm/processed/split_dict_qm9.pt
├── tordf_qm9/processed_train.pt          # DMT stage 2/3
├── tordf_qm9/processed_val.pt
├── tordf_qm9/tordf.{train,val,test}
└── geom_drugs/processed_data.pt          # optional
```

Verify paths in Python:

```python
from torch_pharma.paths import NEXTMOL_QM9_LM, NEXTMOL_QM9_TORDF

assert (NEXTMOL_QM9_LM / "processed" / "data_qm9.pt").exists()
assert (NEXTMOL_QM9_TORDF / "processed_train.pt").exists()
```

## Step 2: Example Scripts Index

| Script | Workflow | Data root | Checkpoints |
|--------|----------|-----------|-------------|
| `examples/molecule_generation/next_mol/train_mollama.py` | Stage 1 — MoLlama SELFIES LM | `qm9_lm` | `checkpoints/nextmol/mollama/` |
| `examples/molecule_generation/next_mol/train_dmt_uncond.py` | Stage 2 — de novo 3D / conformer | `tordf_qm9` | `checkpoints/nextmol/dmt/` |
| `examples/molecule_generation/next_mol/train_dmt_cond.py` | Stage 2 — property-conditioned 3D | `tordf_qm9` | `checkpoints/nextmol/dmt/` |

Placeholder CLIs (logging only — use example scripts above for real workflows):

- `scripts/train.py` — unified training entry stub
- `scripts/evaluate.py` — evaluation entry stub
- `examples/molecule_generation/generate.py` — generation stub

## Step 3: Training Examples

### Stage 1 — MoLlama (Workflow A)

Train the SELFIES language model with LoRA fine-tuning:

```bash
python examples/molecule_generation/next_mol/train_mollama.py

# Smoke test (few epochs, small batch)
python examples/molecule_generation/next_mol/train_mollama.py \
  --max_epochs 1 --batch_size 4

# Custom paths and logging
python examples/molecule_generation/next_mol/train_mollama.py \
  --data_root ~/.torch_pharma/data/nextmol/qm9_lm \
  --checkpoint_dir ~/.torch_pharma/checkpoints/nextmol/mollama/run1 \
  --batch_size 32 --max_epochs 10 \
  --llm_model acharkq/MoLlama \
  --log_level DEBUG --log_file mollama_train.log
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | `$TORCH_PHARMA_HOME/data/nextmol/qm9_lm` | QM92014 cache |
| `--checkpoint_dir` | `.../checkpoints/nextmol/mollama` | Best `val_loss` checkpoint saved here |
| `--batch_size` | `32` | Training batch size |
| `--max_epochs` | `10` | Training epochs |
| `--llm_model` | `acharkq/MoLlama` | HuggingFace model id |

### Stage 2 — DMT Unconditional (Workflow A / C)

De novo 3D generation with LLM conditioning (default):

```bash
python examples/molecule_generation/next_mol/train_dmt_uncond.py

# Faster debug run
python examples/molecule_generation/next_mol/train_dmt_uncond.py \
  --max_epochs 2 --batch_size 4

# Custom checkpoint location
python examples/molecule_generation/next_mol/train_dmt_uncond.py \
  --checkpoint_dir ~/.torch_pharma/checkpoints/nextmol/dmt/uncond_qm9 \
  --max_epochs 100 --batch_size 32
```

Conformer prediction (DMT only, no LLM) — set `use_llm=False` programmatically (see [Programmatic training](#programmatic-training) below). The example script defaults `use_llm=True`.

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | `.../data/nextmol/tordf_qm9` | GEOM-QM9 TorDF cache |
| `--checkpoint_dir` | `.../checkpoints/nextmol/dmt` | DMT checkpoints |
| `--batch_size` | `32` | Training batch size |
| `--max_epochs` | `100` | Training epochs |
| `--use_llm` | `True` | Enable MoLlama conditioning |

`DeltaTrainingCallback` freezes all parameters except the LLM projector for the first 10 epochs (matching NExT-Mol delta training).

### Stage 2 — Property-Conditioned DMT (Workflow B)

```bash
# HOMO-LUMO gap conditioning (default)
python examples/molecule_generation/next_mol/train_dmt_cond.py

# Other QM9 properties
python examples/molecule_generation/next_mol/train_dmt_cond.py --property homo
python examples/molecule_generation/next_mol/train_dmt_cond.py --property lumo
python examples/molecule_generation/next_mol/train_dmt_cond.py --property mu
python examples/molecule_generation/next_mol/train_dmt_cond.py --property alpha
python examples/molecule_generation/next_mol/train_dmt_cond.py --property Cv

# Full run with logging
python examples/molecule_generation/next_mol/train_dmt_cond.py \
  --property gap \
  --batch_size 32 --max_epochs 100 \
  --log_file dmt_cond_gap.log
```

| `--property` choices | `mu`, `alpha`, `homo`, `lumo`, `gap`, `Cv` |

### Full three-stage pipeline (shell)

```bash
# 1. Install data (once)
py -3.11 scripts/install_nextmol_data.py

# 2. Train MoLlama
python examples/molecule_generation/next_mol/train_mollama.py --max_epochs 10

# 3. Train DMT with LLM conditioning
python examples/molecule_generation/next_mol/train_dmt_uncond.py --max_epochs 100

# 4. (Optional) Property-conditioned variant
python examples/molecule_generation/next_mol/train_dmt_cond.py --property gap --max_epochs 100
```

## Programmatic Training

Use the same components as the example scripts inside notebooks or custom drivers:

```python
from torch_pharma.data.datamodules.nextmol_dm import build_nextmol_dataloaders
from torch_pharma.models.diffusion.config import DGTDiffusionConfig, NextMolTrainingConfig
from torch_pharma.models.llm.mollama import init_mollama_tokenizer
from torch_pharma.tasks.molecule_generation.nextmol_dmt_task import NextMolDMTTask
from torch_pharma.training import DeltaTrainingCallback, ModelCheckpoint, Trainer

train_cfg = NextMolTrainingConfig(use_llm=False, delta_train_epochs=10)  # conformer-only
dmt_cfg = DGTDiffusionConfig.dmt_b()
tokenizer = init_mollama_tokenizer(train_cfg.llm_model)

train_loader, val_loader = build_nextmol_dataloaders(
    dataset_name="QM9-df",
    selfies_tokenizer=tokenizer,
    batch_size=8,
    mode="dmt",
)

task = NextMolDMTTask(train_cfg, dmt_cfg)
trainer = Trainer(
    max_epochs=2,
    precision="bf16-mixed",
    callbacks=[
        DeltaTrainingCallback(warmup_epochs=10),
        ModelCheckpoint(monitor="val_loss"),
    ],
)
trainer.fit(task, train_loader, val_loader)
```

MoLlama stage programmatically:

```python
from torch_pharma.data.datamodules.nextmol_dm import build_nextmol_dataloaders
from torch_pharma.models.diffusion.config import NextMolTrainingConfig
from torch_pharma.models.llm.mollama import init_mollama_tokenizer
from torch_pharma.tasks.molecule_generation.nextmol_llm_task import NextMolLLMTask
from torch_pharma.training import ModelCheckpoint, Trainer

cfg = NextMolTrainingConfig(llm_tune="lora", max_epochs=2)
tokenizer = init_mollama_tokenizer(cfg.llm_model)
train_loader, val_loader = build_nextmol_dataloaders(
    dataset_name="QM9",
    selfies_tokenizer=tokenizer,
    batch_size=8,
    mode="llm",
)
task = NextMolLLMTask(cfg)
trainer = Trainer(max_epochs=2, callbacks=[ModelCheckpoint(monitor="val_loss")])
trainer.fit(task, train_loader, val_loader)
```

## Step 4: Loading Checkpoints

### Native torch_pharma checkpoints

Checkpoints are saved by `ModelCheckpoint` under the `--checkpoint_dir` you pass (or the default under `TORCH_PHARMA_HOME`).

```python
import torch
from torch_pharma.models.diffusion.config import DGTDiffusionConfig, NextMolTrainingConfig
from torch_pharma.tasks.molecule_generation.nextmol_dmt_task import NextMolDMTTask

task = NextMolDMTTask(NextMolTrainingConfig(), DGTDiffusionConfig.dmt_b())
task.setup(torch.device("cuda"))

ckpt = torch.load("~/.torch_pharma/checkpoints/nextmol/dmt/best.ckpt", map_location="cuda")
task.load_state_dict(ckpt["state_dict"], strict=False)
task.eval()
```

### NExT-Mol Lightning checkpoints (OSF)

Place downloaded `.ckpt` files under `$TORCH_PHARMA_HOME/pretrained/nextmol/`:

```python
import torch
from torch_pharma.paths import NEXTMOL_PRETRAINED
from torch_pharma.models.diffusion.checkpoint_migration import load_nextmol_dmt_checkpoint
from torch_pharma.models.diffusion.config import DGTDiffusionConfig, NextMolTrainingConfig
from torch_pharma.tasks.molecule_generation.nextmol_dmt_task import NextMolDMTTask

task = NextMolDMTTask(NextMolTrainingConfig(), DGTDiffusionConfig.dmt_b())
task.setup(torch.device("cuda"))
load_nextmol_dmt_checkpoint(
    task.model,
    NEXTMOL_PRETRAINED / "dmt_b_epoch99.ckpt",
    strict=False,
)
```

See [13-checkpoint-migration.md](13-checkpoint-migration.md) for prefix remapping details.

## Step 5: Evaluation Examples

### Validate on held-out data

```python
import torch
from torch_pharma.data.datamodules.nextmol_dm import build_nextmol_dataloaders
from torch_pharma.models.diffusion.config import DGTDiffusionConfig, NextMolTrainingConfig
from torch_pharma.models.llm.mollama import init_mollama_tokenizer
from torch_pharma.tasks.molecule_generation.nextmol_dmt_task import NextMolDMTTask
from torch_pharma.training import Trainer

train_cfg = NextMolTrainingConfig()
tokenizer = init_mollama_tokenizer(train_cfg.llm_model)
_, val_loader = build_nextmol_dataloaders(
    dataset_name="QM9-df",
    selfies_tokenizer=tokenizer,
    batch_size=16,
    mode="dmt",
)

task = NextMolDMTTask(train_cfg, DGTDiffusionConfig.dmt_b())
trainer = Trainer(max_epochs=1)
trainer.validate(task, val_loader)  # logs val_loss, val_lm_loss, val_diff_loss
```

### Conformer metrics (COV-R / MAT-R)

```python
from rdkit import Chem

from torch_pharma.evaluation.nextmol.conformer_metrics import (
    conformer_recall,
    set_rdmol_positions,
)

# Each entry: (smiles, mol_with_2_conformers — index 0 = GT, index 1 = pred)
predictions = []
for smiles, gt_pos, pred_pos in your_results:
    mol = Chem.MolFromSmiles(smiles)
    mol = set_rdmol_positions(mol, gt_pos, add_conformer=True)
    mol = set_rdmol_positions(mol, pred_pos, add_conformer=True)
    predictions.append((smiles, mol))

metrics = conformer_recall(predictions, threshold=0.5)
print(metrics)  # cov_mean, mat_mean, cov_median, mat_median
```

### 2D generation metrics

```python
from torch_pharma.evaluation.nextmol.generation_metrics import (
    novelty_rate,
    uniqueness_rate,
    validity_rate,
)

generated_smiles = [...]  # from MoLlama decode or post-processing
training_smiles = set(open("train_smiles.txt").read().splitlines())

print("validity:", validity_rate(generated_smiles))
print("uniqueness:", uniqueness_rate(generated_smiles))
print("novelty:", novelty_rate(generated_smiles, training_smiles))
```

### Sampling (inference)

```python
import torch
from torch_pharma.data.datamodules.nextmol_dm import build_nextmol_dataloaders
from torch_pharma.models.diffusion.config import DGTDiffusionConfig, NextMolTrainingConfig
from torch_pharma.models.llm.mollama import init_mollama_tokenizer
from torch_pharma.tasks.molecule_generation.nextmol_dmt_task import NextMolDMTTask
from torch_pharma.training import Trainer

train_cfg = NextMolTrainingConfig(sampling_steps=100, pos_std=1.7226)
tokenizer = init_mollama_tokenizer(train_cfg.llm_model)
_, val_loader = build_nextmol_dataloaders(
    dataset_name="QM9-df",
    selfies_tokenizer=tokenizer,
    batch_size=4,
    mode="dmt",
)

task = NextMolDMTTask(train_cfg, DGTDiffusionConfig.dmt_b())
task.setup(torch.device("cuda"))
# load checkpoint here if needed (see Step 4)

trainer = Trainer()
samples = trainer.predict(task, val_loader)
# samples: list of sampled 3D coordinates per batch
```

## Step 6: Geom-DRUGS (large molecules)

After installing geom_drugs data (omit `--skip-geom`):

```python
from torch_pharma.data.datamodules.nextmol_dm import build_nextmol_dataloaders
from torch_pharma.models.diffusion.config import NextMolTrainingConfig
from torch_pharma.models.llm.mollama import init_mollama_tokenizer
from torch_pharma.paths import NEXTMOL_GEOM_DRUGS

train_cfg = NextMolTrainingConfig(pos_std=2.4777)  # Geom-DRUGS std
tokenizer = init_mollama_tokenizer(train_cfg.llm_model)
train_loader, val_loader = build_nextmol_dataloaders(
    root=NEXTMOL_GEOM_DRUGS,
    dataset_name="geom-drugs",
    selfies_tokenizer=tokenizer,
    batch_size=8,
    mode="dmt",
)
```

Wire into `NextMolDMTTask` + `Trainer` the same way as QM9 examples.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError` for `data_qm9.pt` | Data not installed | Run `scripts/install_nextmol_data.py` |
| HF model download fails | Network / auth | `huggingface-cli login` or set `HF_TOKEN` |
| CUDA OOM | Batch too large | `--batch_size 4` or `precision="32"` in Trainer |
| `val_loss` is NaN | lr / mixed precision | Lower `--batch_size`, try `precision="32"` |
| Missing unseen SELFIES tokens | Stale tokenizer | `add_unseen_selfies_tokens` runs automatically in dataloaders |
| Checkpoint key mismatch | Lightning vs native | Use `load_nextmol_dmt_checkpoint(..., strict=False)` |

## Related Docs

- [09-training-workflows.md](09-training-workflows.md) — workflow overview and hyperparameters
- [10-evaluation-parity.md](10-evaluation-parity.md) — metric definitions and benchmark targets
- [07-data-pipeline.md](07-data-pipeline.md) — collaters and dataset layout
- [13-checkpoint-migration.md](13-checkpoint-migration.md) — OSF weight loading
