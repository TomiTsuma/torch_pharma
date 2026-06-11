# Evaluation Parity

Metrics live in `torch_pharma/evaluation/nextmol/`. For runnable end-to-end examples (validate, sample, score), see **[14-examples-and-runbook.md](14-examples-and-runbook.md)**.

## Conformer Metrics

`torch_pharma/evaluation/nextmol/conformer_metrics.py`:

- RMSD via `AllChem.GetBestRMS`
- COV-R / MAT-R (simplified validation implementation)

### Published Benchmarks (QM9)

| Metric | NExT-Mol reported |
|--------|-------------------|
| Mean RMSD | ~0.104 Å |
| COV-R @ 0.5Å | high recall |

### Example: COV-R on predictions

```python
from rdkit import Chem

from torch_pharma.evaluation.nextmol.conformer_metrics import (
    compute_rmsd,
    conformer_recall,
    set_rdmol_positions,
)

# Build (smiles, mol) pairs with GT conformer (id 0) and predicted conformer (id 1)
predictions = []
for smiles, gt_coords, pred_coords in eval_pairs:
    mol = Chem.MolFromSmiles(smiles)
    mol = set_rdmol_positions(mol, gt_coords, add_conformer=True)
    mol = set_rdmol_positions(mol, pred_coords, add_conformer=True)
    predictions.append((smiles, mol))

metrics = conformer_recall(predictions, threshold=0.5)
# metrics: cov_mean, mat_mean, cov_median, mat_median
```

Pairwise RMSD:

```python
rmsd = compute_rmsd(mol_pred, mol_gt)  # returns inf on failure
```

## 2D Generation Metrics

`torch_pharma/evaluation/nextmol/generation_metrics.py`:

- Validity rate (RDKit parseable)
- Uniqueness rate
- Novelty rate (vs training set)

### Example: score generated SMILES

```python
from torch_pharma.evaluation.nextmol.generation_metrics import (
    novelty_rate,
    uniqueness_rate,
    validity_rate,
)

generated = ["CCO", "CC", "invalid", "CCO"]  # example outputs
train_set = {"C", "CC", "CCC"}

validity_rate(generated)      # fraction RDKit-parseable
uniqueness_rate(generated)    # unique / total
novelty_rate(generated, train_set)  # not in training set
```

## Validation Loop (loss-based)

Use `Trainer.validate` during or after training — no separate `scripts/evaluate.py` workflow yet:

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
trainer = Trainer()

# Load trained weights before validate (native or migrated checkpoint)
# task.load_state_dict(...) or load_nextmol_dmt_checkpoint(task.model, path)

task.setup(torch.device("cuda"))
metrics = trainer.validate(task, val_loader)
print(metrics)  # val_loss, val_lm_loss, val_diff_loss
```

## Sampling + Evaluation Pipeline

```python
import torch
from torch_pharma.data.datamodules.nextmol_dm import build_nextmol_dataloaders
from torch_pharma.models.diffusion.config import DGTDiffusionConfig, NextMolTrainingConfig
from torch_pharma.models.llm.mollama import init_mollama_tokenizer
from torch_pharma.paths import NEXTMOL_PRETRAINED
from torch_pharma.models.diffusion.checkpoint_migration import load_nextmol_dmt_checkpoint
from torch_pharma.tasks.molecule_generation.nextmol_dmt_task import NextMolDMTTask
from torch_pharma.training import Trainer

train_cfg = NextMolTrainingConfig(sampling_steps=100, pos_std=1.7226)
tokenizer = init_mollama_tokenizer(train_cfg.llm_model)
_, val_loader = build_nextmol_dataloaders(
    dataset_name="QM9-df",
    selfies_tokenizer=tokenizer,
    batch_size=8,
    mode="dmt",
)

task = NextMolDMTTask(train_cfg, DGTDiffusionConfig.dmt_b())
task.setup(torch.device("cuda"))
load_nextmol_dmt_checkpoint(task.model, NEXTMOL_PRETRAINED / "your_dmt.ckpt", strict=False)

trainer = Trainer()
sampled_positions = trainer.predict(task, val_loader)

# Convert positions → RDKit mols → conformer_recall / generation_metrics
```

## Full JODO Stack

Not fully ported. For publication-grade numbers, extend `evaluation/nextmol/` with NExT-Mol `evaluation/jodo/` modules.

## Test Procedure (parity checklist)

1. Install data: `py -3.11 scripts/install_nextmol_data.py`
2. Load checkpoint via `load_nextmol_dmt_checkpoint` (OSF) or native `state_dict`
3. Run `Trainer.predict(task, val_loader)` or `NextMolDMTTask.predict_step` per batch
4. Build RDKit mols with `set_rdmol_positions`
5. Compute `conformer_recall(predictions, threshold=0.5)` for 3D
6. Compute `validity_rate` / `uniqueness_rate` / `novelty_rate` for 2D SELFIES→SMILES outputs
7. Compare against NExT-Mol reported QM9 numbers (RMSD ~0.104 Å)

## Smoke Tests

```bash
pytest tests/test_nextmol_imports.py tests/test_paths.py -q
```

Verifies DMT, scheduler, Trainer, task imports, and path constants without GPU data.
