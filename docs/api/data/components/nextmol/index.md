# NextMol Data Components

Data pipeline components for NExT-Mol (Molecular Diffusion with Language Models).

---

## Overview

The `nextmol` module provides data components for training diffusion models with LLM conditioning:

```python
from torch_pharma.data.components.nextmol import (
    QM9Collater,
    QM9InferCollater,
    LMCollater,
    NextMolBatch,
    get_dataset_info,
)
```

---

## Datasets

### QM9TorDFDataset

QM9 torsional diffusion dataset for conformer generation.

```python
from torch_pharma.data.components.nextmol.datasets import QM9TorDFDataset

dataset = QM9TorDFDataset(
    root="path/to/data",
    split="train",
    rand_smiles=False,
    addHs=False
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `root` | `Optional[str]` | Root directory for dataset |
| `split` | `str` | Dataset split: "train", "val", "test" |
| `rand_smiles` | `str | bool` | Randomize SMILES: False, "canonical", "restricted" |
| `addHs` | `bool` | Include hydrogens |

### QM9LMDataset

QM9 SELFIES language modeling dataset.

```python
from torch_pharma.data.components.nextmol.datasets import QM9LMDataset

dataset = QM9LMDataset(
    root="path/to/data",
    split="train",
    transform=None
)
```

### GeomDrugsTorDFDataset

Geom-DRUGS dataset for large molecule conformer generation.

```python
from torch_pharma.data.components.nextmol.datasets import GeomDrugsTorDFDataset

dataset = GeomDrugsTorDFDataset(
    root="path/to/data",
    split="train",
    rand_smiles=False,
    addHs=False
)
```

---

## Collators

### QM9Collater

Batch collater with VP noise injection for DMT training.

```python
from torch_pharma.data.components.nextmol import QM9Collater
from torch_pharma.models.diffusion.vp_scheduler import NoiseScheduleVPV2

scheduler = NoiseScheduleVPV2(schedule="cosine")
collater = QM9Collater(
    max_atoms=29,
    max_sf_tokens=256,
    selfies_tokenizer=tokenizer,
    noise_scheduler=scheduler,
    aug_rotation=False,
    t_cond="t",
    disable_com=False,
    aug_translation=False,
    load_mapping=True
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `max_atoms` | `int` | Maximum number of atoms per molecule |
| `max_sf_tokens` | `int` | Maximum SELFIES tokens |
| `selfies_tokenizer` | `Tokenizer` | SELFIES tokenizer |
| `noise_scheduler` | `NoiseScheduleVPV2` | VP noise scheduler |
| `aug_rotation` | `bool` | Apply rotational augmentation |
| `t_cond` | `str` | Time conditioning: "t" or "noise_level" |
| `disable_com` | `bool` | Disable center-of-mass removal |
| `aug_translation` | `bool` | Apply translational augmentation |
| `load_mapping` | `bool` | Load atom-to-SELFIES mapping |

### QM9InferCollater

Inference collater without noise schedule.

```python
from torch_pharma.data.components.nextmol import QM9InferCollater

collater = QM9InferCollater(
    max_atoms=29,
    max_sf_tokens=256,
    selfies_tokenizer=tokenizer,
    disable_com=False,
    load_mapping=True
)
```

### LMCollater

SELFIES-only collater for MoLlama training.

```python
from torch_pharma.data.components.nextmol import LMCollater

collater = LMCollater(
    max_sf_tokens=256,
    selfies_tokenizer=tokenizer
)
```

---

## Batch Container

### NextMolBatch

PyG graph batch with SELFIES tokenizer output.

```python
from torch_pharma.data.components.nextmol import NextMolBatch

batch = NextMolBatch(
    data_batch=graph_batch,
    selfies_batch=tokenizer_output
)

# Move to device
batch = batch.to(device)
```

---

## Configuration

### Dataset Info

```python
from torch_pharma.data.components.nextmol import get_dataset_info, QM9_DF_CONFIG, GEOM_DRUGS_CONFIG

# Get dataset metadata
info = get_dataset_info("QM9-df")
# Returns: {"name": str, "pos_std": float, "max_atoms": int, ...}

# Predefined configs
qm9_config = QM9_DF_CONFIG
geom_config = GEOM_DRUGS_CONFIG
```

---

## Path Constants

```python
from torch_pharma.paths import (
    NEXTMOL_DATA,
    NEXTMOL_QM9_LM,
    NEXTMOL_QM9_TORDF,
    NEXTMOL_GEOM_DRUGS,
    NEXTMOL_CHECKPOINTS,
    NEXTMOL_PRETRAINED,
)
```

---

## Data Pipeline Example

```python
from torch_pharma.data.components.nextmol import (
    build_nextmol_dataloaders,
    QM9Collater,
)
from torch_pharma.models.diffusion.vp_scheduler import NoiseScheduleVPV2

# Build dataloaders for DMT training
train_loader, val_loader = build_nextmol_dataloaders(
    root="data/nextmol",
    dataset_name="QM9-df",
    selfies_tokenizer=tokenizer,
    batch_size=32,
    num_workers=4,
    mode="dmt",
    noise_scheduler="cosine"
)

# Iterate through data
for data_batch, selfies_batch in train_loader:
    # data_batch: PyG Batch with 3D coordinates
    # selfies_batch: Tokenizer output with input_ids, attention_mask
    pass
```

---

## Mappings

### Atom-to-SELFIES Mapping

The `mol_mapping` module provides utilities for aligning RDKit atoms with SELFIES tokens:

```python
from torch_pharma.data.components.nextmol.mol_mapping import (
    get_smiles2selfies_mapping,
    build_rdkit2cano_smiles_withoutH_mapping,
    build_rdkit2rand_smiles_withoutH_mapping,
)
```

---

## Utilities

### Property Normalizations

```python
from torch_pharma.data.components.nextmol.dataset_config import PROPERTY_NORMALIZATIONS

# Normalization for QM9 properties
mu_norm = PROPERTY_NORMALIZATIONS["mu"]  # {"mean": 2.6726, "mad": 1.0339}
```
