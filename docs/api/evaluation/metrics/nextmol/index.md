# NextMol Evaluation Metrics

Metrics for evaluating NExT-Mol generation quality.

---

## Overview

The `nextmol` evaluation module provides metrics for 2D/3D molecular generation:

```python
from torch_pharma.evaluation.nextmol import (
    compute_rmsd,
    conformer_recall,
    validity_rate,
    uniqueness_rate,
)
```

---

## 2D Generation Metrics

### validity_rate

Calculate percentage of chemically valid SMILES strings.

```python
from torch_pharma.evaluation.nextmol import validity_rate

smiles_list = ["CCO", "CC(=O)O", "invalid"]
rate = validity_rate(smiles_list)
# Returns: 0.667 (2/3 valid)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `smiles_list` | `Iterable[str]` | List of SMILES strings |

**Returns:**

- `float`: Validity rate between 0 and 1

---

### uniqueness_rate

Calculate percentage of unique molecules.

```python
from torch_pharma.evaluation.nextmol import uniqueness_rate

smiles_list = ["CCO", "CCO", "CC(=O)O"]
rate = uniqueness_rate(smiles_list)
# Returns: 0.667 (2/3 unique)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `smiles_list` | `Iterable[str]` | List of SMILES strings |

**Returns:**

- `float`: Uniqueness rate between 0 and 1

---

### novelty_rate

Calculate percentage of novel molecules (not in training set).

```python
from torch_pharma.evaluation.nextmol import novelty_rate

generated = ["CCO", "CC(=O)O", "c1ccccc1"]
training_set = {"CCO", "c1ccccc1"}
rate = novelty_rate(generated, training_set)
# Returns: 0.333 (1/3 novel)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `smiles_list` | `Iterable[str]` | Generated SMILES |
| `training_set` | `Set[str]` | Training set SMILES |

**Returns:**

- `float`: Novelty rate between 0 and 1

---

## 3D Conformer Metrics

### compute_rmsd

Compute RMSD between two molecular conformers.

```python
from torch_pharma.evaluation.nextmol import compute_rmsd
from rdkit import Chem

mol_pred = Chem.Mol(mol_with_pred_coords)
mol_gt = Chem.Mol(mol_with_gt_coords)

rmsd = compute_rmsd(mol_pred, mol_gt)
# Returns: float RMSD value
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `mol_pred` | `Chem.Mol` | Predicted molecule with conformer |
| `mol_gt` | `Chem.Mol` | Ground truth molecule with conformer |

**Returns:**

- `float`: RMSD value (inf if alignment fails)

---

### conformer_recall

Compute COV-R metric for conformer generation.

```python
from torch_pharma.evaluation.nextmol import conformer_recall

predictions = [
    ("mol1", mol_with_2_conformers),
    ("mol2", mol_with_2_conformers),
]
metrics = conformer_recall(predictions, threshold=0.5)
# Returns: {"cov_mean": 0.9, "mat_mean": 0.3, ...}
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `predictions` | `List[Tuple[str, Chem.Mol]]` | Molecules with multiple conformers |
| `threshold` | `float` | RMSD threshold for coverage |

**Returns:**

- `dict`: Metrics including:
  - `cov_mean`: Mean coverage (RMSD < threshold)
  - `mat_mean`: Mean RMSD
  - `cov_median`: Median coverage
  - `mat_median`: Median RMSD

---

## Complete Evaluation Example

```python
from torch_pharma.evaluation.nextmol import (
    validity_rate,
    uniqueness_rate,
    compute_rmsd,
    conformer_recall,
)
from rdkit import Chem

# Generate molecules
smiles_list = model.generate(n_samples=1000)

# 2D metrics
validity = validity_rate(smiles_list)
uniqueness = uniqueness_rate(smiles_list)

print(f"Validity: {validity:.3f}")
print(f"Uniqueness: {uniqueness:.3f}")

# 3D metrics
conformers = model.generate_conformers(smiles_list[:100])
metrics = conformer_recall(conformers, threshold=0.5)

print(f"COV-R (threshold=0.5): {metrics['cov_mean']:.3f}")
print(f"Mean RMSD: {metrics['mat_mean']:.3f}")
```
