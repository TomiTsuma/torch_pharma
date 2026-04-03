# Evaluation API

Metrics and benchmarking for molecular ML.

---

## Overview

The `evaluation` module provides comprehensive metrics for assessing model performance in molecular machine learning tasks.

```python
from torch_pharma.evaluation import (
    BasicMolecularMetrics,
    ScoringFunction,
    Benchmark,
)
```

---

## Molecular Metrics

### BasicMolecularMetrics

Core metrics for molecule generation quality.

```python
class BasicMolecularMetrics:
    """
    Compute validity, uniqueness, and novelty of generated molecules.
    
    These are the standard metrics used to evaluate generative models
    for molecular design.
    """
```

#### Constructor

```python
def __init__(
    self,
    dataset_info: Dict[str, Any],
    data_dir: str,
    dataset_smiles_list: Optional[np.ndarray] = None
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `dataset_info` | `Dict[str, Any]` | Dataset metadata |
| `data_dir` | `str` | Path to dataset directory |
| `dataset_smiles_list` | `Optional[np.ndarray]` | Training set SMILES |

#### Methods

##### compute_validity

```python
def compute_validity(
    self,
    rdmols: List[Chem.RWMol]
) -> Tuple[List[str], float]
```

Calculate percentage of chemically valid molecules.

**Returns:**

- `Tuple[List[str], float]`: Valid SMILES and validity score

**Example:**

```python
from torch_pharma.evaluation import BasicMolecularMetrics

metrics = BasicMolecularMetrics(dataset_info, data_dir)
valid_smiles, validity = metrics.compute_validity(generated_mols)
print(f"Validity: {validity * 100:.2f}%")
```

---

##### compute_uniqueness

```python
def compute_uniqueness(
    self,
    valid: List[str]
) -> Tuple[List[str], float]
```

Calculate percentage of unique molecules.

**Returns:**

- `Tuple[List[str], float]`: Unique SMILES and uniqueness score

---

##### compute_novelty

```python
def compute_novelty(
    self,
    unique: List[str]
) -> Tuple[List[str], float]
```

Calculate percentage of novel molecules (not in training set).

**Returns:**

- `Tuple[List[str], float]`: Novel SMILES and novelty score

---

##### evaluate_rdmols

```python
def evaluate_rdmols(
    self,
    rdmols: List[Chem.RWMol],
    verbose: bool = True
) -> List[float]
```

Compute all metrics at once.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `rdmols` | `List[Chem.RWMol]` | Generated molecules |
| `verbose` | `bool` | Print results |

**Returns:**

- `List[float]`: [validity, uniqueness, novelty]

**Example:**

```python
scores = metrics.evaluate_rdmols(generated_mols, verbose=True)
# Validity over 100 molecules: 95.00%
# Uniqueness over 95 valid molecules: 98.95%
# Novelty over 94 unique molecules: 92.55%
```

---

## Scoring Functions

### ScoringFunction

Property-based scoring for molecules.

```python
class ScoringFunction:
    """
    Compute chemical property scores.
    
    Includes QED, LogP, SA score, and other drug-likeness metrics.
    """
```

#### Methods

##### qed_score

```python
@staticmethod
def qed_score(mol: Chem.Mol) -> float
```

Calculate Quantitative Estimate of Drug-likeness.

**Returns:**

- `float`: QED score between 0 and 1

##### logp_score

```python
@staticmethod
def logp_score(mol: Chem.Mol) -> float
```

Calculate partition coefficient (LogP).

##### sa_score

```python
@staticmethod
def sa_score(mol: Chem.Mol) -> float
```

Calculate synthetic accessibility score.

**Returns:**

- `float`: SA score between 1 (easy) and 10 (hard)

##### diversity_score

```python
@staticmethod
def diversity_score(mols: List[Chem.Mol]) -> float
```

Calculate internal diversity using Tanimoto similarity.

---

## Benchmarks

### Benchmark

Standardized evaluation protocol.

```python
class Benchmark:
    """
    Standardized benchmark for molecular generation models.
    
    Evaluates models on predefined datasets with consistent metrics.
    """
```

#### Constructor

```python
def __init__(
    self,
    name: str,
    dataset: str,
    metrics: List[str],
    n_samples: int = 10000
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Benchmark name |
| `dataset` | `str` | Dataset to use ("qm9", "zinc", "guacamol") |
| `metrics` | `List[str]` | Metrics to compute |
| `n_samples` | `int` | Number of molecules to generate |

#### Methods

##### evaluate

```python
def evaluate(
    self,
    model: nn.Module,
    **sampling_kwargs
) -> Dict[str, float]
```

Run benchmark evaluation.

**Example:**

```python
from torch_pharma.evaluation import Benchmark

benchmark = Benchmark(
    name="qm9_generation",
    dataset="qm9",
    metrics=["validity", "uniqueness", "novelty", "qed"]
)

results = benchmark.evaluate(model, n_samples=10000)
print(results)
# {'validity': 0.95, 'uniqueness': 0.98, 'novelty': 0.92, 'qed': 0.45}
```

---

## Property Metrics

### PropertyMetrics

Metrics for property prediction tasks.

```python
class PropertyMetrics:
    """
    Metrics for regression and classification tasks.
    """
    
    @staticmethod
    def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float
    
    @staticmethod
    def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float
    
    @staticmethod
    def r2(predictions: torch.Tensor, targets: torch.Tensor) -> float
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float
    
    @staticmethod
    def roc_auc(predictions: torch.Tensor, targets: torch.Tensor) -> float
```

---

## PoseBusters Integration

### PoseBusterMetrics

Evaluate generated molecular poses.

```python
class PoseBusterMetrics:
    """
    Validate generated 3D conformations.
    
    Uses PoseBusters to check stereochemistry,
    bond lengths, and clash detection.
    """
    
    def __init__(self)
    
    def validate(self, mol: Chem.Mol) -> Dict[str, bool]
    """
    Validate a molecule's 3D structure.
    
    Returns:
        Dictionary of validation results
    """
```

---

## Complete Evaluation Example

```python
from torch_pharma.evaluation import (
    BasicMolecularMetrics,
    ScoringFunction,
    PropertyMetrics
)
from rdkit import Chem

# Generate molecules
molecules = model.sample(n_samples=1000)

# Convert to RDKit
rdmols = [Chem.MolFromSmiles(s) for s in molecules]

# Basic metrics
metrics = BasicMolecularMetrics(dataset_info, data_dir)
validity, uniqueness, novelty = metrics.evaluate_rdmols(rdmols)

# Property scores
qed_scores = [ScoringFunction.qed_score(m) for m in rdmols if m is not None]
avg_qed = np.mean(qed_scores)

# Print results
print(f"Validity: {validity:.3f}")
print(f"Uniqueness: {uniqueness:.3f}")
print(f"Novelty: {novelty:.3f}")
print(f"Avg QED: {avg_qed:.3f}")
```
