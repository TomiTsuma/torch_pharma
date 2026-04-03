# API Reference

Complete API documentation for Torch Pharma.

---

## Package Structure

```python
import torch_pharma

# Molecules
from torch_pharma import molecules

# Data
from torch_pharma import data

# Models
from torch_pharma import models

# Tasks
from torch_pharma import tasks

# Training
from torch_pharma import training

# Evaluation
from torch_pharma import evaluation

# Reinforcement Learning
from torch_pharma import rl

# Utilities
from torch_pharma import utils
```

---

## Module Overview

### [Molecules](molecules/index.md)

Molecular representation and chemistry utilities.

| Class/Function | Description |
|---------------|-------------|
| `Molecule` | Core molecular representation |
| `AtomFeaturizer` | Atom feature extraction |
| `BondFeaturizer` | Bond feature extraction |
| `build_molecule` | Build RDKit molecule from tensors |
| `mol2smiles` | Convert molecule to SMILES |

### [Data](data/index.md)

Datasets, data loaders, and transforms.

| Class/Function | Description |
|---------------|-------------|
| `QM9Dataset` | QM9 molecular dataset |
| `ZINCDataset` | ZINC database |
| `BindingDBDataset` | Binding affinity data |
| `DataLoader` | Batch loading |
| `SMILES2Graph` | SMILES to graph transform |

### [Models](models/index.md)

Neural network architectures for molecules.

| Class | Description |
|-------|-------------|
| `GCN` | Graph Convolutional Network |
| `GAT` | Graph Attention Network |
| `MPNN` | Message Passing Neural Network |
| `EGNN` | Equivariant Graph Neural Network |
| `GCPNet` | Geometric Clifford Perceptron Network |
| `GraphTransformer` | Transformer for graphs |
| `EDMDiffusion` | EDM diffusion model |

### [Tasks](tasks/index.md)

Task definitions and training logic.

| Class | Description |
|-------|-------------|
| `PropertyPredictionTask` | Molecular property prediction |
| `MoleculeGenerationTask` | De novo molecule generation |
| `BindingAffinityTask` | Protein-ligand binding |
| `ToxicityPredictionTask` | ADMET prediction |

### [Training](training/index.md)

Training engine and utilities.

| Class | Description |
|-------|-------------|
| `Trainer` | Main training loop |
| `Callback` | Training hooks base class |
| `ModelCheckpoint` | Save model checkpoints |
| `EarlyStopping` | Early stopping callback |

### [Reinforcement Learning](rl/index.md)

RL components for molecule optimization.

| Class | Description |
|-------|-------------|
| `MoleculeEnv` | Molecular editing environment |
| `PPOAgent` | PPO RL agent |
| `DQNAgent` | DQN RL agent |
| `SACAgent` | SAC RL agent |

### [Evaluation](evaluation/index.md)

Metrics and benchmarking.

| Class | Description |
|-------|-------------|
| `BasicMolecularMetrics` | Validity, uniqueness, novelty |
| `ScoringFunction` | Property scoring |
| `Benchmark` | Standardized evaluation |

---

## Type Annotations

Torch Pharma uses type hints throughout:

```python
from torchtyping import TensorType
from typeguard import typechecked

@typechecked
def build_molecule(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    dataset_info: Dict[str, Any]
) -> Chem.RWMol:
    ...
```

---

## Configuration Classes

Most components accept configuration dictionaries:

```python
config = {
    "model": {
        "name": "GCN",
        "hidden_channels": 128,
        "num_layers": 3
    },
    "optimizer": {
        "name": "Adam",
        "lr": 1e-3
    }
}
```

---

## Exception Handling

Torch Pharma defines custom exceptions:

```python
from torch_pharma.exceptions import (
    MoleculeParseError,    # SMILES parsing failed
    InvalidMoleculeError,  # Invalid molecular structure
    ConfigurationError,   # Invalid configuration
    TrainingError,        # Training failure
)
```

---

## Version Information

```python
import torch_pharma

print(torch_pharma.__version__)  # "0.1.0"
```
