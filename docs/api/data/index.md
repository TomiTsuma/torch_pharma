# Data API

Datasets, data loaders, and transforms.

---

## Overview

The `data` module provides utilities for loading and preprocessing molecular datasets.

```python
from torch_pharma.data import (
    QM9Dataset,
    ZINCDataset,
    DataLoader,
    SMILES2Graph,
)
```

---

## Datasets

### BaseDataset

Base class for all molecular datasets.

```python
class BaseDataset(Dataset):
    """
    Base class for molecular datasets.
    
    Provides common functionality for downloading, processing,
    and caching molecular datasets.
    """
    
    def __init__(self, root: Optional[str] = None):
        self.root = root or TORCH_PHARMA_HOME
        
    def download(self):
        """Download the dataset."""
        raise NotImplementedError
        
    def process(self):
        """Process raw data into PyTorch format."""
        raise NotImplementedError
```

---

### QM9Dataset

The QM9 dataset of molecular properties.

```python
class QM9Dataset(BaseDataset):
    """
    QM9 dataset with 130k molecules and 17 molecular properties.
    
    Properties include:
    - dipole moment (mu)
    - polarizability (alpha)
    - HOMO energy (homo)
    - LUMO energy (lumo)
    - gap (gap = lumo - homo)
    - spatial extent (R2)
    - zero point vibrational energy (zpve)
    - enthalpy (U0, U, H, G)
    - heat capacity (Cv)
    """
```

#### Constructor

```python
def __init__(
    self,
    root: Optional[str] = None,
    transform: Optional[Callable] = None,
    pre_transform: Optional[Callable] = None,
    calculate_thermo: bool = True,
    subset: Optional[List[int]] = None
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `root` | `Optional[str]` | Root directory for dataset |
| `transform` | `Optional[Callable]` | On-the-fly transform |
| `pre_transform` | `Optional[Callable]` | Pre-processing transform |
| `calculate_thermo` | `bool` | Calculate thermochemical properties |
| `subset` | `Optional[List[int]]` | Indices to use (for debugging) |

#### Properties

```python
@property
def num_features(self) -> int
# Number of atom features (default: 11)

@property
def num_classes(self) -> int
# Number of regression targets (17)
```

#### Methods

```python
def get(self, idx: int) -> Data
# Get molecule at index

def compute_smiles(self, remove_h: bool = False) -> List[str]
# Compute SMILES for all molecules
```

**Example:**

```python
from torch_pharma.data import QM9Dataset

# Load full QM9 dataset
dataset = QM9Dataset()

# Access a molecule
data = dataset[0]
print(data.smiles)        # SMILES string
print(data.y)            # Target properties
print(data.pos)          # 3D coordinates
```

---

### ZINCDataset

The ZINC database for molecular optimization.

```python
class ZINCDataset(BaseDataset):
    """
    ZINC database of commercially available compounds.
    
    Contains ~230M molecules filtered for drug-likeness.
    Useful for training generative models.
    """
```

#### Constructor

```python
def __init__(
    self,
    root: Optional[str] = None,
    subset: str = "250K",  # Options: 250K, 1M, full
    transform: Optional[Callable] = None
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `subset` | `str` | Subset size: "250K", "1M", or "full" |

---

### BindingDBDataset

Protein-ligand binding affinity data.

```python
class BindingDBDataset(BaseDataset):
    """
    BindingDB dataset with protein-ligand binding affinities.
    
    Contains Ki, Kd, and IC50 values for target-ligand pairs.
    """
```

#### Constructor

```python
def __init__(
    self,
    root: Optional[str] = None,
    affinity_type: str = "Ki",  # Options: Ki, Kd, IC50
    transform: Optional[Callable] = None
)
```

---

## Data Loaders

### DataLoader

Batch loading for molecular graphs.

```python
class DataLoader(torch_geometric.loader.DataLoader):
    """
    DataLoader for molecular graphs.
    
    Automatically handles batching of graphs with different sizes
    using the PyTorch Geometric batching strategy.
    """
```

#### Constructor

```python
def __init__(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `dataset` | `Dataset` | Dataset to load |
| `batch_size` | `int` | Number of samples per batch |
| `shuffle` | `bool` | Whether to shuffle data |
| `num_workers` | `int` | Number of parallel workers |

**Example:**

```python
from torch_pharma.data import QM9Dataset, DataLoader

dataset = QM9Dataset()
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for batch in loader:
    print(batch.x)          # Node features
    print(batch.edge_index)   # Edge connectivity
    print(batch.batch)        # Batch assignment
    print(batch.y)            # Targets
```

---

## Transforms

Transforms for preprocessing molecular data.

### SMILES2Graph

Convert SMILES strings to graph representation.

```python
class SMILES2Graph:
    """
    Transform SMILES strings to PyTorch Geometric Data objects.
    """
    
    def __init__(
        self,
        add_hydrogens: bool = False,
        featurizer: Optional[AtomFeaturizer] = None
    )
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `add_hydrogens` | `bool` | Add explicit hydrogens |
| `featurizer` | `Optional[AtomFeaturizer]` | Custom atom featurizer |

**Example:**

```python
from torch_pharma.data import SMILES2Graph

transform = SMILES2Graph(add_hydrogens=True)
data = transform("CCO")  # Ethanol as graph
```

---

### NormalizeFeatures

Normalize node and edge features.

```python
class NormalizeFeatures:
    """
    Normalize features using mean and std from training data.
    """
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor)
    
    def __call__(self, data: Data) -> Data
```

---

### AddSelfLoops

Add self-loops to molecular graphs.

```python
class AddSelfLoops:
    """
    Add self-loops (diagonal edges) to the graph.
    Useful for GCN and similar architectures.
    """
    
    def __call__(self, data: Data) -> Data
```

---

### Augment3D

Generate 3D conformers for molecules.

```python
class Augment3D:
    """
    Generate 3D conformers using RDKit.
    
    Adds 3D coordinates to molecules without them.
    """
    
    def __init__(
        self,
        num_conformers: int = 1,
        optimize: bool = True
    )
```

---

## Data Components

### EDM Components

Utilities for EDM (Equivariant Diffusion Model) datasets.

```python
from torch_pharma.data.components.edm import (
    retrieve_dataloaders,
    get_bond_order_batch,
    get_bond_length_arrays,
)
```

#### retrieve_dataloaders

```python
def retrieve_dataloaders(
    args,
    train_loader_only: bool = False
) -> Dict[str, DataLoader]:
```

Get data loaders for EDM training.

---

## Dataset Utilities

### get_dataset_info

```python
def get_dataset_info(
    dataset_name: str,
    remove_h: bool = False
) -> Dict[str, Any]:
```

Get metadata for a dataset.

**Returns:**

- `Dict` containing:
  - `name`: Dataset name
  - `atom_encoder`: Mapping from element to index
  - `atom_decoder`: Mapping from index to element
  - `with_h`: Whether dataset includes hydrogens

---

### download_qm9

```python
def download_qm9(
    root: str = TORCH_PHARMA_HOME,
    cleanup: bool = True
) -> None:
```

Download and extract the QM9 dataset.

---

## Custom Datasets

Creating a custom dataset:

```python
from torch_pharma.data import BaseDataset
from torch_geometric.data import Data

class MyDataset(BaseDataset):
    def __init__(self, smiles_list, targets, **kwargs):
        super().__init__(**kwargs)
        self.smiles = smiles_list
        self.targets = targets
        
    def __len__(self):
        return len(self.smiles)
    
    def get(self, idx):
        from torch_pharma.molecules import Molecule
        mol = Molecule.from_smiles(self.smiles[idx])
        
        return Data(
            x=mol.node_features,
            edge_index=mol.edge_index,
            y=self.targets[idx],
            smiles=self.smiles[idx]
        )
```

---

## Type Aliases

```python
from typing import Union, List
from torch_geometric.data import Data, Batch

DataOrBatch = Union[Data, Batch]
Transform = Callable[[Data], Data]
```
