# Molecules API

Core molecular representation and chemistry utilities.

---

## Overview

The `molecules` module provides the fundamental abstractions for working with molecular data in Torch Pharma.

```python
from torch_pharma.molecules import (
    Molecule,
    AtomFeaturizer,
    BondFeaturizer,
    build_molecule,
    mol2smiles,
)
```

---

## Molecule

The `Molecule` class is the core data structure representing a chemical compound.

```python
class Molecule:
    """
    Core molecular representation.
    
    Provides methods for parsing SMILES, accessing molecular graphs,
    and converting to various formats.
    """
```

### Class Methods

#### `from_smiles`

```python
@classmethod
def from_smiles(
    cls,
    smiles: str,
    sanitize: bool = True,
    add_hydrogens: bool = False
) -> "Molecule"
```

Create a Molecule from a SMILES string.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `smiles` | `str` | SMILES representation of molecule |
| `sanitize` | `bool` | Whether to sanitize the molecule |
| `add_hydrogens` | `bool` | Whether to add explicit hydrogens |

**Returns:**

- `Molecule`: The parsed molecule

**Example:**

```python
from torch_pharma.molecules import Molecule

# Parse aspirin
mol = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O")

# Parse with explicit hydrogens
mol = Molecule.from_smiles("CCO", add_hydrogens=True)
```

#### `from_rdkit`

```python
@classmethod
def from_rdkit(cls, rdmol: Chem.Mol) -> "Molecule"
```

Create a Molecule from an RDKit Mol object.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `rdmol` | `Chem.Mol` | RDKit molecule object |

---

### Properties

#### `num_nodes`

```python
@property
def num_nodes(self) -> int
```

Number of atoms in the molecule.

#### `num_edges`

```python
@property
def num_edges(self) -> int
```

Number of bonds in the molecule.

#### `node_features`

```python
@property
def node_features(self) -> torch.Tensor
```

Atom feature matrix of shape `[num_nodes, num_features]`.

#### `edge_index`

```python
@property
def edge_index(self) -> torch.Tensor
```

Edge connectivity of shape `[2, num_edges]` in COO format.

#### `edge_features`

```python
@property
def edge_features(self) -> torch.Tensor
```

Edge feature matrix of shape `[num_edges, num_edge_features]`.

#### `positions`

```python
@property
def positions(self) -> Optional[torch.Tensor]
```

3D coordinates of shape `[num_nodes, 3]` if available, otherwise `None`.

---

### Methods

#### `to_smiles`

```python
def to_smiles(self, canonical: bool = True) -> str
```

Convert the molecule to a SMILES string.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `canonical` | `bool` | Whether to return canonical SMILES |

**Returns:**

- `str`: SMILES representation

#### `to_rdkit`

```python
def to_rdkit(self) -> Chem.Mol
```

Convert the molecule to an RDKit Mol object.

**Returns:**

- `Chem.Mol`: RDKit molecule

---

## AtomFeaturizer

Extract features from atoms in a molecule.

```python
class AtomFeaturizer:
    """
    Extracts features from atoms for graph neural networks.
    
    Supports customizable feature sets including atomic number,
    degree, formal charge, hybridization, aromaticity, etc.
    """
```

### Constructor

```python
def __init__(
    self,
    features: List[str] = ["atomic_number", "degree", "formal_charge"],
    allowable_sets: Optional[Dict[str, List]] = None
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `features` | `List[str]` | List of features to extract |
| `allowable_sets` | `Optional[Dict]` | Custom value sets for categorical features |

### Methods

#### `featurize`

```python
def featurize(self, atom: Chem.Atom) -> torch.Tensor
```

Extract features from a single atom.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `atom` | `Chem.Atom` | RDKit atom object |

**Returns:**

- `torch.Tensor`: Feature vector

#### `__call__`

```python
def __call__(self, mol: Chem.Mol) -> torch.Tensor
```

Extract features from all atoms in a molecule.

**Returns:**

- `torch.Tensor`: Feature matrix of shape `[num_atoms, num_features]`

---

## BondFeaturizer

Extract features from bonds in a molecule.

```python
class BondFeaturizer:
    """
    Extracts features from bonds for graph neural networks.
    
    Supports bond type, conjugation, ring membership, and stereo configuration.
    """
```

### Constructor

```python
def __init__(
    self,
    features: List[str] = ["bond_type", "conjugated", "ring"],
    self_loops: bool = False
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `features` | `List[str]` | List of features to extract |
| `self_loops` | `bool` | Whether to include self-loop features |

### Methods

#### `featurize`

```python
def featurize(self, bond: Chem.Bond) -> torch.Tensor
```

Extract features from a single bond.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `bond` | `Chem.Bond` | RDKit bond object |

---

## Chemistry Functions

### build_molecule

```python
@typechecked
def build_molecule(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    dataset_info: Dict[str, Any],
    charges: Optional[TensorType["num_nodes"]] = None,
    add_coords: bool = False,
    use_openbabel: bool = False
) -> Chem.RWMol:
```

Build an RDKit molecule from atomic positions and types.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `positions` | `TensorType["num_nodes", 3]` | Atomic positions [N, 3] |
| `atom_types` | `TensorType["num_nodes"]` | Atom type indices [N] |
| `dataset_info` | `Dict[str, Any]` | Dataset metadata including atom_decoder |
| `charges` | `Optional[TensorType["num_nodes"]]` | Atomic charges |
| `add_coords` | `bool` | Add conformer to molecule |
| `use_openbabel` | `bool` | Use OpenBabel for bond perception |

**Returns:**

- `Chem.RWMol`: RDKit molecule

**Example:**

```python
import torch
from torch_pharma.molecules import build_molecule

positions = torch.randn(5, 3)  # 5 atoms
atom_types = torch.tensor([0, 1, 1, 2, 3])  # C, O, O, N, H
dataset_info = {"atom_decoder": ["C", "O", "N", "H", "F"]}

mol = build_molecule(positions, atom_types, dataset_info)
```

---

### mol2smiles

```python
@typechecked
def mol2smiles(mol: Chem.Mol) -> Optional[str]:
```

Convert an RDKit molecule to SMILES string.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `mol` | `Chem.Mol` | RDKit molecule |

**Returns:**

- `Optional[str]`: SMILES string or `None` if invalid

**Example:**

```python
from rdkit import Chem
from torch_pharma.molecules import mol2smiles

mol = Chem.MolFromSmiles("CCO")
smiles = mol2smiles(mol)  # "CCO"
```

---

### process_molecule

```python
@typechecked
def process_molecule(
    rdmol: Chem.Mol,
    add_hydrogens: bool = False,
    sanitize: bool = False,
    relax_iter: int = 0,
    largest_frag: bool = False
) -> Optional[Chem.Mol]:
```

Apply filters and transformations to an RDKit molecule.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `rdmol` | `Chem.Mol` | Input molecule |
| `add_hydrogens` | `bool` | Add explicit hydrogens |
| `sanitize` | `bool` | Sanitize molecule |
| `relax_iter` | `int` | UFF optimization iterations |
| `largest_frag` | `bool` | Keep only largest fragment |

**Returns:**

- `Optional[Chem.Mol]`: Processed molecule or `None` if invalid

**Example:**

```python
from torch_pharma.molecules import process_molecule

# Sanitize and optimize
mol = process_molecule(
    rdmol,
    sanitize=True,
    relax_iter=200
)
```

---

## Graph Representation

The `graph` submodule provides utilities for working with molecular graphs.

```python
from torch_pharma.molecules.graph import (
    smiles_to_graph,
    batch_molecules,
    unbatch_molecules,
)
```

### smiles_to_graph

```python
def smiles_to_graph(
    smiles: str,
    featurizer: Optional[AtomFeaturizer] = None,
    bond_featurizer: Optional[BondFeaturizer] = None
) -> Data:
```

Convert a SMILES string to a PyTorch Geometric `Data` object.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `smiles` | `str` | SMILES string |
| `featurizer` | `Optional[AtomFeaturizer]` | Atom featurizer |
| `bond_featurizer` | `Optional[BondFeaturizer]` | Bond featurizer |

**Returns:**

- `Data`: PyTorch Geometric data object

---

## Feature Specifications

### Atom Features

| Feature | Type | Values |
|---------|------|--------|
| `atomic_number` | One-hot | Elements H-Pu |
| `degree` | One-hot | 0-6 |
| `formal_charge` | One-hot | -2 to +3 |
| `chiral_tag` | One-hot | None, R, S |
| `num_Hs` | One-hot | 0-4 |
| `hybridization` | One-hot | SP, SP2, SP3, etc. |
| `aromatic` | Binary | 0 or 1 |
| `mass` | Scalar | Atomic mass (fraction of C) |

### Bond Features

| Feature | Type | Values |
|---------|------|--------|
| `bond_type` | One-hot | Single, Double, Triple, Aromatic |
| `conjugated` | Binary | 0 or 1 |
| `ring` | Binary | 0 or 1 |
| `stereo` | One-hot | None, E, Z |

---

## Type Aliases

```python
from typing import Union
import torch
from rdkit import Chem

MolType = Union[str, Chem.Mol, Molecule]
FeatureVector = torch.Tensor
FeatureMatrix = torch.Tensor
```

---

## Exceptions

```python
from torch_pharma.molecules.exceptions import (
    MoleculeParseError,     # SMILES parsing failed
    InvalidMoleculeError,  # Invalid molecular structure
    FeaturizationError,    # Feature extraction failed
)
```
