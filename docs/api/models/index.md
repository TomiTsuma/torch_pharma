# Models API

Neural network architectures for molecular machine learning.

---

## Overview

The `models` module provides pre-built architectures for molecular tasks.

```python
from torch_pharma.models import (
    GCN,
    GAT,
    MPNN,
    EGNN,
    GraphTransformer,
    EDMDiffusion,
)
```

---

## GNN Models

### GCN

Graph Convolutional Network.

```python
class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2016).
    
    Implements the message passing:
    h_i^{l+1} = ReLU(sum_j (A_ij * h_j^l * W^l))
    
    where A is the normalized adjacency matrix.
    """
```

#### Constructor

```python
def __init__(
    self,
    in_channels: int,
    hidden_channels: int,
    num_layers: int = 2,
    out_channels: Optional[int] = None,
    dropout: float = 0.0,
    act: Callable = nn.ReLU(),
    norm: Optional[nn.Module] = None
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `in_channels` | `int` | Input feature dimension |
| `hidden_channels` | `int` | Hidden layer dimension |
| `num_layers` | `int` | Number of GCN layers |
| `out_channels` | `Optional[int]` | Output dimension (None for node-level) |
| `dropout` | `float` | Dropout rate |
| `act` | `Callable` | Activation function |
| `norm` | `Optional[nn.Module]` | Normalization layer |

**Example:**

```python
from torch_pharma.models import GCN

model = GCN(
    in_channels=11,      # QM9 atom features
    hidden_channels=128,
    num_layers=3,
    out_channels=1,      # For regression
    dropout=0.1
)

# Forward pass
out = model(x, edge_index, batch)
```

---

### GAT

Graph Attention Network.

```python
class GAT(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2017).
    
    Uses attention mechanisms to weight neighbor contributions:
    h_i^{l+1} = concat_k sum_j alpha_ij^k W^k h_j^l
    
    where alpha_ij is the attention coefficient.
    """
```

#### Constructor

```python
def __init__(
    self,
    in_channels: int,
    hidden_channels: int,
    num_layers: int = 2,
    out_channels: Optional[int] = None,
    heads: int = 1,
    concat: bool = False,
    dropout: float = 0.0,
    add_self_loops: bool = True
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `heads` | `int` | Number of attention heads |
| `concat` | `bool` | Concatenate heads (True) or average (False) |
| `add_self_loops` | `bool` | Add self-loops to graph |

---

### MPNN

Message Passing Neural Network.

```python
class MPNN(nn.Module):
    """
    Message Passing Neural Network (Gilmer et al., 2017).
    
    General framework for graph neural networks with:
    - Message function: m_ij = M(h_i, h_j, e_ij)
    - Update function: h_i' = U(h_i, sum_j m_ij)
    """
```

#### Constructor

```python
def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_dim: int,
    num_steps: int = 3,
    aggregation: str = "sum"  # or "mean", "max"
)
```

---

## Equivariant Models

### EGNN

Equivariant Graph Neural Network.

```python
class EGNN(nn.Module):
    """
    Equivariant Graph Neural Network (Satorras et al., 2022).
    
    Maintains E(n) equivariance - rotating input coordinates
    produces rotated output coordinates.
    
    Key equation:
    m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, a_ij)
    x_i' = x_i + sum_j (x_i - x_j) * phi_x(m_ij) / N
    h_i' = phi_h(h_i, sum_j m_ij)
    """
```

#### Constructor

```python
def __init__(
    self,
    in_node_nf: int,
    hidden_nf: int,
    out_node_nf: int,
    in_edge_nf: int = 0,
    device: str = 'cpu',
    act_fn: Callable = nn.SiLU(),
    n_layers: int = 4,
    attention: bool = False,
    tanh: bool = False,
    coords_range: float = 15.0,
    norm_constant: float = 1.0,
    inv_sublayers: int = 1,
    sin_embedding: bool = False,
    normalization_factor: float = 1.0,
    aggregation_method: str = 'sum'
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `in_node_nf` | `int` | Input node feature dimension |
| `hidden_nf` | `int` | Hidden feature dimension |
| `out_node_nf` | `int` | Output node feature dimension |
| `in_edge_nf` | `int` | Input edge feature dimension |
| `n_layers` | `int` | Number of EGNN layers |
| `attention` | `bool` | Use attention in message passing |
| `sin_embedding` | `bool` | Use sinusoidal distance embedding |

**Example:**

```python
from torch_pharma.models import EGNN

model = EGNN(
    in_node_nf=11,
    hidden_nf=128,
    out_node_nf=1,
    n_layers=4,
    attention=True
)

# Forward pass with coordinates
h, x = model(h, x, edges, edge_attr)
```

---

### GCPNet

Geometric Clifford Perceptron Network.

```python
class GCPNet(nn.Module):
    """
    Geometry-Complete Perceptron Network (Morehead et al., 2023).
    
    Uses geometric algebra for SE(3)-equivariant message passing.
    """
```

---

## Transformers

### GraphTransformer

Transformer architecture for graphs.

```python
class GraphTransformer(nn.Module):
    """
    Transformer for graph-structured data (Dwivedi & Bresson, 2020).
    
    Uses Laplacian positional encodings and multi-head attention
    over fully connected graph with edge gating.
    """
```

#### Constructor

```python
def __init__(
    self,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
    edge_dim: int = 0
)
```

---

### MolTransformer

Molecular Transformer for SMILES sequences.

```python
class MolTransformer(nn.Module):
    """
    Transformer for molecular SMILES sequences.
    
    Treats molecules as sequences of tokens rather than graphs.
    """
```

---

## Diffusion Models

### EDMDiffusion

Equivariant Diffusion Model for 3D molecule generation.

```python
class EDMDiffusion(nn.Module):
    """
    Equivariant Diffusion Model (Hoogeboom et al., 2022).
    
    Generates 3D molecular structures by denoising diffusion process.
    """
```

#### Constructor

```python
def __init__(
    self,
    n_atoms: int = 29,
    n_atom_types: int = 5,
    hidden_dim: int = 256,
    n_layers: int = 9,
    timesteps: int = 1000,
    noise_schedule: str = "cosine"
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `n_atoms` | `int` | Maximum number of atoms |
| `n_atom_types` | `int` | Number of atom types |
| `hidden_dim` | `int` | Hidden dimension |
| `n_layers` | `int` | Number of layers |
| `timesteps` | `int` | Diffusion timesteps |
| `noise_schedule` | `str` | "cosine" or "linear" |

**Methods:**

```python
def forward(self, x, h, t, edges)
# Forward diffusion step

def sample(self, n_samples, n_atoms)
# Generate new molecules
```

---

### VariationalDiffusion

Variational Diffusion Model.

```python
class VariationalDiffusion(nn.Module):
    """
    Variational diffusion with learned prior.
    """
```

---

## Protein Models

### ProteinEncoder

```python
class ProteinEncoder(nn.Module):
    """
    Encode protein structures for protein-ligand tasks.
    """
```

---

## Model Heads

### RegressionHead

```python
class RegressionHead(nn.Module):
    """
    Head for regression tasks.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_targets: int = 1,
        pooling: str = "mean"
    )
```

### ClassificationHead

```python
class ClassificationHead(nn.Module):
    """
    Head for classification tasks.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        pooling: str = "mean"
    )
```

---

## Model Utilities

### load_model

```python
def load_model(
    checkpoint_path: str,
    device: str = "cpu"
) -> nn.Module:
```

Load a model from a checkpoint.

### save_model

```python
def save_model(
    model: nn.Module,
    checkpoint_path: str
) -> None:
```

Save a model checkpoint.

---

## Base Classes

### BaseGNN

Base class for GNN models.

```python
class BaseGNN(nn.Module, ABC):
    """
    Abstract base class for graph neural networks.
    """
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass
```

---

## Type Aliases

```python
from typing import Union

Model = Union[
    GCN, GAT, MPNN,
    EGNN, GCPNet,
    GraphTransformer,
    EDMDiffusion
]
```
