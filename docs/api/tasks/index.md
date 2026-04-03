# Tasks API

Task definitions and training logic.

---

## Overview

Tasks encapsulate the training logic for specific molecular machine learning objectives.

```python
from torch_pharma.tasks import (
    PropertyPredictionTask,
    MoleculeGenerationTask,
    BindingAffinityTask,
    ToxicityPredictionTask,
)
```

---

## Base Task

### BaseTask

Abstract base class for all tasks.

```python
class BaseTask(nn.Module, ABC):
    """
    Base class for molecular ML tasks.
    
    Combines model, loss function, and optimizer into a single
    unit that can be trained with the Trainer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: type = torch.optim.Adam,
        lr: float = 1e-3,
        **optimizer_kwargs
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), lr=lr, **optimizer_kwargs)
```

#### Methods

```python
def training_step(
    self,
    batch: Data,
    batch_idx: int
) -> torch.Tensor:
    """
    Perform one training step.
    
    Args:
        batch: PyTorch Geometric Data object
        batch_idx: Index of current batch
    
    Returns:
        Loss value
    """
    
def validation_step(
    self,
    batch: Data,
    batch_idx: int
) -> Dict[str, Any]:
    """
    Perform one validation step.
    
    Returns:
        Dictionary of metrics
    """
    
def configure_optimizers(self) -> torch.optim.Optimizer:
    """
    Return optimizer(s) for training.
    """
```

---

## Property Prediction

### PropertyPredictionTask

Regression or classification on molecular properties.

```python
class PropertyPredictionTask(BaseTask):
    """
    Task for predicting molecular properties.
    
    Supports both single-task and multi-task prediction.
    Targets can be from QM9 (homo, lumo, etc.) or custom datasets.
    """
```

#### Constructor

```python
def __init__(
    self,
    model: nn.Module,
    target: Union[str, List[str]] = "homo",
    criterion: Optional[nn.Module] = None,
    metrics: Optional[List[Callable]] = None,
    task_type: str = "regression",  # or "classification"
    **optimizer_kwargs
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `model` | `nn.Module` | GNN model |
| `target` | `Union[str, List[str]]` | Target property name(s) |
| `criterion` | `Optional[nn.Module]` | Loss function (default: MSELoss) |
| `metrics` | `Optional[List[Callable]]` | Evaluation metrics |
| `task_type` | `str` | "regression" or "classification" |

**Example:**

```python
from torch_pharma.models import GCN
from torch_pharma.tasks import PropertyPredictionTask

model = GCN(in_channels=11, hidden_channels=128, out_channels=1)

task = PropertyPredictionTask(
    model=model,
    target="homo",
    task_type="regression"
)

# Train
trainer.fit(task, train_loader, val_loader)
```

**Multi-task:**

```python
task = PropertyPredictionTask(
    model=model,
    target=["homo", "lumo", "gap"],  # Multiple targets
    task_type="regression"
)
```

---

## Molecule Generation

### MoleculeGenerationTask

De novo molecule generation.

```python
class MoleculeGenerationTask(BaseTask):
    """
    Task for training generative models.
    
    Supports autoregressive, VAE, and diffusion-based generation.
    """
```

#### Constructor

```python
def __init__(
    self,
    model: nn.Module,
    generation_type: str = "diffusion",  # or "vae", "autoregressive"
    criterion: Optional[nn.Module] = None,
    sampling_steps: int = 1000,
    temperature: float = 1.0,
    **optimizer_kwargs
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `model` | `nn.Module` | Generative model (EDM, VAE, etc.) |
| `generation_type` | `str` | Type of generative model |
| `sampling_steps` | `int` | Number of sampling steps |
| `temperature` | `float` | Sampling temperature |

**Methods:**

```python
def sample(
    self,
    n_samples: int,
    n_atoms: Optional[int] = None
) -> List[Chem.Mol]:
    """
    Generate new molecules.
    
    Args:
        n_samples: Number of molecules to generate
        n_atoms: Target number of atoms (None for variable)
    
    Returns:
        List of RDKit molecules
    """
```

**Example:**

```python
from torch_pharma.models import EDMDiffusion
from torch_pharma.tasks import MoleculeGenerationTask

model = EDMDiffusion(n_atoms=29, n_atom_types=5)
task = MoleculeGenerationTask(model=model)

# Train
trainer.fit(task, train_loader)

# Generate
molecules = task.sample(n_samples=100)
```

---

## Binding Affinity

### BindingAffinityTask

Protein-ligand binding affinity prediction.

```python
class BindingAffinityTask(BaseTask):
    """
    Task for predicting protein-ligand binding affinity.
    
    Takes separate encoders for proteins and ligands,
    combines them for affinity prediction.
    """
```

#### Constructor

```python
def __init__(
    self,
    ligand_model: nn.Module,
    protein_model: nn.Module,
    interaction_head: Optional[nn.Module] = None,
    criterion: Optional[nn.Module] = None,
    **optimizer_kwargs
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `ligand_model` | `nn.Module` | Model for ligand encoding |
| `protein_model` | `nn.Module` | Model for protein encoding |
| `interaction_head` | `Optional[nn.Module]` | Head for combining representations |

---

## Toxicity Prediction

### ToxicityPredictionTask

ADMET property prediction.

```python
class ToxicityPredictionTask(BaseTask):
    """
    Task for predicting toxicity and ADMET properties.
    
    Properties include:
    - Toxicity (mutagenicity, hepatotoxicity)
    - ADMET (absorption, distribution, metabolism, excretion)
    - Bioavailability
    """
```

---

## Custom Tasks

Creating a custom task:

```python
from torch_pharma.tasks import BaseTask

class MyTask(BaseTask):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.custom_layer = nn.Linear(128, 1)
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        h = self.model(batch.x, batch.edge_index, batch.batch)
        out = self.custom_layer(h)
        
        # Compute loss
        loss = self.criterion(out, batch.y)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        h = self.model(batch.x, batch.edge_index, batch.batch)
        out = self.custom_layer(h)
        loss = self.criterion(out, batch.y)
        
        return {
            "val_loss": loss,
            "val_mae": torch.abs(out - batch.y).mean()
        }
```

---

## Task Configuration

Tasks can be configured via YAML:

```yaml
task:
  name: PropertyPredictionTask
  target: homo
  task_type: regression
  lr: 0.001
  
  # Task-specific
  metrics:
    - mae
    - rmse
```

```python
from torch_pharma.cli import load_config, create_task

config = load_config("config.yaml")
task = create_task(config["task"])
```

---

## Type Aliases

```python
from typing import Union

Task = Union[
    PropertyPredictionTask,
    MoleculeGenerationTask,
    BindingAffinityTask,
    ToxicityPredictionTask
]
```
