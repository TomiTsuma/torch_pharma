# Quick Start

Get up and running with Torch Pharma in minutes.

---

## Installation

First, install Torch Pharma:

```bash
pip install torch-pharma
```

See the [Installation Guide](../installation.md) for detailed instructions.

---

## Your First Model

Let's build a simple property prediction model on the QM9 dataset.

### 1. Import the Necessary Modules

```python
import torch
import torch.nn as nn
from torch_pharma.data import QM9Dataset, DataLoader
from torch_pharma.models import GCN
from torch_pharma.tasks import PropertyPredictionTask
from torch_pharma.training import Trainer
```

### 2. Load the Dataset

```python
# Load QM9 dataset
dataset = QM9Dataset()

# Split into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
```

### 3. Define the Model

```python
# Initialize a Graph Convolutional Network
model = GCN(
    in_channels=11,        # Number of input features per atom
    hidden_channels=128,   # Hidden dimension
    num_layers=3,          # Number of GCN layers
    out_channels=1         # Output dimension (for regression)
)
```

### 4. Create the Task

```python
# Property prediction task for HOMO energy
task = PropertyPredictionTask(
    model=model,
    target='homo',           # Target property from QM9
    criterion=nn.MSELoss(),
    lr=1e-3
)
```

### 5. Train the Model

```python
# Initialize trainer
trainer = Trainer(
    max_epochs=10,
    accelerator='auto'     # Automatically use GPU if available
)

# Train
trainer.fit(task, train_loader, val_loader)
```

### 6. Evaluate

```python
# Evaluate on validation set
metrics = trainer.validate(task, val_loader)
print(f"Validation MAE: {metrics['val_mae']:.4f}")
```

---

## Next Steps

- **Tutorials**: Learn about [molecule generation](tutorials/molecule_generation.md) and [RL optimization](tutorials/rl_optimization.md)
- **Datasets**: Explore available [datasets](datasets/index.md)
- **API Reference**: Check the [API documentation](api/index.md)

---

## Common Patterns

### Configuration-Based Training

Torch Pharma supports YAML-based configuration:

```yaml
# config.yaml
model:
  name: GCN
  hidden_channels: 128
  num_layers: 3

task:
  name: PropertyPredictionTask
  target: homo

trainer:
  max_epochs: 100
  accelerator: auto
```

```python
from torch_pharma.cli import load_config, train

config = load_config('config.yaml')
train(config)
```

### Custom Dataset

```python
from torch_pharma.data import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, smiles_list, targets):
        super().__init__()
        self.smiles = smiles_list
        self.targets = targets
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        from torch_pharma.molecules import Molecule
        mol = Molecule.from_smiles(self.smiles[idx])
        return {
            'x': mol.node_features,
            'edge_index': mol.edge_index,
            'y': self.targets[idx]
        }
```

---

## Getting Help

- **GitHub Issues**: [https://github.com/TomiTsuma/torch_pharma/issues](https://github.com/TomiTsuma/torch_pharma/issues)
- **API Documentation**: [https://torch-pharma.readthedocs.io](https://torch-pharma.readthedocs.io)
