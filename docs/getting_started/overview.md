# Overview

Torch Pharma is built around several core concepts that work together to provide a unified framework for molecular machine learning.

---

## Architecture

The framework is organized into modular components:

```
torch_pharma/
├── molecules/      # Molecular representations and chemistry
├── data/           # Datasets, loaders, and transforms
├── models/         # Neural network architectures
├── tasks/          # Task definitions and training logic
├── training/       # Training engine and utilities
├── evaluation/     # Metrics and benchmarking
├── rl/             # Reinforcement learning components
├── features/       # Feature engineering
├── integrations/   # Third-party integrations
└── utils/          # Utility functions
```

---

## Core Concepts

### 1. Molecules

The `Molecule` class is the fundamental unit of data in Torch Pharma. It provides:

- **SMILES parsing** and validation
- **Graph representation** conversion (nodes and edges)
- **Featurization** (atom and bond features)
- **3D coordinates** handling

```python
from torch_pharma.molecules import Molecule

# Create molecule from SMILES
mol = Molecule.from_smiles("CCO")

# Access graph representation
print(mol.node_features)   # Atom features
print(mol.edge_index)      # Bond connections
print(mol.edge_features)   # Bond features
```

### 2. Models

Torch Pharma provides pre-built models for common architectures:

- **GCN**: Graph Convolutional Networks
- **GAT**: Graph Attention Networks
- **MPNN**: Message Passing Neural Networks
- **EGNN**: Equivariant Graph Neural Networks
- **Transformers**: Graph Transformer architectures
- **Diffusion**: Denoising Diffusion models

All models follow the PyTorch `nn.Module` interface:

```python
from torch_pharma.models import GCN

model = GCN(
    in_channels=11,
    hidden_channels=128,
    num_layers=3,
    out_channels=1
)

output = model(x, edge_index, batch)
```

### 3. Tasks

Tasks encapsulate the training logic for specific objectives:

- **PropertyPredictionTask**: Regression/classification on molecular properties
- **MoleculeGenerationTask**: Generative modeling of molecules
- **BindingAffinityTask**: Protein-ligand binding prediction
- **ToxicityPredictionTask**: ADMET property prediction

Tasks combine models, loss functions, and optimizers:

```python
from torch_pharma.tasks import PropertyPredictionTask

task = PropertyPredictionTask(
    model=model,
    target='homo',
    criterion=nn.MSELoss(),
    optimizer=torch.optim.Adam
)
```

### 4. Data

The data module provides:

- **Built-in datasets**: QM9, ZINC, BindingDB
- **Custom dataset support**: Easy extension for new data sources
- **Data loaders**: Efficient batching with PyTorch Geometric
- **Transforms**: SMILES to graph, normalization, augmentation

```python
from torch_pharma.data import QM9Dataset, DataLoader

dataset = QM9Dataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 5. Training

The `Trainer` class provides a unified training interface:

- **Automatic device handling**: CPU/GPU/Multi-GPU
- **Checkpointing**: Save and resume training
- **Logging**: Integration with Weights & Biases, TensorBoard
- **Callbacks**: Custom training hooks

```python
from torch_pharma.training import Trainer

trainer = Trainer(
    max_epochs=100,
    accelerator='auto',
    callbacks=[EarlyStopping(), ModelCheckpoint()]
)

trainer.fit(task, train_loader, val_loader)
```

### 6. Evaluation

Comprehensive evaluation metrics for molecular ML:

- **Validity**: Percentage of chemically valid molecules
- **Uniqueness**: Percentage of unique molecules
- **Novelty**: Percentage of molecules not in training set
- **Diversity**: Internal diversity of generated molecules
- **Property metrics**: QED, LogP, SA score

```python
from torch_pharma.evaluation import BasicMolecularMetrics

metrics = BasicMolecularMetrics(dataset_info)
validity, uniqueness, novelty = metrics.evaluate_rdmols(generated_mols)
```

### 7. Reinforcement Learning

For de novo molecule optimization:

- **Environments**: Molecular editing environments
- **Agents**: DQN, PPO, SAC implementations
- **Rewards**: Chemical property-based rewards

```python
from torch_pharma.rl import MoleculeEnv, PPOAgent

env = MoleculeEnv(target_property='qed')
agent = PPOAgent(env.observation_space, env.action_space)
trainer = Trainer(max_steps=10000)
trainer.fit(agent, env)
```

---

## Design Philosophy

### Modularity

Each component is self-contained and can be used independently:

```python
# Use just the molecules module
from torch_pharma.molecules import Molecule

mol = Molecule.from_smiles("CCO")

# Use just the models
from torch_pharma.models import GCN

model = GCN(...)

# Combine as needed
```

### PyTorch Native

Torch Pharma is built on top of PyTorch and integrates seamlessly:

- Standard PyTorch `nn.Module` models
- PyTorch `DataLoader` for data loading
- PyTorch Lightning-style Trainer API
- Automatic differentiation and optimization

### Extensibility

Easy to extend with custom components:

```python
from torch_pharma.models import BaseGNN

class MyCustomGNN(BaseGNN):
    def __init__(self, ...):
        super().__init__()
        # Custom implementation
    
    def forward(self, x, edge_index, batch):
        # Custom forward pass
        return output
```

### Configuration-Driven

YAML-based configuration for reproducibility:

```yaml
experiment:
  name: qm9_homo_prediction
  seed: 42

model:
  name: GCN
  hidden_channels: 128

trainer:
  max_epochs: 100
  accelerator: auto
```

---

## Integration Ecosystem

Torch Pharma integrates with popular tools:

| Tool | Integration | Purpose |
|------|-------------|---------|
| RDKit | `torch_pharma.integrations.rdkit` | Cheminformatics |
| PyTorch Geometric | Native | Graph neural networks |
| Weights & Biases | `torch_pharma.integrations.wandb` | Experiment tracking |
| OpenBabel | `torch_pharma.molecules` | File format conversion |
| PoseBusters | `torch_pharma.evaluation` | Pose validation |

---

## Performance

Torch Pharma is designed for efficiency:

- **GPU acceleration**: Automatic CUDA support
- **Batching**: Efficient batch processing with PyTorch Geometric
- **Memory optimization**: Gradient checkpointing and sparse operations
- **Distributed training**: Multi-GPU and cluster support

---

## Comparison with Other Frameworks

| Feature | Torch Pharma | PyTorch Geometric | DeepChem |
|---------|--------------|-------------------|----------|
| PyTorch Native | ✅ | ✅ | ⚠️ (TensorFlow/PyTorch) |
| Molecular Tasks | ✅ | ❌ | ✅ |
| GNN Models | ✅ | ✅ | ⚠️ |
| RL for Molecules | ✅ | ❌ | ⚠️ |
| 3D Generation | ✅ | ⚠️ | ❌ |
| Integrated Training | ✅ | ❌ | ⚠️ |

Torch Pharma fills the gap by combining the best of PyTorch Geometric (graph learning) and DeepChem (drug discovery) with a unified, PyTorch-native interface.
