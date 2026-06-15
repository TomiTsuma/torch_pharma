# Torch Pharma Integration Documentation

This directory contains documentation for the torch_pharma package structure and integration patterns.

## Package Structure

```
torch_pharma/
├── config/              # Configuration management
│   ├── default.yaml     # Default hyperparameters
│   ├── model/           # Model-specific configs
│   ├── task/            # Task-specific configs
│   └── training/        # Training hyperparameters
├── data/                # Data handling
│   ├── components/      # Data components (QM9, GEOM, etc.)
│   ├── datasets/        # Dataset implementations
│   ├── loaders/         # Custom data loaders
│   └── utils/           # Data utilities
├── diffusion/           # Diffusion models
│   ├── categorical.py   # Discrete diffusion
│   ├── continuous.py    # Continuous diffusion
│   └── e3moldiffusion/  # E3MolDiffusion implementation
├── dynamics/            # Molecular dynamics models
│   ├── egnn.py          # EGNN implementation
│   ├── gcpnet.py        # GCPNet implementation
│   └── utils.py         # Dynamics utilities
├── evaluation/          # Evaluation metrics
│   ├── metrics.py       # General metrics
│   └── molecules/       # Molecule-specific metrics
├── features/            # Feature extraction
│   ├── geometry.py      # Geometric features
│   └── utils.py         # Feature utilities
├── models/              # Model architectures
│   ├── gnn/             # Graph neural networks
│   ├── diffusion/       # Diffusion models
│   ├── dynamics/        # Dynamics models
│   └── utils.py         # Model utilities
├── molecules/           # Molecule utilities
│   └── molecule_utils.py
├── rl/                  # Reinforcement learning
│   ├── agents/          # RL agents (DQN, PPO, SAC)
│   └── environments/    # RL environments
├── sampling/            # Sampling strategies
├── tasks/               # Training tasks
│   ├── base.py          # Base task class
│   ├── molecule_generation.py
│   ├── property_prediction.py
│   └── binding_affinity.py
└── training/            # Training utilities
    ├── trainer.py       # Trainer class
    └── callbacks.py     # Training callbacks
```

## Integration Patterns

### 1. Task-Based Training

All training follows a task-based pattern:

```python
from torch_pharma.tasks import BaseTask
from torch_pharma.training import Trainer

class MyTask(BaseTask):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.kwargs = kwargs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

task = MyTask(model=my_model)
trainer = Trainer(max_epochs=100)
trainer.fit(task)
```

### 2. Data Pipeline

Data flows through datasets -> loaders -> collaters:

```python
from torch_pharma.data import MyDataset
from torch_pharma.data.loaders import AdaptiveDataLoader

dataset = MyDataset()
loader = AdaptiveDataLoader(dataset, batch_size=32)
```

### 3. Model Configuration

Models use dataclass configurations:

```python
from torch_pharma.models.diffusion.config import DGTDiffusionConfig

config = DGTDiffusionConfig.dmt_b()
model = DGTDiffusion(config)
```

## Related Documentation

- [NExT-Mol Integration](../next-mol-implementation/README.md) - Native port of NExT-Mol (MoLlama + DMT)
- [Installation](../installation.md) - Installation guide
- [Roadmap](../roadmap.md) - Implementation roadmap
