# Torch Pharma

**A PyTorch-native framework for drug discovery and molecular deep learning**

---

[![PyPI Version](https://badge.fury.io/py/torch-pharma.svg)](https://pypi.org/project/torch-pharma/)
[![Testing Status](https://github.com/TomiTsuma/torch_pharma/actions/workflows/testing.yml/badge.svg)](https://github.com/TomiTsuma/torch_pharma/actions/workflows/testing.yml)
[![Docs Status](https://readthedocs.org/projects/torch-pharma/badge/?version=latest)](https://torch-pharma.readthedocs.io)
[![Code Coverage](https://codecov.io/gh/TomiTsuma/torch_pharma/branch/main/graph/badge.svg)](https://codecov.io/gh/TomiTsuma/torch_pharma)

Torch Pharma is a [PyTorch](https://pytorch.org/)-native framework for drug discovery and molecular deep learning. It provides a unified interface for building, training, and evaluating models across molecular property prediction, molecule generation, and reinforcement learning-based optimization.

The framework is designed to bridge the gap between graph-based deep learning libraries and domain-specific drug discovery toolkits by integrating molecular representations, learning algorithms, and evaluation pipelines into a single modular system.

---

## Why Torch Pharma?

Current tooling in molecular machine learning is fragmented. Libraries such as [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) provide strong primitives for graph learning, while [DeepChem](https://deepchem.io/) offers domain-specific utilities. However, there is no unified framework that:

- **Treats drug discovery tasks as first-class abstractions**
- **Provides a singular framework** for collecting, featurizing, and training on molecular datasets
- **Integrates GNNs, RL, and other deep learning models** with molecular modeling
- **Provides consistent training and evaluation pipelines**
- **Assesses validity** of de novo generated molecules

Torch Pharma addresses these limitations by introducing a **task-oriented** and **extensible** framework tailored for drug discovery workflows.

---

## Key Features

### Molecular Representations
- **SMILES to graph conversion** with chemistry-aware parsing
- **Atom and bond featurization** with customizable feature extractors
- **RDKit integration** for cheminformatics tasks
- **PoseBuster integration** for pose generation
- **3D structure support** for geometry-complete models
- **Chemistry-aware validation** and constraints

### Advanced Modeling
- **Graph Neural Networks**: GCN, MPNN, GAT, and more
- **Graph Transformers** for molecular learning
- **Diffusion Models** for 3D molecule generation (EDM, GEOM)
- **Equivariant Networks**: EGNN, GCPNet
- **Extensible architectures** for protein and multimodal models

### Specialized Tasks
- **Property Prediction**: Predict solubility, toxicity, binding affinity
- **Molecule Generation**: Generate novel molecules with chemical validity
- **Binding Affinity Prediction**: Protein-ligand interaction modeling
- **Toxicity Prediction**: ADMET property modeling

### Reinforcement Learning
- **Molecular environments** for optimization
- **Reward functions** based on chemical properties (QED, LogP)
- **Agents**: DQN, PPO, SAC implementations
- **Multi-objective optimization** support

### Training & Evaluation
- **Unified Trainer abstraction** with PyTorch Lightning-style API
- **Checkpointing and callbacks** for experiment management
- **Metrics**: Validity, novelty, diversity, QED, LogP
- **Benchmarks**: Standardized evaluation on molecular datasets

---

## Quick Start

### Installation

```bash
# From PyPI
pip install torch-pharma

# From source
git clone https://github.com/TomiTsuma/torch_pharma.git
cd torch_pharma
pip install -e .
```

### Property Prediction Example

```python
import torch
from torch_pharma.data import QM9Dataset, DataLoader
from torch_pharma.models import GCN
from torch_pharma.tasks import PropertyPredictionTask
from torch_pharma.training import Trainer

# Load dataset
dataset = QM9Dataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model and task
model = GCN(in_channels=11, hidden_channels=128, num_layers=3)
task = PropertyPredictionTask(
    model=model,
    target='homo',
    criterion=torch.nn.MSELoss()
)

# Train
trainer = Trainer(max_epochs=100, accelerator='auto')
trainer.fit(task, train_loader)
```

### Molecule Generation Example

```python
from torch_pharma.models import EDMDiffusion
from torch_pharma.tasks import MoleculeGenerationTask
from torch_pharma.training import Trainer

# Initialize diffusion model for molecule generation
model = EDMDiffusion(n_atoms=29, n_atom_types=5)
task = MoleculeGenerationTask(model=model)

# Train
trainer = Trainer(max_epochs=1000)
trainer.fit(task)
```

---

## Navigation

<div class="grid cards" markdown>

- :material-rocket-launch: **Getting Started**

    ---

    New to Torch Pharma? Start here for installation and basic concepts.

    [:octicons-arrow-right-24: Quick Start](getting_started/quickstart.md)

- :material-school: **Tutorials**

    ---

    Step-by-step guides for common drug discovery workflows.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

- :material-book-open: **API Reference**

    ---

    Comprehensive documentation of all modules, classes, and functions.

    [:octicons-arrow-right-24: API Reference](api/index.md)

- :material-database: **Datasets**

    ---

    Information on built-in datasets and how to use custom data.

    [:octicons-arrow-right-24: Datasets](datasets/index.md)

</div>

---

## Research Alignment

Torch Pharma includes implementations and reproducible experiments inspired by key research papers in molecular machine learning:

| Paper | Task | Implementation |
|-------|------|----------------|
| [Gilmer et al., 2017](https://arxiv.org/abs/1704.01212) | Neural Message Passing | `torch_pharma.models.MPNN` |
| [Hoogeboom et al., 2022](https://arxiv.org/abs/2203.17003) | 3D Diffusion Models | `torch_pharma.models.EDMDiffusion` |
| [Satorras et al., 2022](https://arxiv.org/abs/2102.09844) | EGNN | `torch_pharma.models.EGNN` |
| [Morehead et al., 2023](https://arxiv.org/abs/2306.07505) | GCPNet | `torch_pharma.models.GCPNet` |

---

## Citation

If you use Torch Pharma in your research, please cite:

```bibtex
@software{torchpharma2026,
  title={Torch Pharma: A PyTorch Framework for Drug Discovery},
  author={Tsuma, Tomi},
  year={2026},
  url={https://github.com/TomiTsuma/torch_pharma}
}
```

---

## License

Torch Pharma is released under the [MIT License](https://github.com/TomiTsuma/torch_pharma/blob/main/LICENSE).

---

## Community

- **GitHub**: [https://github.com/TomiTsuma/torch_pharma](https://github.com/TomiTsuma/torch_pharma)
- **Issues**: [GitHub Issues](https://github.com/TomiTsuma/torch_pharma/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TomiTsuma/torch_pharma/discussions)

---

## Roadmap

- [ ] Diffusion models for 3D molecule generation
- [ ] Protein-ligand docking integration
- [ ] Multi-objective reinforcement learning
- [ ] Pretrained molecular foundation models
- [ ] HuggingFace integration
- [ ] Weights & Biases logging improvements
