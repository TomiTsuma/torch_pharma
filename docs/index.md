# Torch Pharma

Torch Pharma is a PyTorch-native framework for drug discovery and molecular deep learning. It provides a unified interface for building, training, and evaluating models across molecular property prediction, molecule generation, and reinforcement learning-based optimization.

The framework is designed to bridge the gap between graph-based deep learning libraries and domain-specific drug discovery toolkits by integrating molecular representations, learning algorithms, and evaluation pipelines into a single modular system.

---

## 🚀 Key Features

### Molecular Representations
- **SMILES to Graph**: Direct conversion with RDKit integration.
- **Featurization**: Extensive atom and bond featurization options.
- **Cheminformatics**: Chemistry-aware validation and constraints.

### Advanced Modeling
- **Graph Neural Networks**: Diffusion models, EGNN, GANs, and Transformers.
- **Extensible Architectures**: Modular designs for protein-ligand and multimodal tasks.

### specialized Tasks
- **Property Prediction**: Predict solubility, toxicity, and binding affinity.
- **De novo Generation**: Generate novel molecules with chemical validity.
- **RL Optimization**: Optimize molecular properties via reinforcement learning.

---

## 🛠️ Quick Start

### Installation

```bash
pip install torch-pharma
```

See the [Installation Guide](installation.md) for more details on installing from source or managing dependencies.

### Property Prediction Example

```python
from torch_pharma.molecules import Molecule
from torch_pharma.models.gnn import GCN
from torch_pharma.tasks import PropertyPredictionTask
from torch_pharma.training import Trainer

# Initialize model and task
model = GCN(hidden_dim=128)
task = PropertyPredictionTask(model=model)

# Train on your dataset
trainer = Trainer(max_epochs=10)
trainer.fit(task)
```

---

## 🗺️ Navigation

| Section | Description |
| --- | --- |
| [**Tutorials**](tutorials/property_prediction.md) | Step-by-step guides for common drug discovery workflows. |
| [**Datasets**](datasets/QM9.md) | Details on supported datasets and loading mechanics. |
| [**API Reference**](api/index.md) | Detailed documentation of the Torch Pharma API. |

---

## 🔮 Roadmap

- [ ] Diffusion models for 3D molecule generation.
- [ ] Protein-ligand docking integration.
- [ ] Multi-objective reinforcement learning.
- [ ] Pretrained molecular foundation models.

---

## 🌟 Vision

Torch Pharma aims to provide a unified framework for AI-driven drug discovery, enabling researchers and engineers to build, evaluate, and deploy molecular machine learning models efficiently.
