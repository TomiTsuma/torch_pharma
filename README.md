[pypi-image]: https://badge.fury.io/py/torch-pharma.svg
[pypi-url]: https://pypi.org/project/torch-pharma/
[testing-image]: https://github.com/yourusername/torch-pharma/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/yourusername/torch-pharma/actions/workflows/testing.yml
[docs-image]: https://readthedocs.org/projects/torch-pharma/badge/?version=latest
[docs-url]: https://torch-pharma.readthedocs.io
[coverage-image]: https://codecov.io/gh/yourusername/torch-pharma/branch/main/graph/badge.svg
[coverage-url]: https://codecov.io/gh/yourusername/torch-pharma

# Torch Pharma

[![PyPI Version][pypi-image]][pypi-url]
[![Testing Status][testing-image]][testing-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]

--------------------------------------------------------------------------------

Torch Pharma is a PyTorch-native framework for drug discovery and molecular deep learning. It provides a unified interface for building, training, and evaluating models across molecular property prediction, molecule generation, and reinforcement learning-based optimization.

The framework is designed to bridge the gap between graph-based deep learning libraries and domain-specific drug discovery toolkits by integrating molecular representations, learning algorithms, and evaluation pipelines into a single modular system.

--------------------------------------------------------------------------------

## Motivation

Current tooling in molecular machine learning is fragmented. Libraries such as PyTorch Geometric provide strong primitives for graph learning, while DeepChem offers domain-specific utilities. However, there is no unified framework that:

- Treats drug discovery tasks as first-class abstractions
- Provides a singular framework for collecting, featurizing, and training on molecular datasets
- Integrates GNNs, RL, and other deep learning models with molecular modeling
- Provides consistent training and evaluation pipelines
- Assesses validity of de novo generated molecules

Torch Pharma addresses these limitations by introducing a task-oriented and extensible framework tailored for drug discovery workflows that runs on yaml files.

--------------------------------------------------------------------------------

## Features

### Molecular Representations
- SMILES to graph conversion
- Atom and bond featurization
- RDKit integration for cheminformatics tasks
- PoseBuster integration for pose generation
- Geometry-Complete integration for 3D structure generation
- Chemistry-aware validation and constraints

### Models
- Graph Neural Networks (Diffusion, Convolutional, GANs etc.)
- Graph transformers for molecular learning
- Extensible architecture for protein and multimodal models

### Tasks
- Property prediction
- Molecule generation
- Binding affinity prediction
- Toxicity prediction

### Reinforcement Learning
- Molecular environments for optimization
- Reward functions based on chemical properties
- Support for DQN, PPO, and SAC

### Training
- Unified Trainer abstraction
- Checkpointing and callbacks
- Experiment reproducibility

### Evaluation
- Metrics for validity, novelty, and diversity
- Domain-specific scoring (QED, LogP)

--------------------------------------------------------------------------------

## Installation

### From PyPI

```bash
pip install torch-pharma
````

### From Source

```bash
git clone https://github.com/TomiTsuma/torch-pharma.git
cd torch-pharma
pip install -e .
```

---

## Quick Example

### Property Prediction

```python
from torch_pharma.molecules import Molecule
from torch_pharma.models.gnn import GCN
from torch_pharma.tasks import PropertyPredictionTask
from torch_pharma.training import Trainer

mol = Molecule.from_smiles("CCO")

model = GCN(hidden_dim=128)
task = PropertyPredictionTask(model=model)

trainer = Trainer(max_epochs=10)
trainer.fit(task)
```

### Reinforcement Learning for Molecule Optimization

```python
from torch_pharma.rl.envs import MoleculeEnv
from torch_pharma.rl.agents import PPOAgent
from torch_pharma.training import Trainer

env = MoleculeEnv(task="optimize_qed")
agent = PPOAgent()

trainer = Trainer(max_steps=10000)
trainer.fit(agent, env)
```

---

## Experiments and Research Alignment

Torch Pharma includes implementations and reproducible experiments inspired by key research papers in molecular machine learning and reinforcement learning.

### Molecular Property Prediction

* Dataset: QM9, ZINC
* Models: GCN, MPNN
* Related work:

  * Gilmer et al., "Geometry-Complete Diffusion for 3D Molecule Generation and Optimization" (2017)
  * Wu et al., "MoleculeNet: A Benchmark for Molecular Machine Learning" (2018)

Example:

```bash
python examples/property_prediction/train_qm9.py
```

---

### Molecule Generation

* Graph-based molecule generation models
* Sequence-based generation via SMILES

Related work:

* Morehead et al., "Geometry-Complete Diffusion for 3D Molecule Generation and Optimization" (2023)

Example:

```bash
python examples/molecule_generation/2302.04313/generate.py
```

---

### Reinforcement Learning for Drug Optimization

* Objective: maximize drug-likeness (QED), minimize toxicity
* Environment: molecular graph editing

Related work:

* Olivecrona et al., "Molecular De-Novo Design through Deep Reinforcement Learning" (2017)
* Zhou et al., "Optimization of Molecules via Deep Reinforcement Learning" (2019)

Example:

```bash
python examples/rl_optimization/optimize_qed.py
```

---

### Binding Affinity Prediction

* Task: predict protein-ligand binding strength
* Models: graph-based encoders

Related work:

* Öztürk et al., "DeepDTA: Deep Drug–Target Binding Affinity Prediction" (2018)

Example:

```bash
python examples/binding_affinity/train_binding.py
```

---

## Project Structure

```
torch_pharma/
  molecules/      # molecular abstractions
  models/         # neural architectures
  tasks/          # task definitions
  rl/             # reinforcement learning modules
  training/       # training engine
  evaluation/     # metrics and benchmarks
```

---

## Running Tests

```bash
pytest
```

---

## Documentation

Documentation is available at:

[https://torch-pharma.readthedocs.io](https://torch-pharma.readthedocs.io)

---

## Contributing

Contributions are welcome. You can contribute by:

* Adding new models or datasets
* Implementing new reinforcement learning algorithms
* Extending evaluation metrics
* Improving documentation and examples

---

## Citation

If you use Torch Pharma in your research, please cite:

```bibtex
@software{torchpharma2026,
  title={Torch Pharma: A PyTorch Framework for Drug Discovery},
  author={Your Name},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License.

---

## Roadmap

* Diffusion models for molecule generation
* Protein-ligand docking integration
* Multi-objective reinforcement learning
* Pretrained molecular foundation models

---

## Vision

Torch Pharma aims to provide a unified framework for AI-driven drug discovery, enabling researchers and engineers to build, evaluate, and deploy molecular machine learning models efficiently.

---