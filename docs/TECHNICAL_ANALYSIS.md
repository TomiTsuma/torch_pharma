# Torch Pharma - Technical Analysis

## Project Overview

**Torch Pharma** is a PyTorch-native framework for drug discovery and molecular deep learning. It provides a unified interface for building, training, and evaluating models across molecular property prediction, molecule generation, and reinforcement learning-based optimization.

## Architecture Summary

```
torch_pharma/
├── molecules/          # Molecular abstractions (Molecule, chemistry utils)
├── models/             # Neural architectures (DDPM, EGNN, GCPNet, Diffusion)
├── tasks/              # Task definitions (PropertyPrediction, MoleculeGeneration, etc.)
├── rl/                 # Reinforcement learning (agents, envs, rewards)
├── data/               # Datasets, transforms, loaders (QM9, ZINC, BindingDB)
├── features/           # Feature engineering (ScalarVector, geometry utils)
├── evaluation/         # Metrics and benchmarks (QED, PoseBusters)
├── training/           # Training engine (Trainer, callbacks, checkpointing)
├── utils/              # Utilities (logging, tracking, visualization)
└── cli/                # Command-line interface
```

## Key Abstractions

### 1. `ScalarVector` ([features/geometry.py](features/geometry.py))

A core tuple-based data structure that holds both scalar and vector features:

```python
class ScalarVector(tuple):
    def __new__(cls, scalar, vector):
        return tuple.__new__(cls, (scalar, vector))
    
    @property
    def scalar(self):
        return self[0]
    
    @property
    def vector(self):
        return self[1]
```

**Purpose**: Enables E(3) equivariance for 3D molecular data by separating:
- **Scalar features**: Rotationally invariant (e.g., atom types, charges)
- **Vector features**: Rotationally covariant (e.g., coordinates, orientations)

**Key operations**:
- Element-wise addition and multiplication
- Concatenation across dimensions
- Masking for batch processing
- Centralization/decentralization for coordinate normalization

### 2. `Molecule` ([molecules/molecule.py](molecules/molecule.py))

Molecular abstraction with:

- SMILES parsing and validation
- RDKit integration for cheminformatics
- OpenBabel support for format conversion
- 3D structure generation and relaxation

**Key functions**:
- `mol2smiles(mol)`: Convert RDKit Mol to SMILES string
- `build_molecule(positions, atom_types, dataset_info)`: Build molecule from coordinates
- `process_molecule(rdmol, add_hydrogens, sanitize, relax_iter)`: Clean and optimize molecule

### 3. `Task` ([tasks/base.py](tasks/base.py))

Abstract base class for all tasks:

```python
class Task(ABC):
    @abstractmethod
    def run(self, model, data_loader):
        pass
```

**Implemented tasks**:
- `PropertyPredictionTask`: Predict molecular properties (e.g., QM9 targets)
- `MoleculeGenerationTask`: Generate new molecules via diffusion
- `BindingAffinityTask`: Predict protein-ligand binding strength
- `ToxicityPredictionTask`: Predict molecular toxicity

### 4. `Trainer` ([training/trainer.py](training/trainer.py))

Core training engine:

```python
class Trainer:
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
```

**Features**:
- Epoch-based training loop
- Checkpointing support
- Callback system for logging and metrics

## Core Models

### DDPM (Denoising Diffusion Probabilistic Models)

**Files**: [models/ddpm/ddpm.py](models/ddpm/ddpm.py), [models/diffusion/variational_diffusion.py](models/diffusion/variational_diffusion.py)

**Purpose**: Learn to denoise data through a Markov chain of noise addition and removal.

**Key components**:
- **Forward process**: Add Gaussian noise incrementally
- **Reverse process**: Learn to predict and remove noise
- **Noise schedule**: Controls noise variance at each timestep

**Architecture**: Uses EGNN for E(3) equivariant message passing.

### E(3)-GNN (Equivariant Graph Neural Networks)

**Files**: [models/dynamics/egnn.py](models/dynamics/egnn.py)

**EGNN_Sparse Layer**:
```python
class EGNN_Sparse(MessagePassing):
    def forward(self, x, edge_index, edge_attr, batch):
        # x: (n_points, pos_dim + feat_dim)
        coors, feats = x[:, :pos_dim], x[:, pos_dim:]
        # Message passing with coordinate updates
```

**Key features**:
- **E(3) equivariance**: Preserves rotational and translational symmetry
- **Coordinate updates**: Learn forces from feature messages
- **Feature updates**: Aggregate messages with normalized coordinates

**EGNNDynamics**:
- Time-conditioned diffusion model
- Self-conditioning for improved predictions
- Context conditioning for molecular properties

### GCPNet (Geometry-Complete Protein Networks)

**File**: [models/dynamics/gcpnet.py](models/dynamics/gcpnet.py)

**Purpose**: Full GCPNet implementation for protein-ligand interaction modeling.

**Features**:
- Geometry-complete feature representations
- Multi-scale message passing
- Protein-ligand binding affinity prediction

## Data Pipeline

### Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| QM9 | Small organic molecules with quantum properties | 134k |
| ZINC | Commercially available compounds | 250M+ |
| BindingDB | Protein-ligand binding affinities | 2M+ |

### Feature Engineering

**Node features**:
- One-hot encoded atom types
- Partial charges
- Hybridization state
- Aromaticity flags

**Edge features**:
- Distance (radial)
- Directional (vector)
- Bond type encoding

**Geometry features**:
- Centralized coordinates (translation-invariant)
- Local frames (rotation-invariant)
- Edge-wise distance and angle features

### Data Loaders

**ProcessedDataset**: Pre-processed PyG dataset with:
- Graph construction from molecular data
- Feature normalization
- Shuffle and batch support

**retrieve_dataloaders**: Configurable dataloader factory supporting:
- Batch size configuration
- Number of workers
- Filter by atom count
- Train/val/test splits

## Reinforcement Learning

### Components

**Environments** ([rl/envs/](rl/envs/)):
- Molecular graph editing environments
- Action space: add/remove atoms, modify bonds
- State space: molecular graph with coordinates

**Rewards** ([rl/rewards/](rl/rewards/)):
- **QED** (Quantitative Estimation of Drug-likeness): Balance of properties
- **LogP**: Lipophilicity penalty
- **Toxicity**: Safety penalty

**Agents** ([rl/agents/](rl/agents/)):
- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Network)
- SAC (Soft Actor-Critic)

### Training Flow

```
Environment → Agent → Action → New State → Reward → Replay Buffer
                                    ↓
                              Update Policy
```

## Evaluation

### Metrics

**Molecular validity**:
- Chemical validity (RDKit sanitization)
- Novelty (not in training set)
- Diversity (unique SMILES)

**Property-specific**:
- **QM9**: MAE for energy, gap, homo/lumo
- **Binding affinity**: Pearson correlation, RMSE
- **Toxicity**: AUC-ROC, precision-recall

### Benchmarks

**Files**: [evaluation/benchmarks.py](evaluation/benchmarks.py)

**Supported benchmarks**:
- PoseBusters: Protein-ligand pose validation
- QM9 reference metrics
- Custom task-specific metrics

## Tracking & Logging

### Activation Tracking

**Files**: [utils/tracking/](utils/tracking/)

**ActivationStore**: Centralized activation tracking:
```python
class ActivationStore:
    def __init__(self):
        self.stats = {}  # Layer name → list of stats dicts
        self.nodes = {}  # Node tensors
        self.edges = {}  # Edge tensors
```

**Loggers**:
- **MLflowLogger**: Log to MLflow experiment
- **WandbActivationLogger**: Log to Weights & Biases

### Configuration

**Files**: [config/](config/)

**default.yaml**: Default training configuration:
```yaml
training:
  max_epochs: 100
  batch_size: 128
  learning_rate: 1e-4

data:
  dataset: QM9
  filter_n_atoms: null

model:
  type: ddpm
  num_layers: 9
```

## Installation & Dependencies

### Core Dependencies

| Package | Purpose |
|---------|---------|
| torch | Deep learning framework |
| torch-geometric | Graph neural networks |
| torch-scatter | Scatter operations |
| rdkit | Cheminformatics |
| openbabel | Molecular format conversion |
| numpy/pandas | Data processing |
| pyyaml | Configuration |
| omegaconf | Config management |
| wandb/mlflow | Experiment tracking |

### Installation

```bash
# Install PyG dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.9.0+cu128.html

# Install torch-pharma
pip install --no-build-isolation -e .
```

## Current State & Roadmap

### Implemented Features

- [x] Molecular property prediction
- [x] 3D molecule generation via diffusion
- [x] E(3) equivariant models (EGNN, GCPNet)
- [x] QM9/ZINC dataset integration
- [x] RDKit/OpenBabel integration
- [x] MLflow/WandB tracking
- [x] QED reward function

### Roadmap

- [ ] Protein-ligand docking integration
- [ ] Multi-objective RL
- [ ] Pretrained molecular foundation models
- [ ] Active learning loop
- [ ] Bayesian optimization for hyperparameters

## Notable Design Patterns

### 1. Equivariance Pattern

```python
# Separate scalar (invariant) and vector (covariant) features
features = ScalarVector(scalar_tensor, vector_tensor)

# Operations preserve equivariance
result = layer(features)  # Output is also ScalarVector
```

### 2. Batch Processing Pattern

```python
# PyG Batch object contains all graphs in batch
batch = Batch(
    x=coordinates,        # Node positions
    h=node_features,      # Node scalars
    chi=node_vectors,     # Node vectors
    e=edge_features,      # Edge scalars
    xi=edge_vectors,      # Edge vectors
    edge_index=edge_index,
    batch=batch_index     # Graph assignment
)
```

### 3. Diffusion Pattern

```python
# Forward process: add noise
x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

# Reverse process: predict noise
noise_pred = model(x_t, t)
```

### 4. Modularity Pattern

```python
# Each component is independently configurable
task = PropertyPredictionTask(model=model, dataset=dataset)
trainer = Trainer(task=task, callbacks=[logging, checkpoint])
```

## Example Usage

### Property Prediction

```python
from torch_pharma.molecules import Molecule
from torch_pharma.tasks import PropertyPredictionTask
from torch_pharma.training import Trainer

# Load data
dataset = QM9Dataset()
dataloader = DataLoader(dataset, batch_size=32)

# Create model and task
model = EGNN_Sparse_Network(n_layers=9, feats_dim=64)
task = PropertyPredictionTask(model=model, dataset=dataset)

# Train
trainer = Trainer(max_epochs=100)
trainer.fit(task)
```

### Molecule Generation

```python
from torch_pharma.models.ddpm import DDPM
from torch_pharma.tasks import MoleculeGenerationTask

# Create diffusion model
model = DDPM(
    num_atom_types=16,
    num_encoder_layers=9
)

# Generate molecules
task = MoleculeGenerationTask(model=model)
samples = task.generate(num_samples=100)
```

### Reinforcement Learning

```python
from torch_pharma.rl.envs import MoleculeEnv
from torch_pharma.rl.agents import PPOAgent

# Create environment
env = MoleculeEnv(task="optimize_qed")

# Create agent
agent = PPOAgent(
    state_space=env.observation_space,
    action_space=env.action_space
)

# Train
agent.train(env, num_steps=10000)
```

## Conclusion

Torch Pharma provides a comprehensive framework for molecular deep learning with:
- **E(3) equivariant models** for 3D molecular data
- **Diffusion models** for molecule generation
- **RL integration** for optimization
- **Modular design** for easy extension

The framework bridges the gap between general-purpose deep learning libraries (PyTorch, PyG) and domain-specific drug discovery tools (RDKit, OpenBabel).
