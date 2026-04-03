# Core Concepts

This guide explains the fundamental concepts and abstractions used throughout Torch Pharma.

---

## Molecule Representation

### SMILES vs. Graph

Molecular data in Torch Pharma is primarily represented as graphs. The transformation pipeline is:

```
SMILES String → RDKit Mol → Graph (nodes, edges) → PyTorch Tensors
```

```python
from torch_pharma.molecules import Molecule

# SMILES to graph
mol = Molecule.from_smiles("CCO")  # Ethanol

# Graph properties
print(mol.num_nodes)       # Number of atoms
print(mol.num_edges)       # Number of bonds
print(mol.node_features)   # Atom feature matrix [num_nodes, num_features]
print(mol.edge_index)      # Edge connectivity [2, num_edges]
```

### Atom Features

Each atom is represented by a feature vector containing:

- **Atomic number**: One-hot encoded element type
- **Degree**: Number of bonds
- **Formal charge**: Electronic charge
- **Hybridization**: sp, sp2, sp3, etc.
- **Aromaticity**: Boolean flag
- **Implicit hydrogens**: Number of H atoms
- **Chirality**: R/S configuration

### Bond Features

Each bond is represented by:

- **Bond type**: Single, double, triple, aromatic
- **Conjugation**: Whether part of conjugated system
- **Ring membership**: Whether in a ring
- **Stereo**: E/Z configuration

### 3D Coordinates

For 3D-aware models, molecules include spatial information:

```python
# Get 3D coordinates
positions = mol.positions  # [num_nodes, 3]

# Compute distances
distances = torch.cdist(positions, positions)
```

---

## Graph Neural Networks

### Message Passing

GNNs operate through message passing between nodes:

1. **Message**: Each node computes messages to neighbors
2. **Aggregate**: Messages are aggregated at each node
3. **Update**: Node features are updated based on aggregated messages

```python
# Message passing step
x = self.lin(x)  # Transform features
return self.propagate(edge_index, x=x)  # Propagate messages
```

### Common GNN Layers

Torch Pharma provides several GNN variants:

| Layer | Message Function | Aggregation | Characteristics |
|-------|------------------|-------------|-----------------|
| GCN | Normalized features | Mean | Simple, efficient |
| GAT | Attention-weighted | Sum | Captures importance |
| MPNN | Learned message | Sum | Flexible, powerful |
| EGNN | Equivariant | Sum | 3D-aware, SE(3) invariant |

### Readout Functions

To get graph-level predictions, we need readout functions:

```python
# Global mean pooling
x = global_mean_pool(x, batch)

# Global add pooling
x = global_add_pool(x, batch)

# Learned attention pooling
x = global_attention_pool(x, batch)
```

---

## Tasks and Objectives

### Property Prediction

Predicting molecular properties from structure:

- **Regression**: HOMO/LUMO energies, solubility
- **Classification**: Toxicity, activity
- **Multi-task**: Predict multiple properties simultaneously

```python
task = PropertyPredictionTask(
    model=model,
    target='homo',
    criterion=nn.MSELoss()
)
```

### Molecule Generation

Generating novel molecules with desired properties:

- **Autoregressive**: Generate atom-by-atom
- **VAE**: Latent space interpolation
- **Diffusion**: Denoising from random noise
- **Flow models**: Invertible transformations

```python
task = MoleculeGenerationTask(
    model=diffusion_model,
    sampling_steps=1000
)
```

### Binding Affinity

Predicting protein-ligand interactions:

```python
task = BindingAffinityTask(
    ligand_model=gcn,
    protein_model=cnn,
    interaction_head=mlp
)
```

---

## Training Workflow

### The Trainer

The `Trainer` orchestrates the training process:

```
┌─────────────┐
│   Trainer   │
├─────────────┤
│  - Loop     │
│  - Logging  │
│  - Checkpoint│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Task     │
├─────────────┤
│  - Forward  │
│  - Loss     │
│  - Optimizer│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Model    │
├─────────────┤
│  - Layers   │
│  - Forward  │
└─────────────┘
```

### Training Loop

```python
for epoch in range(max_epochs):
    for batch in train_loader:
        # Forward pass
        loss = task.training_step(batch)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation
    val_metrics = task.validation_step(val_loader)
```

### Callbacks

Callbacks hook into the training lifecycle:

- **on_train_start**: Setup before training
- **on_train_epoch_start**: Per-epoch setup
- **on_train_batch_end**: Log batch metrics
- **on_validation_epoch_end**: Evaluate on validation set
- **on_train_end**: Cleanup after training

```python
class EarlyStopping(Callback):
    def on_validation_epoch_end(self, trainer, task):
        if should_stop:
            trainer.should_stop = True
```

---

## Reinforcement Learning

### Molecular Environment

The `MoleculeEnv` simulates molecular editing:

- **State**: Current molecular graph
- **Actions**: Add/remove atoms/bonds
- **Rewards**: Chemical property scores
- **Episode**: Sequence of modifications

```python
env = MoleculeEnv(
    initial_smiles="C",
    max_steps=10,
    reward_function=qed_reward
)
```

### Reward Functions

Rewards guide molecule optimization:

| Function | Description | Range |
|----------|-------------|-------|
| QED | Quantitative Estimate of Drug-likeness | [0, 1] |
| LogP | Lipophilicity | (-∞, ∞) |
| SA | Synthetic Accessibility | [1, 10] |
| Custom | User-defined properties | Variable |

### Agents

RL agents learn to optimize molecules:

- **DQN**: Deep Q-Network for discrete actions
- **PPO**: Proximal Policy Optimization
- **SAC**: Soft Actor-Critic

```python
agent = PPOAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr=3e-4
)
```

---

## Evaluation Metrics

### Validity

Percentage of generated molecules that are chemically valid:

```python
from rdkit import Chem

def validity(molecules):
    valid = [mol for mol in molecules if mol is not None]
    return len(valid) / len(molecules)
```

### Uniqueness

Percentage of unique molecules among valid ones:

```python
def uniqueness(molecules):
    smiles = [Chem.MolToSmiles(mol) for mol in molecules]
    return len(set(smiles)) / len(smiles)
```

### Novelty

Percentage of molecules not in the training set:

```python
def novelty(generated, training_set):
    novel = [mol for mol in generated if mol not in training_set]
    return len(novel) / len(generated)
```

### Diversity

Internal diversity of generated molecules:

```python
def diversity(molecules):
    distances = pairwise_tanimoto(molecules)
    return np.mean(distances)
```

---

## Configuration System

Torch Pharma uses YAML for experiment configuration:

```yaml
experiment:
  name: my_experiment
  seed: 42
  tags: [gcn, qm9, baseline]

data:
  dataset: QM9
  batch_size: 32
  num_workers: 4
  train_split: 0.8

model:
  name: GCN
  in_channels: 11
  hidden_channels: 128
  num_layers: 3

task:
  name: PropertyPredictionTask
  target: homo
  criterion: MSELoss
  optimizer: Adam
  lr: 0.001

trainer:
  max_epochs: 100
  accelerator: auto
  
logging:
  use_wandb: true
  project: torch-pharma
  entity: my-team
```

---

## Best Practices

### Data Preprocessing

- **Normalize features** based on training statistics
- **Handle imbalanced data** with stratified sampling
- **Augment data** when appropriate (e.g., conformer generation)

### Model Selection

| Task | Recommended Model | Notes |
|------|-----------------|-------|
| Property Prediction | GCN, MPNN | Fast, good baseline |
| 3D Properties | EGNN, SchNet | SE(3) equivariant |
| Generation | Diffusion, VAE | High quality samples |
| Optimization | RL + GNN | Action space design |

### Training Tips

- **Start simple**: Use GCN baseline before complex models
- **Monitor metrics**: Track validation loss, not just training loss
- **Use early stopping**: Prevent overfitting
- **Log everything**: Use Weights & Biases or TensorBoard
- **Reproduce first**: Ensure you can reproduce paper baselines
