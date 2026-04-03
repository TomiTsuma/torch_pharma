# GCDM: Geometry-Complete Diffusion Model

This example implements the **Geometry-Complete Diffusion Model (GCDM)** from the paper ["Geometric Clifford Perceptron Networks for 3D Molecular Generation and Optimization"](https://arxiv.org/abs/2302.04313).

## Paper Reference

**Title:** Geometric Clifford Perceptron Networks for 3D Molecular Generation and Optimization  
**Authors:** Morehead et al., 2023  
**Paper:** [arXiv:2302.04313](https://arxiv.org/abs/2302.04313)

---

## Overview

GCDM is a **Denoising Diffusion Probabilistic Model (DDPM)** for generating 3D molecular structures. It combines:

- **Equivariant Graph Neural Networks**: EGNN or GCPNet for SE(3)-equivariant message passing
- **Variational Diffusion**: Learned prior distribution over molecular configurations
- **3D Structure Generation**: Generates both atom types and 3D coordinates simultaneously

### Key Features

| Feature | Description |
|---------|-------------|
| **Equivariance** | Respects rotations and translations (SE(3) group) |
| **Joint Generation** | Simultaneously generates atom types and 3D positions |
| **Conditional Generation** | Can condition on molecular properties (e.g., polarizability) |
| **Inpainting** | Can modify specific parts of molecules while keeping others fixed |
| **Stability Metrics** | Evaluates chemical validity of generated structures |

---

## Architecture

```
Input: Random noise (positions + atom types)
  ↓
Denoising Network (GCPNet/EGNN)
  - SE(3)-equivariant message passing
  - Updates both scalar (h) and vector (x) features
  ↓
Output: Denoised molecular structure
  ↓
Iterative Refinement (T=1000 steps)
  ↓
Final: Valid 3D molecular structure
```

### Components

#### 1. Dynamics Network

**GCPNetDynamics** (Recommended):
- Geometric Clifford Perceptron layers
- Superior expressiveness for 3D geometry
- Handles directional features effectively

**EGNNDynamics** (Alternative):
- Equivariant Graph Neural Network
- Simpler architecture, faster training
- Good baseline performance

#### 2. Variational Diffusion

```python
from torch_pharma.models.diffusion.variational_diffusion import EquivariantVariationalDiffusion

ddpm = EquivariantVariationalDiffusion(
    dynamics_network=dynamics_network,
    dataset_info=dataset_info,
    num_atom_types=5,      # C, N, O, F, H (or 4 without H)
    num_x_dims=3,          # 3D coordinates
    num_timesteps=1000,    # Diffusion steps
    loss_type="l2"         # or "nll"
)
```

#### 3. Training Module

```python
from examples.molecule_generation.2302_04313.qm9_mol_gen_ddpm_inference import QM9MoleculeGenerationDDPM

model = QM9MoleculeGenerationDDPM(
    optimizer=torch.optim.AdamW,
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
    dynamics_network="gcpnet",    # or "egnn"
    num_timesteps=1000,
    conditioning=['alpha'],        # Property to condition on
    remove_h=True,                 # Remove hydrogens
    ddpm_mode="unconditional"      # or "inpainting"
)
```

---

## Files

| File | Description |
|------|-------------|
| `qm9_mol_gen_ddpm_train.py` | Training script (if available) |
| `qm9_mol_gen_ddpm_inference.py` | Inference and sampling module |
| `utils.py` | Utility functions for context preparation |

### Location

```
examples/molecule_generation/2302_04313/
├── qm9_mol_gen_ddpm_inference.py
├── qm9_mol_gen_ddpm_train.py (optional)
├── utils.py
└── README.md
```

---

## Usage

### Training

```python
from torch_pharma.data.components.edm import retrieve_dataloaders
from examples.molecule_generation.2302_04313.qm9_mol_gen_ddpm_inference import QM9MoleculeGenerationDDPM

# Load data
dataloader_cfg = {
    "dataset": "QM9",
    "batch_size": 64,
    "num_workers": 4,
    "remove_h": True,
    "include_charges": True,
    "data_dir": "~/.torch_pharma"
}
dataloaders, _ = retrieve_dataloaders(dataloader_cfg)

# Initialize model
model = QM9MoleculeGenerationDDPM(
    optimizer=torch.optim.AdamW,
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
    dynamics_network="gcpnet",
    num_timesteps=1000,
    conditioning=['alpha'],
    remove_h=True,
    dataloaders=dataloaders
)

# Train
for epoch in range(1000):
    for batch in dataloaders['train']:
        metrics = model.training_step(batch, batch_idx)
        # ... optimization step
```

### Sampling

```python
# Generate molecules
model.eval()
x, one_hot, charges, batch_index = model.sample(num_samples=100)

# Convert to RDKit molecules
from torch_pharma.molecules.chemistry import build_molecule

molecules = []
for i in range(num_samples):
    mask = batch_index == i
    pos = x[mask]
    atom_types = one_hot[mask].argmax(-1)
    
    mol = build_molecule(pos, atom_types, model.dataset_info)
    molecules.append(mol)
```

### Conditional Generation

```python
# Condition on specific property value
context = torch.tensor([[alpha_value]])  # Normalized property value

x, one_hot, charges, batch_index = model.sample(
    num_samples=100,
    context=context
)
```

### Inpainting

```python
# Fix part of molecule, generate the rest
node_mask = torch.zeros(num_nodes, dtype=torch.bool)
node_mask[0:5] = True  # Fix first 5 atoms

molecules = model.generate_molecules(
    ddpm_mode="inpainting",
    num_samples=1,
    node_mask=node_mask
)
```

---

## Configuration

### Model Hyperparameters

```yaml
model:
  dynamics_network: "gcpnet"  # or "egnn"
  num_timesteps: 1000
  loss_type: "l2"            # "l2" or "nll"
  include_charges: true
  
dynamics:
  hidden_nf: 128
  n_layers: 9
  attention: true
  tanh: true

training:
  lr: 1e-4
  batch_size: 64
  epochs: 1000
  clip_gradients: true
```

### Dataset Configuration

```yaml
dataset:
  name: "QM9"
  remove_h: true          # Remove hydrogens (recommended)
  include_charges: true     # Include formal charges
  conditioning: ["alpha"]  # Properties to condition on
```

---

## Evaluation Metrics

The model computes several metrics during training:

| Metric | Description |
|----------|-------------|
| `validity` | Fraction of chemically valid molecules |
| `uniqueness` | Fraction of unique molecules |
| `novelty` | Fraction of molecules not in training set |
| `mol_stable` | Fraction of stable molecules |
| `atm_stable` | Fraction of stable atoms |
| `kl_div_atom_types` | KL divergence of atom type distribution |

---

## Key Implementation Details

### Equivariance

The model maintains **SE(3) equivariance**:
- Input rotations produce output rotations
- Input translations produce output translations
- Critical for learning physical 3D structures

### Diffusion Process

1. **Forward Process**: Gradually add noise to molecular coordinates and atom types
2. **Reverse Process**: Learn to denoise using equivariant networks
3. **Sampling**: Iteratively denoise random initial noise

### Loss Functions

**L2 Loss** (Training):
```python
loss_t = 0.5 * error_t / denom
loss_0 = loss_0_x + loss_0_h
```

**NLL** (Evaluation):
```python
nll = loss_t + loss_0 + kl_prior - delta_log_px - log_pN
```

---

## References

1. **GCDM Paper:** Morehead et al., "Geometric Clifford Perceptron Networks for 3D Molecular Generation and Optimization", arXiv:2302.04313, 2023.

2. **EDM (Base):** Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D", ICML 2022.

3. **EGNN:** Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021.

---

## Citation

If you use this implementation, please cite:

```bibtex
@article{morehead2023geometric,
  title={Geometric Clifford Perceptron Networks for 3D Molecular Generation and Optimization},
  author={Morehead, Alex and Chen, Jianmo and Cheng, Yu and Xie, Jian and Davis, Joseph and Xiong, Jianlin and Wang, Chenyu and Wang, Yanli and Xiao, Jinzhuo and Li, Jianlin},
  journal={arXiv preprint arXiv:2302.04313},
  year={2023}
}
```
