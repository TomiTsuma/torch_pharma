# Molecule Generation Tutorial

Generate novel 3D molecules using diffusion models.

---

## Overview

In this tutorial, you'll learn to:

1. Set up a Geometry-Complete Diffusion Model (GCDM)
2. Train on the QM9 dataset for 3D molecular generation
3. Sample and evaluate new molecules with 3D structures
4. Use conditional generation for property optimization

**Time to complete:** ~45 minutes

---

## Background

### GCDM: Geometry-Complete Diffusion Model

This tutorial uses **GCDM** from the paper ["Geometric Clifford Perceptron Networks for 3D Molecular Generation and Optimization"](https://arxiv.org/abs/2302.04313) (Morehead et al., 2023).

**Key Features:**
- **3D Structure Generation**: Generates both atom types and 3D coordinates
- **SE(3) Equivariance**: Respects rotations and translations
- **Variational Diffusion**: Uses learned prior for better generation quality
- **Property Conditioning**: Can generate molecules with specific properties

### Architecture

```
Input: Random noise (positions + atom types)
    ↓
Denoising Network (GCPNet or EGNN)
    - SE(3)-equivariant message passing
    - Updates both scalar (h) and vector (x) features
    ↓
Output: Denoised molecular structure
    ↓
Iterative Refinement (T=1000 steps)
    ↓
Final: Valid 3D molecular structure
```

---

## Step 1: Setup

```python
import torch
from torch_pharma.data.components.edm import retrieve_dataloaders
from examples.molecule_generation.2302_04313.qm9_mol_gen_ddpm_inference import (
    QM9MoleculeGenerationDDPM
)
```

---

## Step 2: Load Data

```python
# Configure data loading
dataloader_cfg = {
    "dataset": "QM9",
    "batch_size": 64,
    "num_workers": 4,
    "remove_h": True,          # Remove hydrogens (recommended)
    "include_charges": True,   # Include formal charges
    "data_dir": "~/.torch_pharma",
    "subtract_thermo": True,
    "force_download": False,
    "num_radials": 1,
    "device": "cuda",
}

# Load data
dataloaders, charge_scale = retrieve_dataloaders(dataloader_cfg)

print(f"Train batches: {len(dataloaders['train'])}")
print(f"Validation batches: {len(dataloaders['valid'])}")
```

---

## Step 3: Initialize the Model

```python
# Define optimizer and scheduler
def optimizer_fn(params):
    return torch.optim.AdamW(params, lr=1e-4)

def scheduler_fn(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
    )

# Initialize GCDM model
model = QM9MoleculeGenerationDDPM(
    optimizer=optimizer_fn,
    scheduler=scheduler_fn,
    dynamics_network="gcpnet",        # or "egnn"
    num_timesteps=1000,
    num_eval_samples=1000,
    conditioning=['alpha'],           # Condition on polarizability
    remove_h=True,
    ddpm_mode="unconditional",        # or "inpainting"
    loss_type="l2",
    num_atom_types=5,                 # C, N, O, F, H
    num_x_dims=3,                     # 3D coordinates
    include_charges=True,
    clip_gradients=True,
    dataloaders=dataloaders           # Required for conditioning
)

# Move to GPU
model = model.cuda()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Model Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `dynamics_network` | `"gcpnet"`, `"egnn"` | Base architecture |
| `ddpm_mode` | `"unconditional"`, `"inpainting"` | Generation mode |
| `loss_type` | `"l2"`, `"nll"` | Training objective |
| `conditioning` | List of properties | Property to condition on |

---

## Step 4: Training

```python
# Training loop
for epoch in range(1000):
    model.train()
    
    # Training
    for batch_idx, batch in enumerate(dataloaders['train']):
        batch = batch.to(model.device)
        
        # Forward pass
        metrics = model.training_step(batch, batch_idx)
        loss = metrics['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        model.configure_gradient_clipping(
            model.configure_optimizers()['optimizer'],
            gradient_clip_val=1.0
        )
        
        # Update weights
        optimizer = model.configure_optimizers()['optimizer']
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloaders['valid']):
            batch = batch.to(model.device)
            metrics = model.validation_step(batch, batch_idx)
        
        model.on_validation_epoch_end()
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(
            model.state_dict(),
            f"checkpoints/gcdm_epoch_{epoch}.pt"
        )
```

---

## Step 5: Sampling

### Unconditional Generation

```python
model.eval()

# Generate 100 molecules
x, one_hot, charges, batch_index = model.sample(num_samples=100)

print(f"Generated positions: {x.shape}")
print(f"Generated atom types: {one_hot.shape}")
print(f"Batch indices: {batch_index.shape}")
```

### Conditional Generation

```python
# Condition on specific polarizability value
alpha_value = 75.0  # Example value

# Normalize using training statistics
alpha_normalized = (
    alpha_value - model.props_norms['alpha']['mean']
) / model.props_norms['alpha']['mad']

context = torch.tensor([[alpha_normalized]] * 100).to(model.device)

# Sample with conditioning
x, one_hot, charges, batch_index = model.sample(
    num_samples=100,
    context=context
)
```

---

## Step 6: Convert to RDKit Molecules

```python
from torch_pharma.molecules.chemistry import build_molecule, process_molecule
from rdkit import Chem

molecules = []
for i in range(100):
    mask = batch_index == i
    pos = x[mask].cpu()
    atom_types = one_hot[mask].argmax(-1).cpu()
    
    # Build RDKit molecule
    mol = build_molecule(
        positions=pos,
        atom_types=atom_types,
        dataset_info=model.dataset_info,
        add_coords=True
    )
    
    # Process and sanitize
    mol = process_molecule(
        rdmol=mol,
        add_hydrogens=True,
        sanitize=True,
        relax_iter=200
    )
    
    if mol is not None:
        molecules.append(mol)

print(f"Generated {len(molecules)} valid molecules")
```

---

## Step 7: Evaluate

```python
# Use model's built-in evaluation
results = model.sample_and_analyze(
    num_samples=1000,
    batch_size=64,
    save_molecules=True,
    output_dir="output/molecules"
)

print("Evaluation Results:")
print(f"  Validity: {results['validity']:.3f}")
print(f"  Uniqueness: {results['uniqueness']:.3f}")
print(f"  Novelty: {results['novelty']:.3f}")
print(f"  Molecular Stability: {results['mol_stable']:.3f}")
print(f"  Atom Stability: {results['atm_stable']:.3f}")
```

---

## Step 8: Visualization

```python
# Save and visualize generated molecules
model.sample_and_save(
    num_samples=10,
    num_timesteps=1000,
    sampling_output_dir="output/visualizations"
)

# Visualize diffusion chain
model.sample_chain_and_save(
    keep_frames=100,
    num_tries=5
)
```

---

## Complete Code

```python
import torch
from torch_pharma.data.components.edm import retrieve_dataloaders
from examples.molecule_generation.2302_04313.qm9_mol_gen_ddpm_inference import (
    QM9MoleculeGenerationDDPM
)

# 1. Load data
dataloader_cfg = {
    "dataset": "QM9",
    "batch_size": 64,
    "num_workers": 4,
    "remove_h": True,
    "include_charges": True,
    "data_dir": "~/.torch_pharma",
}
dataloaders, _ = retrieve_dataloaders(dataloader_cfg)

# 2. Initialize model
def optimizer_fn(params): return torch.optim.AdamW(params, lr=1e-4)
def scheduler_fn(optimizer): return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

model = QM9MoleculeGenerationDDPM(
    optimizer=optimizer_fn,
    scheduler=scheduler_fn,
    dynamics_network="gcpnet",
    num_timesteps=1000,
    conditioning=['alpha'],
    remove_h=True,
    dataloaders=dataloaders
).cuda()

# 3. Train (simplified)
for epoch in range(100):
    for batch in dataloaders['train']:
        batch = batch.to(model.device)
        metrics = model.training_step(batch, 0)
        # ... optimization code

# 4. Generate
model.eval()
x, one_hot, charges, batch_index = model.sample(num_samples=100)

# 5. Evaluate
results = model.sample_and_analyze(num_samples=1000)
print(f"Validity: {results['validity']:.3f}")
```

---

## Tips for Better Generation

1. **Train longer**: GCDM benefits from 1000+ epochs
2. **Use GCPNet**: Generally outperforms EGNN for 3D generation
3. **Condition on properties**: Guides generation toward desired chemical space
4. **Tune temperature**: Affects sampling diversity (not shown in basic API)
5. **Filter by stability**: Post-process to keep only stable molecules

---

## Advanced: Inpainting

Modify parts of molecules while keeping others fixed:

```python
# Fix first 5 atoms, generate the rest
num_nodes = torch.tensor([20])
node_mask = torch.zeros(20, dtype=torch.bool)
node_mask[0:5] = True  # These will be fixed

molecules = model.generate_molecules(
    ddpm_mode="inpainting",
    num_samples=1,
    num_nodes=num_nodes,
    node_mask=node_mask,
    sanitize=True,
    add_hydrogens=True
)
```

---

## References

- [GCDM Paper](https://arxiv.org/abs/2302.04313) - Morehead et al., 2023
- [EDM Paper](https://arxiv.org/abs/2203.17003) - Hoogeboom et al., 2022
- Example code: `examples/molecule_generation/2302_04313/`
