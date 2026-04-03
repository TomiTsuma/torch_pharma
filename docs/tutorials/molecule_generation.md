# Molecule Generation Tutorial

Generate novel molecules using diffusion models.

---

## Overview

In this tutorial, you'll learn to:

1. Load molecular data for training
2. Build a diffusion model (EDM)
3. Train the model to generate 3D molecular structures
4. Sample and evaluate new molecules

**Time to complete:** ~30 minutes

---

## Background

Diffusion models generate data by learning to reverse a gradual noising process. In molecular generation, this allows creating chemically valid molecules with realistic 3D structures.

---

## Step 1: Setup

```python
import torch
from torch_pharma.data import QM9Dataset, DataLoader
from torch_pharma.models import EDMDiffusion
from torch_pharma.tasks import MoleculeGenerationTask
from torch_pharma.training import Trainer
from torch_pharma.evaluation import BasicMolecularMetrics, ScoringFunction
```

---

## Step 2: Load Data

```python
# Load QM9 dataset for training
dataset = QM9Dataset()
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Training on {len(dataset)} molecules")
```

---

## Step 3: Build the Diffusion Model

```python
# Initialize EDM model
model = EDMDiffusion(
    n_atoms=29,          # Maximum atoms per molecule
    n_atom_types=5,      # H, C, N, O, F
    hidden_dim=256,      # Hidden dimension
    n_layers=9,          # Number of layers
    timesteps=1000,      # Diffusion steps
    noise_schedule="cosine"
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Step 4: Create Generation Task

```python
task = MoleculeGenerationTask(
    model=model,
    generation_type="diffusion",
    sampling_steps=1000,
    temperature=1.0,
    lr=2e-4
)
```

---

## Step 5: Train

```python
trainer = Trainer(
    max_epochs=1000,
    accelerator='auto',
    log_every_n_steps=100
)

trainer.fit(task, train_loader)
```

---

## Step 6: Generate Molecules

```python
# Generate 100 new molecules
generated_mols = task.sample(
    n_samples=100,
    n_atoms=29
)

print(f"Generated {len(generated_mols)} molecules")

# Print some SMILES
for i, mol in enumerate(generated_mols[:5]):
    smiles = mol.to_smiles() if hasattr(mol, 'to_smiles') else str(mol)
    print(f"Molecule {i+1}: {smiles}")
```

---

## Step 7: Evaluate

```python
from rdkit import Chem
from torch_pharma.evaluation import BasicMolecularMetrics

# Convert to RDKit
rdmols = []
for mol in generated_mols:
    if isinstance(mol, str):
        rdmol = Chem.MolFromSmiles(mol)
    else:
        rdmol = mol.to_rdkit() if hasattr(mol, 'to_rdkit') else None
    if rdmol:
        rdmols.append(rdmol)

# Compute metrics
metrics = BasicMolecularMetrics(
    dataset_info=dataset_info,
    data_dir="data"
)

validity, uniqueness, novelty = metrics.evaluate_rdmols(rdmols, verbose=True)

# Compute QED scores
qed_scores = [ScoringFunction.qed_score(m) for m in rdmols]
print(f"Average QED: {sum(qed_scores)/len(qed_scores):.3f}")
```

---

## Complete Code

```python
import torch
from torch_pharma.data import QM9Dataset, DataLoader
from torch_pharma.models import EDMDiffusion
from torch_pharma.tasks import MoleculeGenerationTask
from torch_pharma.training import Trainer
from torch_pharma.evaluation import BasicMolecularMetrics
from rdkit import Chem

# 1. Load data
dataset = QM9Dataset()
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. Build model
model = EDMDiffusion(
    n_atoms=29,
    n_atom_types=5,
    hidden_dim=256,
    n_layers=9,
    timesteps=1000
)

# 3. Create task
task = MoleculeGenerationTask(
    model=model,
    generation_type="diffusion",
    lr=2e-4
)

# 4. Train
trainer = Trainer(max_epochs=1000, accelerator='auto')
trainer.fit(task, train_loader)

# 5. Generate
generated_mols = task.sample(n_samples=100)

# 6. Evaluate
rdmols = [m.to_rdkit() for m in generated_mols]
metrics = BasicMolecularMetrics(dataset_info, "data")
scores = metrics.evaluate_rdmols(rdmols)
```

---

## Tips for Better Generation

1. **Train longer**: Diffusion models need many epochs
2. **Use more data**: Train on ZINC for diversity
3. **Tune temperature**: Lower = more conservative
4. **Filter by property**: Use property prediction models as filters
