# Property Prediction Tutorial

Train a Graph Neural Network to predict molecular properties on the QM9 dataset.

---

## Overview

In this tutorial, you'll learn to:

1. Load and preprocess the QM9 dataset
2. Build a Graph Convolutional Network (GCN)
3. Train the model to predict HOMO energy
4. Evaluate model performance

**Time to complete:** ~15 minutes

---

## Prerequisites

```python
import torch
import torch.nn as nn
from torch_pharma.data import QM9Dataset, DataLoader
from torch_pharma.models import GCN
from torch_pharma.tasks import PropertyPredictionTask
from torch_pharma.training import Trainer
```

---

## Step 1: Load the Dataset

The QM9 dataset contains ~130k molecules with 17 molecular properties.

```python
# Load QM9 dataset
# This will download the data on first run (~100MB)
dataset = QM9Dataset()

print(f"Dataset size: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
```

### Split into Train/Val/Test

```python
# Split dataset
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
```

### Create Data Loaders

```python
# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    num_workers=4
)
```

---

## Step 2: Build the Model

We'll use a Graph Convolutional Network (GCN) with 3 layers.

```python
# Initialize GCN model
model = GCN(
    in_channels=11,        # QM9 has 11 atom features
    hidden_channels=128,   # Hidden dimension
    num_layers=3,          # Number of GCN layers
    out_channels=1,      # Output dimension (single property)
    dropout=0.1
)

print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
```

---

## Step 3: Create the Task

The task combines the model with loss function and optimizer.

```python
# Property prediction task for HOMO energy
task = PropertyPredictionTask(
    model=model,
    target='homo',                    # HOMO energy
    criterion=nn.MSELoss(),            # Mean squared error
    task_type='regression',
    lr=1e-3                           # Learning rate
)
```

### Available QM9 Targets

| Target | Description | Units |
|--------|-------------|-------|
| `mu` | Dipole moment | Debye |
| `alpha` | Polarizability | Bohr³ |
| `homo` | HOMO energy | Hartree |
| `lumo` | LUMO energy | Hartree |
| `gap` | HOMO-LUMO gap | Hartree |
| `r2` | Electronic spatial extent | Bohr² |
| `zpve` | Zero point vibrational energy | Hartree |
| `U0` | Internal energy at 0K | Hartree |
| `U` | Internal energy at 298.15K | Hartree |
| `H` | Enthalpy at 298.15K | Hartree |
| `G` | Free energy at 298.15K | Hartree |
| `Cv` | Heat capacity | cal/(mol·K) |

---

## Step 4: Train the Model

```python
# Initialize trainer
trainer = Trainer(
    max_epochs=50,
    accelerator='auto',      # Use GPU if available
    log_every_n_steps=50,
    callbacks=[
        EarlyStopping(monitor='val_mae', patience=10, mode='min'),
        ModelCheckpoint(monitor='val_mae', save_top_k=3)
    ]
)

# Train
trainer.fit(task, train_loader, val_loader)
```

---

## Step 5: Evaluate

```python
# Evaluate on test set
test_metrics = trainer.test(task, test_loader)
print(f"Test MAE: {test_metrics['test_mae']:.6f} Hartree")

# Make predictions
predictions = trainer.predict(task, test_loader)
```

---

## Complete Code

```python
import torch
import torch.nn as nn
from torch_pharma.data import QM9Dataset, DataLoader
from torch_pharma.models import GCN
from torch_pharma.tasks import PropertyPredictionTask
from torch_pharma.training import Trainer, EarlyStopping, ModelCheckpoint

# 1. Load dataset
dataset = QM9Dataset()

# Split
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 2. Build model
model = GCN(
    in_channels=11,
    hidden_channels=128,
    num_layers=3,
    out_channels=1,
    dropout=0.1
)

# 3. Create task
task = PropertyPredictionTask(
    model=model,
    target='homo',
    criterion=nn.MSELoss(),
    task_type='regression',
    lr=1e-3
)

# 4. Train
trainer = Trainer(
    max_epochs=50,
    accelerator='auto',
    callbacks=[
        EarlyStopping(monitor='val_mae', patience=10),
        ModelCheckpoint(monitor='val_mae', save_top_k=3)
    ]
)

trainer.fit(task, train_loader, val_loader)

# 5. Evaluate
test_metrics = trainer.test(task, test_loader)
print(f"Final Test MAE: {test_metrics['test_mae']:.6f}")
```

---

## Next Steps

- Try different targets (gap, lumo, etc.)
- Experiment with larger models (MPNN, EGNN)
- Use multi-task learning
- Explore the [Molecule Generation Tutorial](molecule_generation.md)
