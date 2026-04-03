# Datasets

Built-in datasets for molecular machine learning.

---

## Available Datasets

<div class="grid cards" markdown>

- :material-database: **QM9**

    ---

    ~130k molecules with 17 molecular properties.

    [:octicons-arrow-right-24: Details](qm9.md)

- :material-flask: **ZINC**

    ---

    ~230M commercially available compounds.

    [:octicons-arrow-right-24: Details](zinc.md)

- :material-link-variant: **BindingDB**

    ---

    Protein-ligand binding affinities.

    [:octicons-arrow-right-24: Details](bindingdb.md)

- :material-plus-circle: **Custom**

    ---

    Load your own molecular datasets.

    [:octicons-arrow-right-24: Guide](custom.md)

</div>

---

## Quick Reference

| Dataset | Size | Task Type | Properties |
|---------|------|-----------|------------|
| QM9 | 130k | Property Prediction | 17 properties |
| ZINC | 230M | Generation | Drug-like structures |
| BindingDB | 2.7M | Affinity Prediction | Ki, Kd, IC50 |

---

## Common API

All datasets inherit from `BaseDataset`:

```python
from torch_pharma.data import QM9Dataset

dataset = QM9Dataset()

# Get length
len(dataset)

# Get item
data = dataset[idx]

# Access properties
print(data.smiles)      # SMILES string
print(data.y)          # Target values
print(data.pos)        # 3D coordinates
print(data.x)          # Node features
print(data.edge_index)   # Edge connectivity
```

---

## Loading Data

```python
from torch_pharma.data import DataLoader

# Create loader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate
for batch in loader:
    print(batch.x)          # Node features
    print(batch.edge_index)   # Edges
    print(batch.batch)        # Batch assignment
    print(batch.y)            # Targets
```

---

## Data Directory

Datasets are downloaded to `~/.torch_pharma/` by default:

```
~/.torch_pharma/
├── QM9/
│   ├── raw/
│   └── processed/
├── ZINC/
│   └── ...
```

Change location with `root` parameter:

```python
dataset = QM9Dataset(root="/path/to/data")
```

---

## Custom Datasets

See [Custom Datasets Guide](custom.md) for loading your own data.

---

## Citation

When using these datasets, please cite the original sources:

**QM9**:
```
Ruddigkeit et al., "Enumeration of 166 billion organic small molecules
in the chemical universe database GDB-17", J. Chem. Inf. Model. 2012
```

**ZINC**:
```
Irwin et al., "ZINC – A Free Database of Commercially Available
Compounds for Virtual Screening", J. Chem. Inf. Model. 2005
```

**BindingDB**:
```
Liu et al., "BindingDB: a web-accessible database of experimentally
determined protein–ligand binding affinities", Nucleic Acids Res. 2007
```
