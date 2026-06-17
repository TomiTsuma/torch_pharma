# Refactoring Guide: Consolidation of QM9 Dataset Classes via Factory Pattern

This document details the step-by-step instructions to refactor and consolidate the various `QM9` and `QM9Dataset` classes across `torch_pharma` using a **Factory Design Pattern**.

---

## 1. Context & Objectives

Currently, there are three distinct QM9 dataset definitions in the codebase:
1. **`QM9` in `torch_pharma/data/components/qm9/qm9_dataset.py` (Lines 228-683)**: An unused/legacy class.
2. **`QM9Dataset` in `torch_pharma/data/components/qm9/qm9_dataset.py` (Lines 691-905)**: A subclass of PyG `InMemoryDataset` used by `QM9DataModule` (graph-based).
3. **`QM9Dataset` in `torch_pharma/data/datasets/qm9.py` (Lines 48-289)**: A dataset class subclassing `BaseDataset` (numpy-based returning SMILES, raw `.npz` creation).

### Why the Factory Pattern?
Since the PyG-based and the raw/SMILES-based datasets serve fundamentally different purposes but are both needed, we will consolidate the duplicate classes and implement a **Factory Class** to resolve the correct variant at runtime.

---

## 2. Step-by-Step Refactoring Instructions

### Step 2.1: Consolidate components in `torch_pharma/data/components/qm9/qm9_dataset.py`

1. Open `torch_pharma/data/components/qm9/qm9_dataset.py`.
2. Locate and **delete** the unused `QM9` class (lines 228 to 683).
3. Rename the class `QM9Dataset` (line 691) to `QM9`:
   ```python
   class QM9(InMemoryDataset):
   ```
4. Update its constructor `__init__` signature to allow default arguments:
   ```python
   def __init__(
       self,
       split: str = "train",
       root: str = os.path.join(TORCH_PHARMA_HOME, "QM9"),
       remove_h: bool = False,
       transform=None,
       pre_transform=None,
       pre_filter=None,
       only_stats=False,
       **kwargs
   ):
   ```
5. Inside the same file, locate `QM9DataModule` and update references from `QM9Dataset` to `QM9`:
   ```python
   train_dataset = QM9(
       split="train", root=root_path, remove_h=cfg.remove_hs, only_stats=only_stats
   )
   val_dataset = QM9(split="val", root=root_path, remove_h=cfg.remove_hs, only_stats=only_stats)
   test_dataset = QM9(split="test", root=root_path, remove_h=cfg.remove_hs, only_stats=only_stats)
   ```

---

### Step 2.2: Implement the Factory in `torch_pharma/data/datasets/qm9.py`

1. Open `torch_pharma/data/datasets/qm9.py`.
2. Rename the existing `QM9Dataset` class (line 48) to `RawQM9Dataset`:
   ```python
   class RawQM9Dataset(BaseDataset):
       # Keep the existing methods (download, process, compute_smiles, etc.) intact.
   ```
3. At the bottom of the file, define the `QM9DatasetFactory` class:
   ```python
   class QM9DatasetFactory:
       @staticmethod
       def get_dataset(dataset_type="pyg", **kwargs):
           """
           Factory method to resolve and instantiate the requested QM9 dataset variant.
           
           Parameters:
               dataset_type (str): Either "pyg" (or "graph") for PyTorch Geometric representation, 
                                   or "raw" (or "smiles") for the raw numpy-based representation.
           """
           if dataset_type in ("pyg", "graph"):
               from torch_pharma.data.components.qm9.qm9_dataset import QM9
               return QM9(**kwargs)
           elif dataset_type in ("raw", "smiles"):
               return RawQM9Dataset(**kwargs)
           else:
               raise ValueError(f"Unknown dataset type: {dataset_type}")
   ```
4. For backward compatibility, map `QM9Dataset` to call the factory or instantiate `RawQM9Dataset` by default:
   ```python
   # Alias for legacy codebases which expect QM9Dataset to refer to the raw numpy dataset:
   QM9Dataset = RawQM9Dataset
   ```

---

## 3. Dependency Impact & Usage Examples

### 3.1 Fetching Graph (PyG) Dataset
```python
from torch_pharma.data.datasets.qm9 import QM9DatasetFactory

pyg_dataset = QM9DatasetFactory.get_dataset(dataset_type="pyg", split="train")
```

### 3.2 Fetching Raw (SMILES) Dataset
```python
from torch_pharma.data.datasets.qm9 import QM9DatasetFactory

raw_dataset = QM9DatasetFactory.get_dataset(dataset_type="raw")
```
