# Deep Dive: The QM9 Dataset

The QM9 dataset (or GDB-9) is a collection of ~134k small organic molecules. In this repository, the dataset is processed from raw scientific archives into high-performance NumPy formats (`.npz`) for deep learning.

This document provides a technical walkthrough of how the data is retrieved, filtered, transformed, and stored.

---

## 1. Data Retrieval: The Source Files

The process begins in `src/datamodules/components/edm/qm9.py`. The function `download_dataset_qm9` orchestrates the retrieval of three fundamental files from Figshare.

### 1.1 `dsgdb9nsd.xyz.tar.bz2` (The Coordinates)
*   **URL**: `https://springernature.figshare.com/ndownloader/files/3195389`
*   **Purpose**: Contains 133,885 XYZ files.
*   **Derivation**:
    ```python
    # src/datamodules/components/edm/qm9.py
    gdb9_url_data = "https://springernature.figshare.com/ndownloader/files/3195389"
    gdb9_tar_data = join(gdb9dir, "dsgdb9nsd.xyz.tar.bz2")
    urllib.request.urlretrieve(gdb9_url_data, filename=gdb9_tar_data)
    ```

### 1.2 `uncharacterized.txt` (The Exclusion List)
*   **URL**: `https://springernature.figshare.com/ndownloader/files/3195404`
*   **Purpose**: Lists 3,054 molecules that failed geometric or chemical consistency checks in the original study.
*   **Derivation**:
    ```python
    # src/datamodules/components/edm/qm9.py: line 107
    gdb9_url_excluded = "https://springernature.figshare.com/ndownloader/files/3195404"
    gdb9_txt_excluded = join(gdb9dir, "uncharacterized.txt")
    urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)
    ```

### 1.3 `atomref.txt` (The Thermochemical Reference)
*   **URL**: `https://springernature.figshare.com/ndownloader/files/3195395`
*   **Purpose**: Contains ground-state energies for H, C, N, O, and F atoms.
*   **Derivation**:
    ```python
    # src/datamodules/components/edm/qm9.py: line 172
    gdb9_url_thermo = "https://springernature.figshare.com/ndownloader/files/3195395"
    gdb9_txt_thermo = join(gdb9dir, "atomref.txt")
    urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)
    ```

---

## 2. Processing Pipeline: From XYZ to Tensor

### 2.1 Filtering & Indexing
The function `gen_splits_gdb9` identifies the "clean" subset of molecules. It subtracts the indices in `uncharacterized.txt` from the full 133,885 set.

```python
# src/datamodules/components/edm/qm9.py: line 125
Ngdb9 = 133885
included_idxs = np.array(sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))
```

### 2.2 Parsing the XYZ Format
The file `src/datamodules/components/edm/process.py` contains `process_xyz_gdb9`, which handles the line-by-line parsing of each extracted molecule:

*   **Line 1**: `num_atoms`
*   **Line 2**: Properties (parsed into a dictionary using `prop_strings`)
*   **Lines 3 to N+2**: Atoms (Symbol $\rightarrow$ Atomic Number) and Coordinates (float).

### 2.3 Padding and Stacking
Molecules have varying atom counts (5 to 29). To allow for efficient matrix operations in PyTorch, the system pads all molecules to the maximum size of **29** using `torch.nn.utils.rnn.pad_sequence`.

```python
# src/datamodules/components/edm/process.py: line 101
if stack:
    molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 
                 else torch.stack(val) for key, val in molecules.items()}
```

---

## 3. The Final Form: `.npz` Deep Dive

The processed data is saved as `train.npz`, `valid.npz`, and `test.npz`.

### 3.1 Python Data Specification
| Key | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `num_atoms` | `int64` | `[N]` | Actual number of atoms before padding. |
| `charges` | `float32`| `[N, 29]` | Atomic numbers ($1=H, 6=C, 7=N, 8=O, 9=F$). |
| `positions` | `float32`| `[N, 29, 3]` | Cartesian coordinates in Angstroms. |
| `A`, `B`, `C` | `float64`| `[N]` | Rotational constants (GHz). |
| `mu` | `float64`| `[N]` | Dipole moment (Debye). |
| `alpha` | `float64`| `[N]` | Isotropic polarizability ($a_0^3$). |
| `homo` | `float64`| `[N]` | Energy of Highest Occupied Molecular Orbital (Ha). |
| `lumo` | `float64`| `[N]` | Energy of Lowest Unoccupied Molecular Orbital (Ha). |
| `gap` | `float64`| `[N]` | Energy gap (Ha). |
| `r2` | `float64`| `[N]` | Electronic spatial extent ($a_0^2$). |
| `zpve` | `float64`| `[N]` | Zero point vibrating energy (Ha). |
| `U0` | `float64`| `[N]` | Internal energy at 0K (Ha). |
| `U` | `float64`| `[N]` | Internal energy at 298.15K (Ha). |
| `H` | `float64`| `[N]` | Enthalpy at 298.15K (Ha). |
| `G` | `float64`| `[N]` | Free energy at 298.15K (Ha). |
| `Cv` | `float64`| `[N]` | Heat capacity at 298.15K (cal/mol K). |
| `*_thermo` | `float64`| `[N]` | Reference energy for the atoms in the molecule. |

### 3.2 Biochemical & Physical Perspective

*   **Atomic Charges & Positions**: These represent the **Molecular Graph** in 3D Euclidean space. The charges define the "identity" of the nodes (nuclei), while positions define the geometry.
*   **Stationary States ($U_0, H, G$)**: These are thermodynamic potentials. $U_0$ is the "frozen" electronic energy. $G$ (Gibbs Free Energy) is critical in biology; it determines if a molecule will spontaneously react or bind to a protein.
*   **HOMO/LUMO Gap**: This is the "Chemical Hardness." A large gap implies a stable, unreactive molecule. A small gap implies a highly reactive molecule, often seen in colored dyes or active binding sites.
*   **Rotational Constants ($A, B, C$)**: These relate to the molecule's **Moment of Inertia**. They are used in spectroscopy to determine the exact bond lengths and angles experimentally.
*   **Atomization Energy (`prop - prop_thermo`)**: By subtracting the thermochemical reference, we isolate the energy held in the **Chemical Bonds** itself, removing the constant background energy of the nuclei.

---

## 4. Evaluation Component: `QM9_smiles.pickle`

Generated in `src/datamodules/components/edm/rdkit_functions.py`, this file contains the SMILES strings for the training set.

**Derivation snippet**:
```python
# src/datamodules/components/edm/rdkit_functions.py: line 70
mol = build_molecule(torch.tensor(positions), torch.tensor(atom_type), dataset_info, charges)
mol = mol2smiles(mol)
if mol is not None:
    mols_smiles.append(mol)
# ...
with open(file_path, "wb") as f:
    pickle.dump(qm9_smiles, f)
```

This file allows for **In-distribution vs. Out-of-distribution** analysis: checking if the generative AI is discovering new chemical structures or merely replicating the dataset.

---

## 5. Data Access Pattern

In this project, the `.npz` files are accessed and converted into PyTorch tensors before being wrapped in a `ProcessedDataset`.

### 5.1 Programmatic Loading
The function `initialize_datasets` in `src/datamodules/components/edm/utils.py` handles the raw loading:

```python
# src/datamodules/components/edm/utils.py: line 145
# Load downloaded/processed datasets
datasets = {}
for split, datafile in datafiles.items():
    with np.load(datafile) as f:
        datasets[split] = {key: torch.from_numpy(val) for key, val in f.items()}
```

### 5.2 Understanding the `NpzFile` Structure
When you run `vars(data)` on an object returned by `np.load()`, you see the internal state of the NumPy `NpzFile` class (as seen in the project's notebooks):
*   **`files`**: A list of high-level keys (e.g., `'num_atoms'`, `'charges'`). These are what you use for indexing: `data['charges']`.
*   **`_files`**: A list of the actual `.npy` filenames stored inside the compressed `.npz` ZIP container (e.g., `'num_atoms.npy'`).

Each key in `f.items()` corresponds to one of these internal files, which the project then immediately converts from NumPy arrays to PyTorch tensors (`torch.from_numpy`) to enable seamless integration with the training loops.
