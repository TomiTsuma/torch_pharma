# Data Pipeline

## SELFIES + Graph Batch

Each training sample provides:

- `x`, `pos`, `edge_index`, `edge_attr` — PyG graph
- `selfies`, `smiles`, `rdmol`
- `rdmol2selfies`, `rdmol2selfies_mask` — atom-to-token alignment
- `context` — optional normalized property (conditional generation)

## Collaters

| Collater | File | Use |
|----------|------|-----|
| `QM9Collater` | `collators.py` | DMT training with VP noise |
| `QM9InferCollater` | `collators.py` | Inference (random init pos) |
| `LMCollater` | `collators.py` | MoLlama-only training |

## Noise Injection

`QM9Collater.add_noise()` samples `t ~ U(0,1)`, computes `(alpha_t, sigma_t)` from `NoiseScheduleVPV2`, and sets:

```
pos_noisy = alpha_t * pos + sigma_t * noise
```

## Dataset Downloads

Preprocess with NExT-Mol scripts or download OSF caches:

- **QM9:** [OSF QM92014.zip](https://osf.io/download/uv6a4/)
- **Geom-DRUGS:** Google Drive link in NExT-Mol `docs/DATASETS.md`

Place processed `.pt` files under **`TORCH_PHARMA_HOME`** (default `~/.torch_pharma`):

| Dataset | OSF source | Path |
|---------|------------|------|
| MoLlama (QM9 LM) | `QM92014/QM92014/` | `$TORCH_PHARMA_HOME/data/nextmol/qm9_lm/` |
| DMT (QM9 TorDF) | `GEOM-QM9/GEOM-QM9/` | `$TORCH_PHARMA_HOME/data/nextmol/tordf_qm9/` |
| Geom-DRUGS | `geom_drugs_jodo.zip` | `$TORCH_PHARMA_HOME/data/nextmol/geom_drugs/` |

Install from a local OSF archive (e.g. `~/Downloads/osfstorage-archive`):

```bash
py -3.11 scripts/install_nextmol_data.py
py -3.11 scripts/install_nextmol_data.py --source "C:/path/to/osfstorage-archive/datasets"
```

Constants are defined in `torch_pharma/paths.py` (`NEXTMOL_QM9_LM`, etc.). Directories are created automatically via `ensure_nextmol_dirs()`.

### Verify installation

After `install_nextmol_data.py`, check required files:

```python
from pathlib import Path
from torch_pharma.paths import NEXTMOL_QM9_LM, NEXTMOL_QM9_TORDF

checks = {
    "qm9_lm": (NEXTMOL_QM9_LM / "processed" / "data_qm9.pt").exists(),
    "tordf_train": (NEXTMOL_QM9_TORDF / "processed_train.pt").exists(),
    "tordf_val": (NEXTMOL_QM9_TORDF / "processed_val.pt").exists(),
}
assert all(checks.values()), f"Missing: {[k for k, v in checks.items() if not v]}"
```

Or run training smoke tests from [14-examples-and-runbook.md](14-examples-and-runbook.md):

```bash
python examples/molecule_generation/next_mol/train_mollama.py --max_epochs 1 --batch_size 2
```

## Mol Mapping

`mol_mapping.py` ports `build_rdkit2cano_smiles_withoutH_mapping` and `get_smiles2selfies_mapping` for constructing `rdmol2selfies` tensors.
