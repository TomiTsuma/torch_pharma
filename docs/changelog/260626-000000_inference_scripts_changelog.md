# Inference Scripts for `examples/molecule_generation`

**Date:** 2026-06-26  
**Scope:** `examples/molecule_generation/`

## Summary

Added or completed inference scripts for all three molecule-generation examples so each can be used to make predictions from a trained checkpoint without re-running training code.

---

## Changes

### `2302_04313` — QM9 DDPM (arXiv:2302.04313)

**File:** `examples/molecule_generation/2302_04313/qm9_mol_gen_ddpm_inference.py`  
**Status:** Was empty (1 line). Now fully implemented.

- Defines `QM9DDPMInference`, a minimal `nn.Module` wrapper around `EquivariantVariationalDiffusion` that **does not** import the training script (avoids triggering WandB/MLflow initialization from the `@track_gnn_activations` decorator).
- Mirrors the exact module layout saved by `qm9_mol_gen_ddpm_train.py` (`self.ddpm.*` keys), so `load_state_dict(strict=False)` restores all DDPM weights cleanly.
- `GCPNetDynamics` is initialized without a `conditioning` kwarg (matching training defaults), keeping the network architecture identical to what was trained.
- `generate()` method handles batched sampling, building RDKit `Mol` objects via `build_molecule` + `process_molecule`.
- **Conditional vs. unconditional:** Zero context (mean normalized property) is used by default. Pass `--load_data` to sample alpha values from the real QM9 training distribution.
- Outputs one SMILES per line.

**CLI:**
```bash
# Unconditional (zero context):
python qm9_mol_gen_ddpm_inference.py --checkpoint checkpoints/model_0.pt --num_samples 100

# Sample conditioning from QM9 dataset:
python qm9_mol_gen_ddpm_inference.py --checkpoint checkpoints/model_0.pt --load_data --num_samples 100
```

---

### `2302_13971` — MoLlama + DMT (arXiv:2302.13971)

**New files:**

#### `infer_mollama.py` — Stage 1: SELFIES language model generation

- Loads `acharkq/MoLlama` (or a fine-tuned variant) and generates SELFIES strings autoregressively from a BOS token.
- SELFIES → canonical SMILES conversion via the `selfies` library + RDKit validation.
- Supports three checkpoint formats: HuggingFace directory, `.pt` state-dict file, or base weights only.
- Auto-detects latest checkpoint from `~/.torch_pharma/checkpoints/nextmol/mollama/` when no `--checkpoint` is given.

**CLI:**
```bash
python infer_mollama.py --num_samples 200 --output generated.smi
python infer_mollama.py --checkpoint path/to/ckpt --temperature 0.8 --num_samples 500
```

#### `infer_dmt.py` — Stage 2/3: 3D conformer generation

- Loads a `NextMolDMTModule` checkpoint (saved by `train_dmt_uncond.py` or `train_dmt_cond.py`).
- Uses `QM9InferCollater` (already in `torch_pharma`) which initialises atomic positions from random noise — no reference conformer needed.
- Runs the reverse VP-SDE sampler (`reverse_vp_sde_sample`) to produce 3D coordinates.
- Saves each conformer as an SDF file in `--output_dir`.
- Supports property-conditioned checkpoints via `--property gap` (or any QM9 property).
- Auto-detects latest checkpoint from `~/.torch_pharma/checkpoints/nextmol/dmt/` when no `--checkpoint` is given.

**CLI:**
```bash
# Unconditional conformer generation:
python infer_dmt.py --checkpoint path/to/dmt.pt --num_samples 100 --output_dir conformers/

# Property-conditioned (trained with train_dmt_cond.py --property gap):
python infer_dmt.py --checkpoint path/to/dmt_gap.pt --property gap --num_samples 100
```

---

### `2309_17296` — Pocket-conditioned ligand generation (arXiv:2309.17296)

**No changes.** This example already has a complete inference script:

- `generate_ligands.py` — generates ligands given a protein pocket (PDB + residue list), evaluates QED/SA/Lipinski/diversity, and optionally runs qVina2 docking.
- `run_evaluation.py` — evaluates a trained model on the QM9/GEOM dataset.

Note: both scripts use `experiments.*` imports from the original paper's codebase layout, not the `torch_pharma` package API. They are standalone and functional as provided.

---

## Architecture decisions

| Decision | Reason |
|----------|--------|
| `QM9DDPMInference` does NOT import `qm9_mol_gen_ddpm_train.py` | Importing that module would trigger `wandb.init()` and `mlflow.set_tracking_uri()` at class-definition time via the `@track_gnn_activations` decorator. |
| `strict=False` in all `load_state_dict` calls | Training checkpoints include torchmetrics state (e.g. `train_loss`, `val_loss.*`) that the inference wrappers omit. |
| `QM9InferCollater` (not `QM9Collater`) for DMT | The inference collater skips the VP noise schedule forward pass and uses random initial positions instead, matching the generation pipeline. |
| Zero context as default for DDPM | Avoids requiring the QM9 dataset at inference time. Zero = mean alpha in normalized space; override with `--load_data` for the full distribution. |
