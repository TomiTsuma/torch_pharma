# Architecture Comparison: EVD vs NExT-Mol DMT

## torch_pharma EVD (Existing)

- **Data:** EDM QM9 PyG batches (`x`, `h`, `pos`)
- **Dynamics:** `EGNNDynamics` / `GCPNetDynamics`
- **Diffusion:** `EquivariantVariationalDiffusion` — joint coordinate + atom type denoising
- **Scheduler:** `PredefinedNoiseSchedule` / `GammaNetwork`
- **Training:** `examples/molecule_generation/2302_04313/qm9_mol_gen_ddpm_train.py`

## NExT-Mol DMT (Ported)

- **Data:** SELFIES + `rdmol2selfies` + VP noise collate
- **LLM:** HuggingFace `acharkq/MoLlama`
- **Diffusion:** `DGTDiffusion` — primarily 3D position denoising
- **Scheduler:** `NoiseScheduleVPV2` (cosine VP-SDE)
- **Training:** `NextMolDMTTask` + `Trainer.fit()`

## When to Use Which

| Use case | Recommended path |
|----------|------------------|
| Joint 2D+3D generation (EDM baseline) | EVD |
| SELFIES-valid de novo 3D | NExT-Mol |
| Conformer prediction from 2D | NExT-Mol DMT |
| Property-conditioned 3D | NExT-Mol (LLM + context) |
