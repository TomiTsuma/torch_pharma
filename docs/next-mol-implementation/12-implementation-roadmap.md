# Implementation Roadmap

## Developer Runbook

Operational docs for running training and evaluation: [14-examples-and-runbook.md](14-examples-and-runbook.md).

## Phase 0 — Unblock DMT Port ✅

- Fix circular imports
- Add `DGTDiffusionConfig`
- Import smoke tests

## Phase 1 — Unified Trainer ✅

- `Trainer`, `Callback`, `ModelCheckpoint`, `EarlyStopping`
- `BaseTask` protocol
- `EVDMoleculeGenerationTask` proof-of-concept

## Phase 2 — Data Pipeline ✅

- `NoiseScheduleVPV2`, collaters, mol_mapping
- `QM9LMDataset`, `QM9TorDFDataset` loaders
- `nextmol_dm.py` factory

## Phase 3 — DMT Training ✅

- `LLMProjector`, Kabsch loss, VP-SDE sampler
- `NextMolDMTTask`, `DeltaTrainingCallback`
- Checkpoint migration utilities

## Phase 4 — LLM Stage ✅

- `NextMolLLMTask`, `load_mollama`
- `train_mollama.py` example

## Phase 5 — Evaluation + Geom-DRUGS ✅

- Conformer + 2D metrics
- `GeomDrugsTorDFDataset` stub
- Benchmark documentation

## Next Steps

1. Run full preprocessing pipeline or document OSF cache layout
2. Validate checkpoint parity against NExT-Mol DMT-B weights
3. Add DDP to Trainer for multi-GPU training
4. Port full JODO evaluation for publication metrics
