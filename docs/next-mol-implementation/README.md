# NExT-Mol Integration Documentation

Native port of [NExT-Mol](https://arxiv.org/abs/2502.12638) (MoLlama + DMT) into **torch_pharma**.

## Status Checklist

- [x] DGTDiffusion backbone (`torch_pharma/models/diffusion/dmt.py`)
- [x] VP-SDE scheduler (`torch_pharma/models/diffusion/vp_scheduler.py`)
- [x] Unified Trainer + Task protocol
- [x] SELFIES collaters + mol mapping
- [x] MoLlama HF integration
- [x] NextMolDMTTask / NextMolLLMTask
- [x] Evaluation metrics (conformer + 2D)
- [x] Checkpoint migration utilities
- [ ] Full QM9/Geom-DRUGS preprocessing pipeline (use NExT-Mol caches)
- [ ] Benchmark parity validation on published numbers

## Document Index

| Doc | Topic |
|-----|-------|
| [01-executive-summary.md](01-executive-summary.md) | What and why |
| [02-architecture-comparison.md](02-architecture-comparison.md) | EVD vs DMT |
| [03-nextr-mol-source-inventory.md](03-nextr-mol-source-inventory.md) | NExT-Mol module map |
| [04-current-torch-pharma-state.md](04-current-torch-pharma-state.md) | Port status |
| [05-target-module-layout.md](05-target-module-layout.md) | Package tree |
| [06-unified-trainer-design.md](06-unified-trainer-design.md) | Trainer + BaseTask |
| [07-data-pipeline.md](07-data-pipeline.md) | SELFIES, collaters, datasets |
| [08-model-components.md](08-model-components.md) | DMT, LLM, sampling |
| [09-training-workflows.md](09-training-workflows.md) | Three workflows |
| [10-evaluation-parity.md](10-evaluation-parity.md) | Metrics |
| [11-dependency-and-config.md](11-dependency-and-config.md) | Dependencies |
| [12-implementation-roadmap.md](12-implementation-roadmap.md) | Phases |
| [13-checkpoint-migration.md](13-checkpoint-migration.md) | Weight loading |
| [14-examples-and-runbook.md](14-examples-and-runbook.md) | **Training, eval, and run examples** |

## Storage Layout (`TORCH_PHARMA_HOME`)

All NExT-Mol data and checkpoints live under `~/.torch_pharma/` (see `torch_pharma/paths.py`):

```
~/.torch_pharma/
├── data/nextmol/
│   ├── qm9_lm/          # MoLlama SELFIES caches
│   ├── tordf_qm9/       # DMT conformer caches
│   └── geom_drugs/      # Geom-DRUGS caches
├── checkpoints/nextmol/
│   ├── mollama/         # Stage-1 checkpoints
│   └── dmt/             # Stage-2/3 checkpoints
└── pretrained/nextmol/  # OSF / migrated NExT-Mol weights
```

## Quick Start

```bash
# 1. Install package + NExT-Mol extras
pip install -e ".[nextmol]"

# 2. Install OSF preprocessed data into ~/.torch_pharma/
py -3.11 scripts/install_nextmol_data.py

# 3. Smoke-test imports
pytest tests/test_nextmol_imports.py -q

# 4. Train (stage 1 → stage 2)
python examples/molecule_generation/next_mol/train_mollama.py --max_epochs 1 --batch_size 4
python examples/molecule_generation/next_mol/train_dmt_uncond.py --max_epochs 2 --batch_size 4

# 5. Property-conditioned DMT (optional)
python examples/molecule_generation/next_mol/train_dmt_cond.py --property gap --max_epochs 2 --batch_size 4
```

Paths default to `TORCH_PHARMA_HOME` (`~/.torch_pharma`). Override with `--data_root` and `--checkpoint_dir` when needed.

For full CLI flags, evaluation snippets, checkpoint loading, and troubleshooting, see **[14-examples-and-runbook.md](14-examples-and-runbook.md)**.
