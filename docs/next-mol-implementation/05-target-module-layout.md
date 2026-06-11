# Target Module Layout

```
torch_pharma/
├── paths.py                       # TORCH_PHARMA_HOME data + checkpoint paths
├── models/
│   ├── diffusion/
│   │   ├── dmt.py                 # DGTDiffusion
│   │   ├── config.py              # DGTDiffusionConfig
│   │   ├── vp_scheduler.py        # NoiseScheduleVPV2
│   │   ├── sde_sampler.py         # Reverse VP-SDE
│   │   └── checkpoint_migration.py
│   ├── llm/
│   │   ├── mollama.py             # HF MoLlama loader
│   │   └── projector.py           # LLMProjector
│   └── dynamics/ + message_passing/
├── data/components/nextmol/
│   ├── collators.py
│   ├── mol_mapping.py
│   ├── dataset_config.py
│   └── datasets/
├── tasks/molecule_generation/
│   ├── nextmol_dmt_task.py
│   ├── nextmol_llm_task.py
│   └── evd_task.py
├── training/
│   ├── trainer.py
│   ├── callbacks.py
│   └── checkpoint.py
└── evaluation/nextmol/
```

Design principle: `data/components/nextmol/` is the isolation boundary for all SELFIES-specific logic.
