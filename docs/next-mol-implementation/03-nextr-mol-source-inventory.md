# NExT-Mol Source Inventory

## Entry Scripts

| Script | Purpose |
|--------|---------|
| `llm_train.py` | MoLlama SELFIES training |
| `train_lm_conf.py` | DMT conformer / conditional training |
| `train_uncond_gene.py` | De novo 3D generation |

## Core Models

| File | Class | Role |
|------|-------|------|
| `model/diffusion_model_dgt.py` | `DGTDiffusion` | 3D denoiser |
| `model/diffusion_pl.py` | `DiffussionPL` | Training + sampling |
| `model/llm_pl.py` | `LLMPL` | LM training |
| `model/modeling_llama.py` | `LlamaForCausalLM` | Custom Llama (use HF in torch_pharma) |

## Data

| File | Role |
|------|------|
| `data_provider/diffusion_data_module.py` | `QM9Collater`, noise injection |
| `data_provider/diffusion_scheduler.py` | `NoiseScheduleVPV2` |
| `data_provider/mol_mapping_utils.py` | Atom-token alignment |
| `data_provider/qm9_dataset_v6.py` | QM9 SELFIES LM dataset |
| `data_provider/qm9_dataset_tordf.py` | Conformer dataset |

## Out of Scope

- `model/equiformer/` — not used by primary DMT path
- `model/unimol.py` — superseded by DMT+LLM
- `model/torsional_diffusion/` — separate research direction
