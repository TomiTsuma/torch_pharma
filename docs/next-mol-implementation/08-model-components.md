# Model Components

## DGTDiffusion

Graph relational transformer denoiser. Key hyperparameters (DMT-B):

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 384 |
| `n_blocks` | 6 |
| `in_node_features` | 44 (QM9) / 74 (Geom-DRUGS) |

Config: `DGTDiffusionConfig` in `torch_pharma/models/diffusion/config.py`

## LLMProjector

Maps LLM hidden states to per-atom embeddings:

```
lm_x = rdmol2selfies @ lm_embeds  # [B, N_atoms, hidden]
```

## MoLlama

Loaded via HuggingFace `acharkq/MoLlama` with optional PEFT LoRA (`torch_pharma/models/llm/mollama.py`).

## VP-SDE Sampler

`reverse_vp_sde_sample()` in `sde_sampler.py` implements the reverse diffusion loop from NExT-Mol `DiffussionPL.sample()`.

## Kabsch Loss

SE(3)-invariant noise prediction loss via `torch_pharma/features/kabsch.py`.
