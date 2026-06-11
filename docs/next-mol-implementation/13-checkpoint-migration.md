# Checkpoint Migration

## NExT-Mol Checkpoint Format

Lightning checkpoints store weights under prefixed keys:

```
diffusion_model.block_0.attn_mpnn.lin_key.weight
llm_model.base_model.model.model.layers.0...
llm_projector.linear_proj.weight
```

## Storage Location

Place downloaded OSF / NExT-Mol Lightning checkpoints under:

```
$TORCH_PHARMA_HOME/pretrained/nextmol/
```

(`torch_pharma.paths.NEXTMOL_PRETRAINED` — created by `ensure_nextmol_dirs()`.)

Training checkpoints are written to:

- `$TORCH_PHARMA_HOME/checkpoints/nextmol/mollama/`
- `$TORCH_PHARMA_HOME/checkpoints/nextmol/dmt/`

## Loading into torch_pharma

```python
from torch_pharma.paths import NEXTMOL_PRETRAINED
from torch_pharma.models.diffusion.checkpoint_migration import load_nextmol_dmt_checkpoint
from torch_pharma.tasks.molecule_generation import NextMolDMTTask

task = NextMolDMTTask()
task.setup(torch.device("cpu"))
ckpt = NEXTMOL_PRETRAINED / "dmt_b_epoch99.ckpt"
load_nextmol_dmt_checkpoint(task.model, ckpt, strict=False)
```

## Key Remapping Rules

`remap_state_dict()` strips these prefixes:

1. `diffusion_model.`
2. `llm_model.` / `llm_model.base_model.model.`
3. `llm_projector.`
4. `model.`

## HuggingFace MoLlama

Load directly via `acharkq/MoLlama` — no remapping needed for base LLM weights when using `load_mollama()`.

## OSF DMT Checkpoints

Download from NExT-Mol README links. Use `strict=False` on first load to identify missing/unexpected keys.

## Validation

Compare `DGTDiffusion` forward outputs on a fixed batch before and after migration (atol=1e-4 for fp32).
