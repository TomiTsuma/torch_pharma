# Dependencies and Configuration

## Optional Extra: `[nextmol]`

```bash
pip install -e ".[nextmol]"
```

Packages added:

- `selfies` — molecular string encoding
- `transformers>=4.36` — MoLlama HF loading
- `peft` — LoRA fine-tuning
- `accelerate` — HF model loading
- `scipy` — rotation augmentation in collaters
- `nltk`, `rouge-score` — LM evaluation (optional)

Eval extras (manual): `fcd-torch`, `moses` (git install)

## Not Required

- `lightning` — replaced by unified Trainer
- `deepspeed` — DDP planned for Trainer v2

## Path Configuration

All paths resolve under `TORCH_PHARMA_HOME` (`~/.torch_pharma` by default):

```python
from torch_pharma.paths import (
    TORCH_PHARMA_HOME,
    NEXTMOL_QM9_LM,
    NEXTMOL_QM9_TORDF,
    NEXTMOL_MOLLAMA_CHECKPOINTS,
    NEXTMOL_DMT_CHECKPOINTS,
    NEXTMOL_PRETRAINED,
    resolve_data_root,
    resolve_checkpoint_dir,
)
```

`Trainer` defaults to `TORCH_PHARMA_HOME/checkpoints/` for checkpoint storage.

## Config Classes

| Class | Location |
|-------|----------|
| `DGTDiffusionConfig` | `models/diffusion/config.py` |
| `NextMolTrainingConfig` | `models/diffusion/config.py` |

## Dataset Config

`get_dataset_info("QM9-df")` returns `pos_std`, `max_atoms`, feature dimensions.
