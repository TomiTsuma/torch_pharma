"""Load NExT-Mol Lightning checkpoints into native torch_pharma modules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from torch_pharma.paths import NEXTMOL_PRETRAINED, ensure_nextmol_dirs
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


def remap_state_dict(state_dict: Dict[str, object], prefix: str = "") -> Dict[str, object]:
    """Strip LightningModule prefixes from NExT-Mol checkpoints."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for strip in (
            "diffusion_model.",
            "llm_model.",
            "llm_model.base_model.model.",
            "llm_projector.",
            "model.",
        ):
            if new_key.startswith(strip):
                new_key = new_key[len(strip) :]
        if prefix:
            new_key = f"{prefix}{new_key}"
        remapped[new_key] = value
    return remapped


def default_pretrained_dir() -> Path:
    ensure_nextmol_dirs()
    return NEXTMOL_PRETRAINED


def load_nextmol_dmt_checkpoint(
    module,
    checkpoint_path: Union[str, Path],
    strict: bool = False,
):
    """Load NExT-Mol DiffussionPL checkpoint into NextMolDMTModule."""
    import torch

    log.info("Migrating NExT-Mol DMT checkpoint from %s (strict=%s)", checkpoint_path, strict)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    diffusion_sd = {k: v for k, v in state.items() if k.startswith("diffusion_model.")}
    llm_sd = {k: v for k, v in state.items() if "llm_model" in k}
    projector_sd = {k: v for k, v in state.items() if k.startswith("llm_projector.")}

    if diffusion_sd:
        module.diffusion_model.load_state_dict(
            remap_state_dict(diffusion_sd, ""), strict=strict
        )
    if llm_sd and hasattr(module, "llm_model"):
        module.llm_model.load_state_dict(remap_state_dict(llm_sd, ""), strict=False)
    if projector_sd and hasattr(module, "llm_projector"):
        module.llm_projector.load_state_dict(
            remap_state_dict(projector_sd, ""), strict=strict
        )
    return module
