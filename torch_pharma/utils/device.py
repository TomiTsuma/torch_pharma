"""Device and compilation helpers."""

import torch


def is_amd_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return "AMD" in torch.cuda.get_device_name(0)
    except Exception:
        return False


def get_half_precision_dtype():
    if not torch.cuda.is_available():
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
