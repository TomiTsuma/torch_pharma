import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Any, Optional

class Swish_(nn.Module):
    """
    Swish activation function fallback for older PyTorch versions.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

# Use nn.SiLU if available, otherwise fallback to Swish_
SiLU = nn.SiLU if hasattr(nn, "SiLU") else Swish_

def get_nonlinearity(
    nonlinearity: Optional[str] = None,
    slope: float = 1e-2,
    return_functional: bool = False
) -> Any:
    """
    Utility to get a nonlinearity/activation function by name.
    """
    if nonlinearity is None:
        return nn.Identity()
        
    nonlinearity = nonlinearity.lower().strip()
    
    if nonlinearity == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif nonlinearity == "leakyrelu":
        return (
            partial(F.leaky_relu, negative_slope=slope)
            if return_functional
            else nn.LeakyReLU(negative_slope=slope)
        )
    elif nonlinearity == "selu":
        return (
            partial(F.selu)
            if return_functional
            else nn.SELU()
        )
    elif nonlinearity == "silu" or nonlinearity == "swish":
        if return_functional:
            return F.silu if hasattr(F, "silu") else lambda x: x * torch.sigmoid(x)
        else:
            return SiLU()
    elif nonlinearity == "sigmoid":
        return torch.sigmoid if return_functional else nn.Sigmoid()
    elif nonlinearity == "tanh":
        return torch.tanh if return_functional else nn.Tanh()
    else:
        raise NotImplementedError(f"The nonlinearity {nonlinearity} is currently not implemented.")
