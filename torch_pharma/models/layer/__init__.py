import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch_pharma.features import ScalarVector
from typing import Union

class GCPLayerNorm(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, dims: ScalarVector, eps: float = 1e-8, use_gcp_norm: bool = True):
        super().__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = nn.LayerNorm(self.scalar_dims) if use_gcp_norm else nn.Identity()
        self.use_gcp_norm = use_gcp_norm
        self.eps = eps

    @staticmethod
    def norm_vector(v: torch.Tensor, use_gcp_norm: bool = True, eps: float = 1e-8) -> torch.Tensor:
        v_norm = v
        if use_gcp_norm:
            vector_norm = torch.clamp(torch.sum(torch.square(v), dim=-1, keepdim=True), min=eps)
            vector_norm = torch.sqrt(torch.mean(vector_norm, dim=-2, keepdim=True))
            v_norm = v / vector_norm
        return v_norm

    def forward(self, x: Union[torch.Tensor, ScalarVector]):
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (x.scalar.shape[0] == 0 or x.vector.shape[0] == 0):
            return x
        elif not self.vector_dims:
            return self.scalar_norm(x)
        s, v = x
        return ScalarVector(self.scalar_norm(s), self.norm_vector(v, use_gcp_norm=self.use_gcp_norm, eps=self.eps))

