import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch_pharma.features import ScalarVector
from typing import Union

class VectorDropout(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = x[0].device
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class GCPDropout(nn.Module):
    """
    From https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float, use_gcp_dropout: bool = True):
        super().__init__()
        self.scalar_dropout = nn.Dropout(drop_rate) if use_gcp_dropout else nn.Identity()
        self.vector_dropout = VectorDropout(drop_rate) if use_gcp_dropout else nn.Identity()

    def forward(self, x: Union[torch.Tensor, ScalarVector]):
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (x.scalar.shape[0] == 0 or x.vector.shape[0] == 0):
            return x
        elif isinstance(x, torch.Tensor):
            return self.scalar_dropout(x)
        return ScalarVector(self.scalar_dropout(x[0]), self.vector_dropout(x[1]))
