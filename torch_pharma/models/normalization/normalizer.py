import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from torchtyping import TensorType
from torch import Tensor
from torch_scatter import scatter_mean, scatter_add
import numpy as np

from torch. nn import functional as F
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





class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    


class LayerNorm(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, Optional[int]],
        eps: float = 1e-6,
        affine: bool = True,
        latent_dim=None,
    ):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.sdim))
            self.bias = nn.Parameter(torch.Tensor(self.sdim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, x: Dict, batch: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        s, v = x.get("s"), x.get("v")
        batch_size = int(batch.max()) + 1
        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, batch, dim=0, dim_size=batch_size)

        s = s - smean[batch]

        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, batch, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)  # .sqrt()
        sout = s / var[batch]

        if self.weight is not None and self.bias is not None:
            sout = sout * self.weight + self.bias

        if v is not None:
            vmean = torch.pow(v, 2).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, batch, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            vout = v / vmean[batch]
        else:
            vout = None

        out = sout, vout

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims}, " f"affine={self.affine})"


class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, Optional[int]],
        latent_dim: int,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.latent_dim = latent_dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight_bias = DenseLayer(latent_dim, 2 * self.sdim, bias=True)
        else:
            print(
                "Affine was set to False. This layer should used the affine transformation"
            )
            raise ValueError
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_bias.bias.data[: self.sdim] = 1
        self.weight_bias.bias.data[self.sdim :] = 0

    def forward(self, x: Dict, batch: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        s, v, z = x["s"], x["v"], x["z"]
        batch_size = int(batch.max()) + 1

        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, batch, dim=0, dim_size=batch_size)
        s = s - smean[batch]
        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, batch, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)  # .sqrt()
        sout = s / var[batch]

        weight, bias = self.weight_bias(z).chunk(2, dim=-1)
        sout = sout * weight[batch] + bias[batch]

        if v is not None:
            vmean = torch.pow(v, 2).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, batch, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            vout = v / vmean[batch]
        else:
            vout = None

        out = sout, vout

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims}, " f"affine={self.affine})"


class SE3Norm(nn.Module):
    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        """Note: There is a relatively similar layer implemented by NVIDIA:
        https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se3transformer_for_pytorch.
        It computes a ReLU on a mean-zero normalized norm, which I find surprising.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.normalized_shape = (1, 1)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(
        self,
        pos: Tensor,
        batch: Tensor,
        batch_lig: Tensor = None,
        pocket_mask: Tensor = None,
    ):
        if pocket_mask is not None:
            norm = torch.norm(pos, dim=-1, keepdim=True) * pocket_mask  # n, 1
        else:
            norm = torch.norm(pos, dim=-1, keepdim=True)
        batch_size = int(batch.max()) + 1
        if batch_lig is not None:
            n_nodes_lig = batch_lig.bincount()
            mean_norm = scatter_add(norm, batch, dim=0, dim_size=batch_size)
            mean_norm = mean_norm / n_nodes_lig.unsqueeze(1)
        else:
            mean_norm = scatter_mean(norm, batch, dim=0, dim_size=batch_size)
        new_pos = self.weight * pos / (mean_norm[batch] + self.eps)
        return new_pos

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}".format(**self.__dict__)

