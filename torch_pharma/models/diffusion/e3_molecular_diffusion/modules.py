import math
from typing import Callable, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn.inits import reset


class GatedEquivBlock(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, Optional[int]],
        hs_dim: Optional[int] = None,
        hv_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
        use_mlp: bool = False,
    ):
        super(GatedEquivBlock, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vo = 0 if self.vo is None else self.vo

        self.hs_dim = hs_dim or max(self.si, self.so)
        self.hv_dim = hv_dim or max(self.vi, self.vo)
        self.norm_eps = norm_eps

        self.use_mlp = use_mlp

        self.Wv0 = DenseLayer(self.vi, self.hv_dim + self.vo, bias=False)

        if not use_mlp:
            self.Ws = DenseLayer(self.hv_dim + self.si, self.vo + self.so, bias=True)
        else:
            self.Ws = nn.Sequential(
                DenseLayer(
                    self.hv_dim + self.si, self.si, bias=True, activation=nn.SiLU()
                ),
                DenseLayer(self.si, self.vo + self.so, bias=True),
            )
            if self.vo > 0:
                self.Wv1 = DenseLayer(self.vo, self.vo, bias=False)
            else:
                self.Wv1 = None

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.Ws)
        reset(self.Wv0)
        if self.use_mlp:
            if self.vo > 0:
                reset(self.Wv1)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        s, v = x
        vv = self.Wv0(v)

        if self.vo > 0:
            vdot, v = vv.split([self.hv_dim, self.vo], dim=-1)
        else:
            vdot = vv

        vdot = torch.clamp(torch.pow(vdot, 2).sum(dim=1), min=self.norm_eps)  # .sqrt()

        s = torch.cat([s, vdot], dim=-1)
        s = self.Ws(s)
        if self.vo > 0:
            gate, s = s.split([self.vo, self.so], dim=-1)
            v = gate.unsqueeze(1) * v
            if self.use_mlp:
                v = self.Wv1(v)

        return s, v


act_class_mapping = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1_buffer = self.vec1_proj(v)

        # detach zero-entries to avoid NaN gradients during force loss backpropagation
        vec1 = torch.zeros(
            vec1_buffer.size(0), vec1_buffer.size(2), device=vec1_buffer.device
        )
        mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).any(dim=1)
        vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2)

        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v
