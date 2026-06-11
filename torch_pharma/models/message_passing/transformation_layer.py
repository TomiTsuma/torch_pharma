import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math
from typing import Tuple, Optional
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter
from torch_pharma.models.diffusion.utils import uncompilable_dropout
from torch_pharma.utils.device import is_amd_gpu

disable_compile = is_amd_gpu()


class TransformLayer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransformLayer, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = nn.Linear(edge_dim, heads * out_channels, bias=False)

        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels

        x_feat = x
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    @torch.compile(dynamic=True, disable=disable_compile)
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.out_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = uncompilable_dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * torch.tanh(self.lin_edge1(edge_attr).view(-1, self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class TransformLayerOptim(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransformLayerOptim, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_qkv = nn.Linear(in_channels, heads * out_channels * 3, bias=bias)

        self.lin_edge = nn.Linear(edge_dim, heads * out_channels * 2, bias=False)

        self.proj = nn.Linear(heads * out_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_qkv.reset_parameters()
        self.lin_edge.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels
        x_feat = x
        qkv = self.lin_qkv(x_feat).view(-1, H, 3, C)
        query, key, value = qkv.unbind(dim=2)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        out_x = self.proj(out_x)
        return out_x

    @torch.compile(dynamic=True, disable=disable_compile)
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_key, edge_value = torch.tanh(self.lin_edge(edge_attr)).view(-1, self.heads, 2, self.out_channels).unbind(dim=2)

        alpha = (query_i * key_j * edge_key).sum(dim=-1) / math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = uncompilable_dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j * edge_value * alpha.view(-1, self.heads, 1)
        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
