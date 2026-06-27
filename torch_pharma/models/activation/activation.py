import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Any, Optional
from torch_pharma.utils.tracking.decorators import register as R
from torch_pharma.features.utils.nn_utils import stable_norm, std_conserve_scatter_sum, graph_to_batch_nx


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





@R.register('XTransEncoderAct')
class XTransEncoderAct(nn.Module):
    def __init__(self, hidden_size, ffn_size, n_rbf, cutoff=7.0, z_requires_grad=False, 
                 edge_size=16, n_layers=3, n_head=4, pre_norm=False, use_edge_feat=False, sparse_k=3, local_mask=False, attn_bias=True,
                 efficient=False, vector_act='none', 
                 # use_ieconv=False, zero_conv=False, efficient_ieconv=False, ieconv_share_edge_feat=False
        ) -> None:
        super().__init__()

        self.encoder = Transformer(
            d_hidden = hidden_size, d_ffn = ffn_size, n_heads = n_head, n_layers = n_layers,
            n_rbf = n_rbf, d_edge = edge_size, cutoff = cutoff, use_edge_feat = use_edge_feat, local_mask = local_mask, attn_bias = attn_bias,
            layer_norm = 'pre' if pre_norm else 'post', sparse_k = sparse_k, efficient = efficient,
            vector_act = vector_act, 
        )

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None):
        H, V = self.encoder(H, Z, block_id, batch_id, edges, edge_attr, topo_edges, topo_edge_attr, attn_mask)
        block_repr = std_conserve_scatter_sum(H, block_id, dim=0)
        graph_repr = std_conserve_scatter_sum(block_repr, batch_id, dim=0)
        # return H, block_repr, graph_repr, V.reshape(Z.shape) + Z
        return H, V.reshape(Z.shape) + Z

