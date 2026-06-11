import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter

disable_compile = torch.cuda.get_device_name(0).find('AMD') >= 0

@torch.compiler.disable
def uncompilable_dropout(x, p, training):
    return F.dropout(x, p=p, training=training, )


def coord2dist(x, edge_index):
    # coordinates to distance
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    return radial

def remove_mean_with_mask(x, node_mask, return_mean=False):
    # masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    # assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    if return_mean:
        return x, mean
    return x


def remove_mean(pos, batch):
    mean_pos = scatter(pos, batch, dim=0, reduce='mean') # shape = [B, 3]
    pos = pos - mean_pos[batch]
    return pos


def modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift