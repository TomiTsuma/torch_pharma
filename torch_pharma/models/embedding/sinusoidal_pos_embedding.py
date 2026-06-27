import torch
from torch import nn
import math
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum


class LearnedSinusodialposEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
    


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos Positional Embedding
    """
    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        position_ids = position_ids[None] # [1, N]
        indices = torch.arange(self.output_dim // 2, device=device, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(-1, self.output_dim)
        return embeddings


class SinusoidalTimeEmbeddings(nn.Module):
    '''
        sin(1*t*2pi), sin(2*t*2pi), ...,
        cos(1*t*2pi), cos(2*t*2pi)
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        if self.dim == 1: # no projection
            return time.reshape(-1,1)
        device = time.device
        half_dim = self.dim // 2
        freq = 2 * torch.arange(half_dim, device=device) * math.pi
        t = freq * time[..., None]
        embeddings = torch.cat((t.sin(), t.cos()), dim=-1)
        return embeddings