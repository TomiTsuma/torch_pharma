import torch
from torch import nn
from torch_pharma.models.gnn.egnn import CoorsNorm
from torch_scatter import scatter
from torch_pharma.models.diffusion.utils import modulate

class ConditionalEquivariantUpdate(nn.Module):
    """Update atom coordinates equivariantly, use time emb condition."""

    def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim, residual=True):
        super().__init__()
        self.coord_norm = CoorsNorm(scale_init=1e-2)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 2)
        )
        input_ch = hidden_dim * 2 + edge_dim + dist_dim
        self.input_lin = nn.Linear(input_ch, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.residual = residual

    def forward(self, h, pos, edge_index, edge_attr, dist, time_emb):
        row, col = edge_index
        h_input = torch.cat([h[row], h[col], edge_attr, dist], dim=1)
        coord_diff = pos[row] - pos[col]
        coord_diff = self.coord_norm(coord_diff)

        shift, scale = self.time_mlp(time_emb).chunk(2, dim=1)
        inv = modulate(self.ln(self.input_lin(h_input)), shift, scale)
        inv = torch.tanh(self.coord_mlp(inv))
        trans = coord_diff * inv
        agg = scatter(trans, edge_index[0], 0, reduce='add', dim_size=pos.size(0))
        if self.residual:
            return pos + agg
        else:
            return agg
