import torch
from torch import nn
from torch_scatter import scatter

# --- DYNAMICS COMPONENTS ---

class EGNN_Sparse(nn.Module):
    def __init__(self, feats_dim, pos_dim=3, m_dim=16, edge_attr_dim=0, coors_tanh=True):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(feats_dim * 2 + edge_attr_dim + 1, m_dim), nn.SiLU(), nn.Linear(m_dim, m_dim), nn.SiLU())
        self.coors_mlp = nn.Sequential(nn.Linear(m_dim, m_dim), nn.SiLU(), nn.Linear(m_dim, 1, bias=False), nn.Tanh() if coors_tanh else nn.Identity())
        self.node_mlp = nn.Sequential(nn.Linear(feats_dim + m_dim, feats_dim), nn.SiLU(), nn.Linear(feats_dim, feats_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        row, col = edge_index
        pos, h = x[:, :3], x[:, 3:]
        dist = torch.sum((pos[row] - pos[col])**2, dim=-1, keepdim=True)
        edge_feats = torch.cat([h[row], h[col], edge_attr, dist], dim=-1)
        m_ij = self.edge_mlp(edge_feats)
        coors_out = pos + scatter(m_ij * (pos[row] - pos[col]), row, dim=0, reduce="sum") / 10.0 # Scaling
        h_out = h + self.node_mlp(torch.cat([h, scatter(m_ij, row, dim=0, reduce="sum")], dim=-1))
        return torch.cat([coors_out, h_out], dim=-1)