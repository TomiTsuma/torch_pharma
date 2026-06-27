import torch
from torch import nn
from torch.nn import functional as F

class GVPFFNLayer(nn.Module):

    def __init__(self, d_hidden, d_ffn, act_fn=nn.SiLU(), d_output=None, vector_act='none'):

        super(GVPFFNLayer, self).__init__()

        self.d_hidden = d_hidden
        self.d_ffn = d_ffn
        self.act_fn = act_fn
        self.d_output = d_hidden if d_output is None else d_output

        self.linear_v = nn.Linear(d_hidden, d_hidden + self.d_output, bias=False)
        self.ffn_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_ffn),
            act_fn,
            nn.Linear(d_ffn, d_hidden + self.d_output)
        )

        self.vector_act = vector_act
        if self.vector_act == 'layernorm':
            self.vector_layernorm = nn.LayerNorm(self.d_output)

    def vector_act_func(self, Vs):
        if self.vector_act == 'none':
            return Vs
        elif self.vector_act == 'sigmoid':
            return F.sigmoid(Vs)
        elif self.vector_act == 'tanh':
            return F.tanh(Vs)
        elif self.vector_act == 'layernorm':
            return self.vector_layernorm(Vs)
        elif self.vector_act == 'one':
            return torch.ones_like(Vs)

    def forward(self, H, V):

        V_proj = self.linear_v(V)
        V1, V2 = V_proj[...,:self.d_hidden], V_proj[...,self.d_hidden:]
        scaler = torch.cat([H, V1.norm(dim=-2)], dim=-1)
        scaler_out = self.ffn_mlp(scaler)
        H_out, V_update = scaler_out[...,:self.d_hidden], scaler_out[...,self.d_hidden:]
        V_out = self.vector_act_func(V_update).unsqueeze(-2) * V2
        return H_out, V_out    
    
