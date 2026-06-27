from torch import nn
class SubLayerWrapper(nn.Module):

    def __init__(self, sub_layer, d_hidden, layer_norm = 'pre', residual = True):

        super(SubLayerWrapper, self).__init__()
        self.sub_layer = sub_layer
        self.d_hidden = d_hidden
        self.layer_norm = layer_norm
        self.ln = nn.LayerNorm(d_hidden)
        self.residual = residual

    def forward(self, H, V, **kwargs):
        H0, V0 = H, V
        if self.layer_norm == 'pre':
            H = self.ln(H0)
        H, V = self.sub_layer(H, V, **kwargs)
        if self.residual:
            H = H + H0
            V = V + V0
        if self.layer_norm == 'post':
            H = self.ln(H)
        return H, V

