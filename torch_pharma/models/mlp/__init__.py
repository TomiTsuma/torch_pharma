from torch import nn
from torch_scatter import scatter_sum
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, act_fn=nn.SiLU(), end_with_act=False, dropout=0.0):
        super().__init__()
        assert n_layers >= 2, f'MLP should have at least two layers (input/output)'
        self.input_linear = nn.Linear(input_size, hidden_size)
        medium_layers = [act_fn]
        for i in range(n_layers):
            medium_layers.append(nn.Linear(hidden_size, hidden_size))
            medium_layers.append(act_fn)
            medium_layers.append(nn.Dropout(dropout))
        self.medium_layers = nn.Sequential(*medium_layers)
        if end_with_act:
            self.output_linear = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                act_fn
            )
        else:
            self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, H):
        '''
        Args:
            H: [..., input_size]
        Returns:
            H: [..., output_size]
        '''
        H = self.input_linear(H)
        H = self.medium_layers(H)
        H = self.output_linear(H)
        return H
    
