import torch
from torch import nn



class ExtendedProjector(nn.Module):
    """Extend an existing projector with a new input."""

    def __init__(self, projector, extend_dim, hidden_dim, disable_extra_gelu):
        super().__init__()
        # the following is weight tying
        self.linear1 = projector[0]
        self.act = projector[1]
        self.linear2 = projector[2]
        self.disable_extra_gelu = disable_extra_gelu
        if self.disable_extra_gelu:
            self.projector = nn.Sequential(
                nn.Linear(extend_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, self.linear1.out_features),
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(extend_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, self.linear1.out_features),
                nn.GELU(),
            )

    def forward(self, x, new_x):
        x = self.linear1(x) + self.projector(new_x)
        return self.linear2(self.act(x))

