from torch import nn
# swish activation fallback

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


SiLU = nn.SiLU if hasattr(nn, "SiLU") else Swish_