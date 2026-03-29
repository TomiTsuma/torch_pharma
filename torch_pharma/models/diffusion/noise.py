import numpy as np
from torch import nn
import torch
import math
from torch.nn import functional as F


# --- NOISE SCHEDULE ---

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad(): self.weight.add_(-2)

    def forward(self, x):
        return F.linear(x, F.softplus(self.weight), self.bias)

class GammaNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.l3 = PositiveLinear(1, 1), PositiveLinear(1, 1024), PositiveLinear(1024, 1)
        self.gamma_0, self.gamma_1 = nn.Parameter(torch.tensor([-5.0])), nn.Parameter(torch.tensor([10.0]))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        g0, g1, gt = self.gamma_tilde(torch.zeros_like(t)), self.gamma_tilde(torch.ones_like(t)), self.gamma_tilde(t)
        return self.gamma_0 + (self.gamma_1 - self.gamma_0) * (gt - g0) / (g1 - g0)

class PredefinedNoiseSchedule(nn.Module):
    def __init__(self, noise_schedule, num_timesteps, noise_precision, **kwargs):
        super().__init__()
        self.T = num_timesteps
        if noise_schedule == "cosine":
            steps = num_timesteps + 2
            x = np.linspace(0, steps, steps)
            alphas2 = np.cos(((x / steps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas2 = alphas2 / alphas2[0]
            alphas2 = np.cumprod(np.clip(alphas2[1:] / alphas2[:-1], 0.001, 1.0))
        else: raise ValueError(f"Unsupported schedule {noise_schedule}")
        self.gamma = nn.Parameter(torch.tensor(-np.log(alphas2 / (1 - alphas2))).float(), requires_grad=False)

    def forward(self, t):
        return self.gamma[torch.round(t * self.T).long()]
