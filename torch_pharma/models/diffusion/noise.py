import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Tuple, Optional

# --- NOISE SCHEDULE UTILITIES ---

def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
    raise_to_power: float = 1
) -> np.ndarray:
    """
    A cosine variance schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    """
    steps = num_timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def clip_noise_schedule(
    alphas2: np.ndarray,
    clip_value: float = 0.001
) -> np.ndarray:
    """
    Clips the noise schedule to improve stability.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)
    return alphas2

def polynomial_schedule(
    num_timesteps: int,
    s: float = 1e-4,
    power: float = 3.0
) -> np.ndarray:
    """
    A noise schedule based on a simple polynomial equation: 1 - (x ^ power).
    """
    steps = num_timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2
    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
    precision = 1 - 2 * s
    alphas2 = precision * alphas2 + s
    return alphas2

# --- NOISE COMPONENTS ---

class PositiveLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        bias: bool = True,
        weight_init_offset: int = -2
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)

class GammaNetwork(nn.Module):
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)
        self.gamma_0 = nn.Parameter(torch.tensor([-5.0]))
        self.gamma_1 = nn.Parameter(torch.tensor([10.0]))

    def gamma_tilde(self, t: torch.Tensor) -> torch.Tensor:
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma
        return gamma

class PredefinedNoiseSchedule(nn.Module):
    def __init__(
        self,
        noise_schedule: str,
        num_timesteps: int,
        noise_precision: float,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__()
        self.timesteps = num_timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(num_timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(num_timesteps, s=noise_precision, power=power)
        else:
            raise ValueError(f"Unsupported noise schedule: {noise_schedule}")

        sigmas2 = 1 - alphas2
        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)
        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = nn.Parameter(
            torch.tensor(-log_alphas2_to_sigmas2).float(),
            requires_grad=False
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]
