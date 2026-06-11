"""VP-SDE reverse sampling for NExT-Mol DMT."""

from __future__ import annotations

import torch

from torch_pharma.models.diffusion.utils import remove_mean
from torch_pharma.models.diffusion.vp_scheduler import NoiseScheduleVPV2, sample_com_rand_pos


@torch.no_grad()
def reverse_vp_sde_sample(
    diffusion_model,
    data_batch,
    noise_scheduler: NoiseScheduleVPV2,
    lm_x=None,
    context=None,
    sampling_steps: int = 100,
    t_cond: str = "t",
    disable_com: bool = True,
    pos_std: float = 1.0,
):
    """Reverse VP-SDE sampler ported from NExT-Mol DiffussionPL.sample()."""
    num_nodes = data_batch.x.shape[0]
    device = data_batch.x.device
    bs = len(data_batch.smiles)
    t_array = torch.linspace(noise_scheduler.T, 0.001, sampling_steps, device=device)
    s_array = torch.cat([t_array[1:], torch.zeros(1, device=device)])

    for i in range(len(t_array)):
        t = t_array[i]
        s = s_array[i]
        alpha_t, sigma_t = noise_scheduler.marginal_prob(t)
        alpha_s, sigma_s = noise_scheduler.marginal_prob(s)

        alpha_t_given_s = alpha_t / alpha_s
        sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
        sigma = sigma_t_given_s * sigma_s / sigma_t
        noise_level = torch.log(alpha_t ** 2 / sigma_t ** 2)

        if t_cond == "t":
            data_batch.t_cond = torch.ones(num_nodes, device=device) * t
        elif t_cond == "noise_level":
            data_batch.t_cond = torch.ones(num_nodes, device=device) * noise_level
        else:
            raise ValueError(f"Unknown t_cond: {t_cond}")

        data_batch.alpha_t = torch.ones((num_nodes, 1), device=device) * alpha_t
        data_batch.sigma_t = torch.ones((num_nodes, 1), device=device) * sigma_t

        if context is not None:
            pred_pos, _ = diffusion_model(data_batch, lm_x, context)
        else:
            pred_pos, _ = diffusion_model(data_batch, lm_x)

        pos_mean = (
            (alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2) * data_batch.pos
            + (alpha_s * sigma2_t_given_s / sigma_t ** 2) * pred_pos
        )
        pos_mean = remove_mean(pos_mean, data_batch.batch)

        if disable_com:
            epsilon_pos = torch.randn(data_batch.pos.shape, device=device)
        else:
            epsilon_pos = sample_com_rand_pos(data_batch.pos.shape, data_batch.batch)

        data_batch.pos = pos_mean + sigma * epsilon_pos

    pos = pos_mean * pos_std
    data_batch.pos = pos
    return data_batch, pos
