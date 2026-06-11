"""Kabsch alignment utilities for SE(3)-invariant diffusion loss (NExT-Mol)."""

import torch
from torch_pharma.models.diffusion.utils import remove_mean_with_mask


@torch.no_grad()
def kabsch_batch(coords_pred: torch.Tensor, coords_tar: torch.Tensor) -> torch.Tensor:
    """Batch Kabsch rotation matrices. Shapes: [B, N, 3]."""
    a = torch.einsum("...ki, ...kj -> ...ij", coords_pred, coords_tar).to(torch.float32)
    u, _, vt = torch.linalg.svd(a)
    sign_deta = torch.sign(torch.det(a))
    corr_mat_diag = torch.ones((a.size(0), u.size(-1)), device=a.device)
    corr_mat_diag[:, -1] = sign_deta
    corr_mat = torch.diag_embed(corr_mat_diag)
    return torch.einsum("...ij, ...jk, ...kl -> ...il", u, corr_mat, vt)


@torch.no_grad()
def get_align_noise(
    pos_t,
    pos_0,
    pos_pred,
    alpha_t,
    sigma_t,
    batch_mask=None,
    translation_correction: bool = False,
    align_prediction: bool = False,
):
    if translation_correction:
        batch_mask = batch_mask.unsqueeze(-1)
        pos_0_centered, _ = remove_mean_with_mask(pos_0, batch_mask, return_mean=True)
        if align_prediction:
            pos_pred_centered, pos_pred_mean = remove_mean_with_mask(
                pos_pred, batch_mask, return_mean=True
            )
            rotations = kabsch_batch(pos_pred_centered, pos_0_centered)
            align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0_centered) + pos_pred_mean
        else:
            pos_t_centered, pos_t_mean = remove_mean_with_mask(pos_t, batch_mask, return_mean=True)
            rotations = kabsch_batch(pos_t_centered, pos_0_centered)
            align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0_centered) + pos_t_mean
        return (pos_t - alpha_t * align_pos_0) / sigma_t

    rotations = kabsch_batch(pos_pred if align_prediction else pos_t, pos_0)
    align_pos_0 = torch.einsum("...ki, ...ji -> ...jk", rotations, pos_0)
    return (pos_t - alpha_t * align_pos_0) / sigma_t
