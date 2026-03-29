import torch
from torch import nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard

@typechecked
def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    keepdim: bool = False,
    sqrt: bool = True
):
    norm = torch.sum(x ** 2, dim=dim, keepdim=keepdim)
    if sqrt:
        norm = torch.sqrt(norm + eps)
    return norm + eps


@typechecked
def norm_no_nan(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
    sqrt: bool = True
):
    """
    From https://github.com/drorlab/gvp-pytorch

    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


@typechecked
def is_identity(nonlinearity: Optional[Union[Callable, nn.Module]] = None):
    return nonlinearity is None or isinstance(nonlinearity, nn.Identity)


@typechecked
def inflate_batch_array(array: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Inflate the batch array (`array`) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e., shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.shape[0],) + (1,) * (len(target.shape) - 1)
    return array.view(target_shape)


@typechecked
def get_grad_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    norm_type: float = 2.0
) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters_list = list(parameters)
    if len(parameters_list) == 0:
        return torch.tensor(0.0)
    device = parameters_list[0].device

    parameters = [p for p in parameters_list if p.grad is not None]
    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.0, device=device)

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type) for p in parameters]
        ),
        p=norm_type
    )
    return total_norm


@typechecked
def batch_tensor_to_list(
    data: torch.Tensor,
    batch_index: TensorType["batch_num_nodes"]
) -> Tuple[torch.Tensor, ...]:
    # note: assumes that `batch_index` is sorted in non-decreasing order
    chunk_sizes = torch.unique(batch_index, return_counts=True)[1].tolist()
    return torch.split(data, chunk_sizes)


@typechecked
def reverse_tensor(x: torch.Tensor) -> torch.Tensor:
    return x[torch.arange(x.size(0) - 1, -1, -1)]
