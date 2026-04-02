import torch
import torch.nn as nn
from typing import Any, Iterable, Union, Tuple
import numpy as np

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

@typechecked
def inflate_batch_array(array: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_shape = (array.shape[0],) + (1,) * (len(target.shape) - 1)
    return array.view(target_shape)

@typechecked
def batch_tensor_to_list(
    data: torch.Tensor,
    batch_index: TensorType["batch_num_nodes"]
) -> Tuple[torch.Tensor, ...]:
    chunk_sizes = torch.unique(batch_index, return_counts=True)[1].tolist()
    return torch.split(data, chunk_sizes)

@typechecked
def reverse_tensor(x: torch.Tensor) -> torch.Tensor:
    return x[torch.arange(x.size(0) - 1, -1, -1)]

class Queue:
    def __init__(self, max_len: int = 50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item: Any):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self) -> Any:
        return np.mean(self.items) if self.items else 0.0

    def std(self) -> Any:
        return np.std(self.items) if self.items else 0.0

@typechecked
def get_grad_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    norm_type: float = 2.0
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    device = parameters[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        p=norm_type
    )
    return total_norm
