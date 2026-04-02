import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from typing import Dict, List, Optional, Tuple, Union
from torch_pharma.utils.logging import get_pylogger
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

log = get_pylogger(__name__)

class NumNodesDistribution(nn.Module):
    def __init__(
        self,
        histogram: Dict[int, int],
        verbose: bool = True,
        eps: float = 1e-30
    ):
        super().__init__()
        self.eps = eps
        num_nodes, self.keys, prob = [], {}, []
        for i, nodes in enumerate(histogram):
            num_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.register_buffer("num_nodes", torch.tensor(num_nodes))
        self.register_buffer("prob", torch.tensor(prob))
        self.prob = self.prob / torch.sum(self.prob)
        self.m = Categorical(self.prob)

        if verbose:
            entropy = torch.sum(self.prob * torch.log(self.prob + eps))
            log.info(f"Entropy of n_nodes: H[N] {entropy.item()}")

    @typechecked
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        idx = self.m.sample((n_samples,))
        return self.num_nodes[idx]

    @typechecked
    def log_prob(self, batch_n_nodes: TensorType["batch_size"]) -> TensorType["batch_size"]:
        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs, device=batch_n_nodes.device)
        log_p = torch.log(self.prob + self.eps)
        log_probs = log_p[idcs]
        return log_probs

class PropertiesDistribution:
    def __init__(
        self,
        dataloader: DataLoader,
        properties: List[str],
        device: Union[torch.device, str],
        num_bins: int = 1000,
        normalizer: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.properties = properties
        self.device = device
        self.num_bins = num_bins
        self.normalizer = normalizer
        self.distributions = {}
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(
                dataloader.dataset.data["num_atoms"].to(device),
                dataloader.dataset.data[prop].to(device),
                self.distributions[prop]
            )

    @typechecked
    def set_normalizer(self, normalizer: Dict[str, Dict[str, torch.Tensor]]):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {"probs": probs, "params": params}

    def _create_prob_given_nodes(self, values, eps=1e-12):
        n_bins = self.num_bins
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + eps
        histogram = torch.zeros(n_bins, device=self.device)
        for val in values:
            i = int((val - prop_min)/prop_range * n_bins)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = Categorical(histogram / torch.sum(histogram))
        return probs, (prop_min, prop_max)

    @typechecked
    def sample(self, num_nodes: int = 19) -> torch.Tensor:
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][num_nodes]
            idx = dist["probs"].sample((1,))
            val = self._idx2value(idx, dist["params"], len(dist["probs"].probs))
            if self.normalizer:
                val = (val - self.normalizer[prop]["mean"]) / self.normalizer[prop]["mad"]
            vals.append(val)
        return torch.cat(vals)

    @typechecked
    def sample_batch(self, num_nodes: TensorType["batch_size"]) -> torch.Tensor:
        return torch.cat([self.sample(n.item()).unsqueeze(0) for n in num_nodes], dim=0)

    def _idx2value(self, idx, params, num_bins):
        prop_range = params[1] - params[0]
        left = idx / num_bins * prop_range + params[0]
        right = (idx + 1) / num_bins * prop_range + params[0]
        return torch.rand(1, device=self.device) * (right - left) + left

class CategoricalDistribution(nn.Module):
    def __init__(self, histogram_dict: Union[Dict[int, int], torch.Tensor, np.ndarray], mapping: Dict[str, int]):
        super().__init__()
        histogram = np.zeros(len(mapping))
        if isinstance(histogram_dict, dict):
            for k, v in histogram_dict.items():
                histogram[k] = v
        else:
            if isinstance(histogram_dict, torch.Tensor):
                histogram_dict = histogram_dict.detach().cpu().numpy()
            histogram[:len(histogram_dict)] = histogram_dict
        self.p = histogram / (histogram.sum() + 1e-10)
        self.mapping = mapping

    def kl_divergence(self, other_samples: List[int]) -> float:
        sample_histogram = np.zeros(len(self.mapping))
        for x in other_samples:
            sample_histogram[x] += 1
        q = sample_histogram / (sample_histogram.sum() + 1e-10)
        return -np.sum(self.p * np.log(q / (self.p + 1e-10) + 1e-10))
