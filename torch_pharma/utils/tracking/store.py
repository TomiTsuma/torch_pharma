"""
Role: Data persistence and statistical calculations.
Contents: Only the ActivationStore class. 
        This file handles the raw state mechanism (layers, nodes, edges, stats) 
        and statistics logic (_compute_stats, summary).
"""

import functools
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
 
import torch
import torch.nn as nn
 
class ActivationStore:
    """
    Holds all captured activation tensors for a single forward pass.
 
    Structure
    ---------
    layers  : { layer_name -> tensor }
    nodes   : { layer_name -> tensor  (shape: [num_nodes, hidden_dim]) }
    edges   : { layer_name -> tensor  (shape: [num_edges, hidden_dim]) }
    stats   : { layer_name -> { mean, std, min, max, norm, sparsity } }
    """
    def __init__(self):
        self.layers: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.nodes:  Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.edges:  Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.stats:  Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self._hooks: List[Any] = []          # torch hook handles

    def register(self, name: str, tensor: torch.Tensor, kind: str = "layer"):
        """
        Store a tensor under *name* in the appropriate bucket.

        Parameters
        ----------
        name   : unique identifier (e.g. 'conv1', 'layer_0.edge_attr')
        tensor : the activation tensor (detached copy is stored)
        kind   : one of 'layer' | 'node' | 'edge'
        """
        saved = tensor.detach().cpu()
        if kind == "node":
            self.nodes[name].append(saved)
        elif kind == "edge":
            self.edges[name].append(saved)
        else:
            self.layers[name].append(saved)

        self.stats[name].append(self._compute_stats(saved)) 

    @staticmethod
    def _compute_stats(t: torch.Tensor) -> Dict[str, float]:
        flat = t.float().flatten()
        return {
            "mean":     flat.mean().item(),
            "std":      flat.std().item(),
            "min":      flat.min().item(),
            "max":      flat.max().item(),
            "norm":     flat.norm().item(),
            "sparsity": (flat == 0).float().mean().item(),
            "shape":    tuple(t.shape),
        }

    def clear(self):
        self.layers.clear()
        self.nodes.clear()
        self.edges.clear()
        self.stats.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self) -> str:
        lines = ["=" * 60, "Activation Sequence Summary", "=" * 60]

        for bucket_name, bucket in [
            ("LAYERS", self.layers),
            ("NODES",  self.nodes),
            ("EDGES",  self.edges),
        ]:
            if not bucket:
                continue
            lines.append(f"\n[{bucket_name}]")
            for name in bucket:
                s_list = self.stats[name]
                if not s_list: continue
                s_last = s_list[-1]
                num_steps = len(s_list)
                lines.append(
                    f"  {name:<30s}  steps={num_steps:<5d} shape={str(s_last['shape']):<20s}"
                    f"  mean(last)={s_last['mean']:+.4f}  std(last)={s_last['std']:.4f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"ActivationStore("
            f"layers={len(self.layers)} (sequences), "
            f"nodes={len(self.nodes)} (sequences), "
            f"edges={len(self.edges)} (sequences))"
        )

    