"""Batch container for NExT-Mol training."""

from dataclasses import dataclass
from typing import Any, Optional

from torch_geometric.data import Batch


@dataclass
class NextMolBatch:
    """PyG graph batch + SELFIES tokenizer output."""

    data_batch: Batch
    selfies_batch: Any

    def to(self, device):
        return NextMolBatch(
            data_batch=self.data_batch.to(device),
            selfies_batch=self.selfies_batch.to(device),
        )
