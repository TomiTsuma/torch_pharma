"""QM9 SELFIES language-model dataset (loads NExT-Mol QM92014 caches)."""

from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data.separate import separate

from torch_pharma.paths import NEXTMOL_QM9_LM, ensure_nextmol_dirs
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)

_SPLIT_ALIASES = {"val": "valid", "validation": "valid"}


class QM9LMDataset(Dataset):
    """Loads preprocessed QM9 SELFIES data from OSF ``QM92014`` layout.

    Expected files under ``root`` (default ``$TORCH_PHARMA_HOME/data/nextmol/qm9_lm``):

    - ``processed/data_qm9.pt``
    - ``processed/split_dict_qm9.pt``
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split: str = "train",
        transform=None,
        pre_transform=None,
    ):
        ensure_nextmol_dirs()
        self.root = Path(root or NEXTMOL_QM9_LM)
        self.split = _SPLIT_ALIASES.get(split, split)
        self.transform = transform
        self.pre_transform = pre_transform

        data_path = self.root / "processed" / "data_qm9.pt"
        split_path = self.root / "processed" / "split_dict_qm9.pt"

        if not data_path.exists() or not split_path.exists():
            self.data = None
            self.slices = None
            self.indices = []
            log.warning(
                "QM9 LM cache missing under %s — run scripts/install_nextmol_data.py",
                self.root,
            )
            return

        self.data, self.slices = torch.load(data_path, map_location="cpu", weights_only=False)
        splits = torch.load(split_path, map_location="cpu", weights_only=False)
        if self.split not in splits:
            raise ValueError(f"Unknown split {split!r}; available: {list(splits.keys())}")
        self.indices = splits[self.split].tolist()
        log.info(
            "Loaded QM9 LM %s split from %s (%d samples)",
            self.split,
            self.root,
            len(self.indices),
        )

    def __len__(self):
        return len(self.indices)

    def _get_data(self, dataset_idx: int):
        if self.data is None:
            raise RuntimeError("QM9 LM dataset not installed")
        return separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=dataset_idx,
            slice_dict=self.slices,
            decrement=False,
        )

    def __getitem__(self, idx):
        data = self._get_data(self.indices[idx])
        if self.transform is not None:
            data = self.transform(data)
        return {
            "selfies": data.selfies,
            "smiles": getattr(data, "smiles", getattr(data, "cano_smiles_woh", "")),
        }
