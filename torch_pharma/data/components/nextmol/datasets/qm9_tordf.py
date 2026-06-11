"""QM9 torsional diffusion dataset loader (conformer prediction)."""

import copy
import random
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data.in_memory_dataset import nested_iter
from torch_geometric.data.separate import separate

from torch_pharma.data.components.nextmol.mol_mapping import (
    build_rdkit2rand_smiles_withoutH_mapping,
    get_smiles2selfies_mapping,
)
from torch_pharma.evaluation.nextmol.conformer_metrics import set_rdmol_positions
from torch_pharma.paths import NEXTMOL_QM9_TORDF, ensure_nextmol_dirs
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)

_SPLIT_FILES = {
    "train": ("processed_train.pt", "tordf.train"),
    "val": ("processed_val.pt", "tordf.val"),
    "valid": ("processed_val.pt", "tordf.val"),
    "test": ("processed_inference_test.pt", "tordf.test"),
}


class QM9TorDFDataset(Dataset):
    """Loads preprocessed QM9 TorDF caches from OSF ``GEOM-QM9`` layout.

    Default root: ``$TORCH_PHARMA_HOME/data/nextmol/tordf_qm9``.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split: str = "train",
        rand_smiles: str | bool = False,
        addHs: bool = False,
    ):
        ensure_nextmol_dirs()
        self.root = Path(root or NEXTMOL_QM9_TORDF)
        self.split = "valid" if split == "val" else split
        self.rand_smiles = rand_smiles
        self.addHs = addHs
        self.pos_std = 1.4182
        self.mode = "valid" if self.split in {"val", "valid"} else "train"

        processed_name, _ = _SPLIT_FILES.get(self.split, _SPLIT_FILES["train"])
        processed_path = self.root / processed_name

        if not processed_path.exists():
            self.data = None
            self.slices = None
            self.pos_list = []
            log.warning(
                "QM9 TorDF cache missing at %s — run scripts/install_nextmol_data.py",
                processed_path,
            )
            return

        self.data, self.slices, self.pos_list = torch.load(
            processed_path, map_location="cpu", weights_only=False
        )
        log.info(
            "Loaded QM9 TorDF %s split from %s (%d samples)",
            self.split,
            processed_path,
            len(self),
        )

    def __len__(self):
        if self.slices is None:
            return 0
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

    def _get_idx_data(self, idx: int):
        if not hasattr(self, "_data_list") or self._data_list is None:
            self._data_list = self.__len__() * [None]
        elif self._data_list[idx] is not None:
            return self._data_list[idx].clone()

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        data["pos"] = self.pos_list[idx]
        self._data_list[idx] = data.clone()
        return data.clone()

    def __getitem__(self, idx):
        data = self._get_idx_data(idx)
        assert isinstance(data["pos"], list)
        if self.mode == "train":
            data["pos"] = random.choice(data["pos"])
        else:
            rng = random.Random(idx)
            data["pos"] = rng.choice(data["pos"])

        data["pos"] -= data["pos"].mean(dim=0, keepdim=True)
        rdmol = copy.deepcopy(data["rdmol"])
        rdmol.RemoveAllConformers()
        data["rdmol"] = set_rdmol_positions(rdmol, data["pos"], removeHs=False)
        data["pos"] /= self.pos_std

        rdmol2smiles, output_smiles = build_rdkit2rand_smiles_withoutH_mapping(
            data.rdmol, self.rand_smiles, self.addHs
        )
        rdmol2smiles = rdmol2smiles.tolist()
        smiles2selfies, selfies_tokens, selfies = get_smiles2selfies_mapping(output_smiles)

        data["smiles"] = output_smiles
        data["selfies"] = selfies
        data["rdmol2smiles"] = rdmol2smiles

        rdmol2selfies = torch.zeros(
            (data.rdmol.GetNumAtoms(), len(selfies_tokens)), dtype=torch.float
        )
        rdmol2selfies_mask = torch.zeros((data.rdmol.GetNumAtoms(),), dtype=torch.bool)
        for i, v in enumerate(rdmol2smiles):
            if v in smiles2selfies:
                for j in smiles2selfies[v]:
                    rdmol2selfies[i, j] = 1
                rdmol2selfies_mask[i] = True
        data["rdmol2selfies"] = rdmol2selfies
        data["rdmol2selfies_mask"] = rdmol2selfies_mask
        return data
