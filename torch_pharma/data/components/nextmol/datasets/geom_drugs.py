"""Geom-DRUGS dataset loader for NExT-Mol large-molecule workflows."""

import copy
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
from torch_pharma.paths import NEXTMOL_GEOM_DRUGS, ensure_nextmol_dirs
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)

_SPLIT_ALIASES = {"val": "valid", "validation": "valid"}


class GeomDrugsTorDFDataset(Dataset):
    """Loads preprocessed Geom-DRUGS JODO caches from OSF ``geom_drugs_jodo``.

    Default root: ``$TORCH_PHARMA_HOME/data/nextmol/geom_drugs``.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split: str = "train",
        rand_smiles: str | bool = False,
        addHs: bool = False,
    ):
        ensure_nextmol_dirs()
        self.root = Path(root or NEXTMOL_GEOM_DRUGS)
        self.split = _SPLIT_ALIASES.get(split, split)
        self.rand_smiles = rand_smiles
        self.addHs = addHs
        self.pos_std = 2.3860

        processed_path = self.root / "processed_data.pt"
        split_path = self.root / "split_dict_geom_drug_1.pt"

        if not processed_path.exists():
            self.data = None
            self.slices = None
            self.indices = []
            log.warning(
                "Geom-DRUGS cache missing at %s — run scripts/install_nextmol_data.py",
                processed_path,
            )
            return

        self.data, self.slices = torch.load(processed_path, map_location="cpu", weights_only=False)
        splits = torch.load(split_path, map_location="cpu", weights_only=False)
        if self.split not in splits:
            raise ValueError(f"Unknown split {split!r}; available: {list(splits.keys())}")
        self.indices = splits[self.split].tolist()
        log.info(
            "Loaded Geom-DRUGS %s split from %s (%d samples)",
            self.split,
            self.root,
            len(self.indices),
        )

    def __len__(self):
        return len(self.indices)

    def _get_idx_data(self, graph_idx: int):
        if not hasattr(self, "_cache"):
            self._cache = {}
        if graph_idx in self._cache:
            return self._cache[graph_idx].clone()

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=graph_idx,
            slice_dict=self.slices,
            decrement=False,
        )
        self._cache[graph_idx] = data.clone()
        return data.clone()

    def __getitem__(self, idx):
        data = self._get_idx_data(self.indices[idx])
        data["rdmol"] = copy.deepcopy(data["rdmol"])
        assert data["rdmol"].GetNumConformers() == 1
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
