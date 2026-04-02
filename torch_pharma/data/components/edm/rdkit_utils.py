"""
Based on rdkit_functions.py from bio-diffusion.
"""

import tempfile
import warnings
import torch
import pickle
import os
import numpy as np

try:
    import openbabel
    HAS_OPENBABEL = True
except ImportError:
    HAS_OPENBABEL = False

from rdkit import Chem
from typing import Any, Dict, List, Optional, Tuple
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams

from . import get_bond_order_batch, get_bond_length_arrays
from torch_pharma.utils.logging import get_pylogger

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

log = get_pylogger(__name__)

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]


class BasicMolecularMetrics(object):
    def __init__(
        self,
        dataset_info: Dict[str, Any],
        data_dir: str,
        dataset_smiles_list: Optional[np.ndarray] = None
    ):
        self.atom_decoder = dataset_info["atom_decoder"]
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # retrieve dataset smiles only for the QM9 dataset currently
        if dataset_smiles_list is None and "QM9" in dataset_info["name"]:
            bonds = get_bond_length_arrays(self.dataset_info["atom_encoder"])
            self.dataset_info["bonds1"], self.dataset_info["bonds2"], self.dataset_info["bonds3"] = (
                bonds[0], bonds[1], bonds[2]
            )
            self.dataset_smiles_list = retrieve_qm9_smiles(self.dataset_info, data_dir)

    @typechecked
    def compute_validity(self, rdmols: List[Chem.RWMol]) -> Tuple[List[str], float]:
        valid = []
        for mol in rdmols:
            smiles = mol2smiles(mol)
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)
        return valid, len(valid) / len(rdmols)

    @typechecked
    def compute_uniqueness(self, valid: List[str]) -> Tuple[List[str], float]:
        # note: `valid` is a list of SMILES strings
        return list(set(valid)), len(set(valid)) / len(valid)

    @typechecked
    def compute_novelty(self, unique: List[str]) -> Tuple[List[str], float]:
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    @typechecked
    def evaluate_rdmols(self, rdmols: List[Chem.RWMol], verbose: bool = True) -> List[float]:
        valid, validity = self.compute_validity(rdmols)
        if verbose:
            log.info(f"Validity over {len(rdmols)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            if verbose:
                log.info(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")
            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                if verbose:
                    log.info(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            uniqueness = 0.0
            novelty = 0.0
        return [validity, uniqueness, novelty]

    @typechecked
    def evaluate(
        self,
        generated: List[Tuple[torch.Tensor, ...]]
    ) -> List[float]:
        """
        note: `generated` is a list of pairs (`positions`: [n x 3], `atom_types`: n [int]; `charges`: n [int]);
        also note: the positions and atom types should already be masked.
        """
        rdmols = [build_molecule(*graph, self.dataset_info) for graph in generated]
        return self.evaluate_rdmols(rdmols)


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)

@typechecked
def build_molecule(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    dataset_info: Dict[str, Any],
    charges: Optional[TensorType["num_nodes"]] = None,
    add_coords: bool = False,
    use_openbabel: bool = False
) -> Chem.RWMol:
    if use_openbabel:
        if not HAS_OPENBABEL:
            log.warning("OpenBabel not installed, falling back to EDM method.")
            return make_mol_edm(positions, atom_types, dataset_info, add_coords)
        return make_mol_openbabel(positions, atom_types, dataset_info["atom_decoder"])
    else:
        return make_mol_edm(positions, atom_types, dataset_info, add_coords)

@typechecked
def make_mol_openbabel(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    atom_decoder: Dict[int, str]
) -> Chem.RWMol:
    # Need write_xyz_file equivalent.
    # For simplicity, we implement it inline or use a utility if available.
    atom_types_str = [atom_decoder[x.item()] for x in atom_types]
    
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
        tmp_file = tmp.name
        with open(tmp_file, "w") as f:
            f.write(f"{len(positions)}\n\n")
            for pos, atom in zip(positions, atom_types_str):
                f.write(f"{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    try:
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)
        
        sdf_file = tmp_file.replace(".xyz", ".sdf")
        obConversion.WriteFile(ob_mol, sdf_file)
        
        mol = Chem.SDMolSupplier(sdf_file, sanitize=False)[0]
        os.remove(sdf_file)
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
            
    return mol

@typechecked
def make_mol_edm(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    dataset_info: Dict[str, Any],
    add_coords: bool
) -> Chem.RWMol:
    n = len(positions)
    limit_bonds_to_one = "GEOM" in dataset_info.get("name", "")

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0).view(-1)
    
    # Efficient cartesian product
    atom_types_expanded = atom_types.repeat_interleave(n)
    atom_types_repeated = atom_types.repeat(n)
    
    E_full = get_bond_order_batch(
        atom_types_expanded, atom_types_repeated, dists, dataset_info, limit_bonds_to_one=limit_bonds_to_one
    ).view(n, n)
    
    E = torch.tril(E_full, diagonal=-1)
    A = E.bool()
    
    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(dataset_info["atom_decoder"][atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        bond_type = bond_dict[E[bond[0], bond[1]].item()]
        if bond_type:
            mol.AddBond(bond[0].item(), bond[1].item(), bond_type)

    if add_coords:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (positions[i, 0].item(),
                                     positions[i, 1].item(),
                                     positions[i, 2].item()))
        mol.AddConformer(conf)

    return mol

@typechecked
def process_molecule(
    rdmol: Chem.Mol,
    add_hydrogens: bool = False,
    sanitize: bool = False,
    relax_iter: int = 0,
    largest_frag: bool = False
) -> Optional[Chem.Mol]:
    if rdmol is None:
        return None
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None

    if relax_iter > 0:
        if not UFFHasAllMoleculeParams(mol):
            return None
        try:
            UFFOptimizeMolecule(mol, maxIters=relax_iter)
            if sanitize:
                Chem.SanitizeMol(mol)
        except Exception:
            return None

    return mol
