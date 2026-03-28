import torch
import os
import tempfile
import warnings
import openbabel
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
from typing import Any, Dict, List, Optional, Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

# Constants
bond_dict = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC
}

def write_xyz_file(positions, atom_types, filename):
    with open(filename, 'w') as f:
        f.write(f"{len(positions)}\n\n")
        # atom_types might be strings or indices. If indices, they need decoding.
        # But this is a helper for make_mol_openbabel
        for pos, atom in zip(positions, atom_types):
            f.write(f"{atom} {pos[0]} {pos[1]} {pos[2]}\n")

def uff_relax(mol, relax_iter):
    AllChem.UFFOptimizeMolecule(mol, maxIters=relax_iter)

def get_bond_order_batch(atoms1, atoms2, dists, dataset_info, limit_bonds_to_one=False):
    """
    Heuristic to estimate bond order based on distances and covalent radii.
    Simplified implementation for demonstration.
    """
    # This is a simplified version. In a real scenario, you'd use a more robust lookup.
    # For now, let's assume single bonds if distance is within some threshold.
    adj = (dists < 2.0).long() 
    return adj

@typechecked
def mol2smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
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
        return make_mol_openbabel(positions, atom_types, dataset_info["atom_decoder"])
    else:
        return make_mol_edm(positions, atom_types, dataset_info, add_coords)

@typechecked
def make_mol_openbabel(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    atom_decoder: List[str] # Changed from Dict[int, str] to match qm9.py usage
) -> Chem.RWMol:
    atom_symbols = [atom_decoder[x] for x in atom_types]

    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
        tmp_file = tmp.name

    try:
        write_xyz_file(positions, atom_symbols, tmp_file)
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)
        # Note: We don't write to the same file because RDKit might be reading it
        sdf_file = tmp_file.replace(".xyz", ".sdf")
        obConversion.WriteFile(ob_mol, sdf_file)
        mol = Chem.SDMolSupplier(sdf_file, sanitize=False)[0]
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        if os.path.exists(tmp_file.replace(".xyz", ".sdf")):
            os.remove(tmp_file.replace(".xyz", ".sdf"))

    return mol

@typechecked
def make_mol_edm(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    dataset_info: Dict[str, Any],
    add_coords: bool
) -> Chem.RWMol:
    n = len(positions)
    limit_bonds_to_one = "GEOM" in dataset_info["name"]

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0).view(-1)
    atoms1, atoms2 = torch.cartesian_prod(atom_types, atom_types).T
    E_full = get_bond_order_batch(
        atoms1, atoms2, dists, dataset_info, limit_bonds_to_one=limit_bonds_to_one
    ).view(n, n)
    E = torch.tril(E_full, diagonal=-1)
    A = E.bool()
    X = atom_types

    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(dataset_info["atom_decoder"][atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(),
                    bond_dict[E[bond[0], bond[1]].item()])

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
    mol = Chem.Mol(rdmol)

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            warnings.warn('Sanitization failed. Returning None.')
            return None

    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

    if largest_frag:
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None

    if relax_iter > 0:
        if not UFFHasAllMoleculeParams(mol):
            warnings.warn('UFF parameters not available for all atoms. Returning None.')
            return None

        try:
            uff_relax(mol, relax_iter)
            if sanitize:
                Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError):
            return None

    return mol
