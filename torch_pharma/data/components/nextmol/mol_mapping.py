"""RDKit atom to SELFIES token alignment (ported from NExT-Mol mol_mapping_utils.py)."""

import copy
import random
import re

import numpy as np
import selfies as sf
from rdkit import Chem

SMI_REGEX_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
SMI_REGEX = re.compile(SMI_REGEX_PATTERN)
filter_regex = re.compile(r"[\(\)\[\]\{\}=\d\#\-+@%\/\\]")
two_letter_atoms = {
    "Al": "A",
    "Si": "Y",
    "Cl": "L",
    "As": "X",
    "Br": "R",
    "Hg": "G",
    "Bi": "B",
}
invalid_int = -99999


def split_smiles(smiles, filter_=False):
    smiles_tokens = []
    start_pos = []
    for match in SMI_REGEX.finditer(smiles):
        result = match.group()
        if result:
            smiles_tokens.append(result)
            start_pos.append(match.start())
    if filter_:
        new_tokens, new_pos = [], []
        for t, pos in zip(smiles_tokens, start_pos):
            if t in {"=", "(", ")", "#", *map(str, range(10))} or t.startswith("%"):
                continue
            new_tokens.append(t)
            new_pos.append(pos)
        return new_tokens, new_pos
    return smiles_tokens, start_pos


def sf_encode_and_attribute(smiles: str):
    selfies_str, attribution = sf.encoder(smiles, attribute=True)
    selfies_tokens = list(sf.split_selfies(selfies_str))
    attribution_tokens = [attr.token for attr in attribution]
    assert selfies_tokens == attribution_tokens
    for i in range(len(attribution)):
        attribution[i].index = i
    return selfies_str, attribution, selfies_tokens


def obtain_atoms_from_smiles(text, regex):
    output = regex.sub("", text).upper()
    for atom, replacement in two_letter_atoms.items():
        output = output.replace(atom.upper(), replacement)
    masked_text = regex.sub("*", text).upper()
    masked_text = re.sub(r"AL|SI|CL|AS|BR|HG|BI", "X*", masked_text)
    masked_text = np.asarray(list(masked_text))
    mapping = np.where(masked_text != "*")[0]
    return output, mapping


def get_smiles2selfies_mapping(cano_smiles: str):
    """Build mapping from canonical SMILES atom indices to SELFIES token indices."""
    selfies_str, attribution, _ = sf_encode_and_attribute(cano_smiles)
    smiles_tokens, start_poses = split_smiles(cano_smiles, True)

    atom_set = set()
    for attr in attribution:
        if attr.attribution is None:
            continue
        for item in attr.attribution:
            atom_set.add((item.index, item.token))
    atom_list = sorted(atom_set, key=lambda x: x[0])
    atom_mapping = {atom: i for i, atom in enumerate(atom_list)}

    selfies2smiles = []
    for attr in attribution:
        selfies2smiles.append([])
        if attr.attribution is None:
            continue
        for item in attr.attribution:
            atom_index = atom_mapping[(item.index, item.token)]
            sp = start_poses[atom_index]
            for i in range(len(item.token)):
                if item.token[i].isalpha():
                    selfies2smiles[-1].append(sp + i)

    smiles2selfies: dict[int, list[int]] = {}
    for i, smiles_id_list in enumerate(selfies2smiles):
        for smiles_id in smiles_id_list:
            smiles2selfies.setdefault(smiles_id, []).append(i)

    selfies_tokens = list(sf.split_selfies(selfies_str))
    return smiles2selfies, selfies_tokens, selfies_str


def build_rdkit2cano_smiles_withoutH_mapping(rdmol):
    """Map RDKit atom indices (with H) to canonical SMILES character positions."""
    rdmol = copy.deepcopy(rdmol)
    for atom in rdmol.GetAtoms():
        atom.SetProp("atom_index", str(atom.GetIdx()))
    rdmol_woh = Chem.RemoveHs(rdmol)
    canonical_smiles = Chem.MolToSmiles(rdmol_woh, canonical=True)
    smiles_atom_order = rdmol_woh.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
    rdmol_woh = Chem.RenumberAtoms(rdmol_woh, list(smiles_atom_order))

    rdmol_wh2rdmol_woh = np.full(rdmol.GetNumAtoms(), invalid_int)
    for i, atom in enumerate(rdmol_woh.GetAtoms()):
        rdmol_wh2rdmol_woh[int(atom.GetProp("atom_index"))] = atom.GetIdx()

    symbols = []
    for atom in rdmol_woh.GetAtoms():
        symbol = atom.GetSymbol()
        if len(symbol) == 2:
            symbols.append(two_letter_atoms[symbol])
        else:
            symbols.append(symbol)
    atoms_in_rdmol_woh = "".join(symbols).upper()
    atoms_in_smiles, output2input = obtain_atoms_from_smiles(canonical_smiles, filter_regex)

    if atoms_in_rdmol_woh == atoms_in_smiles:
        rdmol_woh2smiles = output2input
    else:
        add_h = 0
        rdmol_woh2smiles = []
        for j in range(len(atoms_in_smiles)):
            if j == len(atoms_in_smiles) - 1 and (j - add_h) == len(atoms_in_rdmol_woh):
                continue
            if atoms_in_smiles[j].upper() != atoms_in_rdmol_woh[j - add_h].upper():
                add_h += 1
            else:
                rdmol_woh2smiles.append(output2input[j])
        rdmol_woh2smiles = np.asarray(rdmol_woh2smiles)
        assert len(rdmol_woh2smiles) == len(atoms_in_rdmol_woh)

    rdmol_wh2smiles = []
    for i, j in enumerate(rdmol_wh2rdmol_woh):
        j = int(j)
        if j == invalid_int:
            rdmol_wh2smiles.append(invalid_int)
        else:
            rdmol_wh2smiles.append(rdmol_woh2smiles[j])
    return np.asarray(rdmol_wh2smiles), canonical_smiles


def build_rdkit2rand_smiles_withoutH_mapping(rdmol, rand_smiles=None, addHs=False):
    """Map RDKit atom indices (with H) to randomized SMILES character positions."""
    rdmol = copy.deepcopy(rdmol)
    for atom in rdmol.GetAtoms():
        atom.SetProp("atom_index", str(atom.GetIdx()))
    rdmol_woh = rdmol if addHs else Chem.RemoveHs(rdmol)

    if rand_smiles == "restricted":
        random_order = list(range(rdmol_woh.GetNumAtoms()))
        random.shuffle(random_order)
        random_mol = Chem.RenumberAtoms(rdmol_woh, newOrder=random_order)
        output_smiles = Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
        smiles_atom_order = random_mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
        rdmol_woh = Chem.RenumberAtoms(random_mol, list(smiles_atom_order))
    elif rand_smiles in {"None", "none", "False", "false"} or (not rand_smiles):
        output_smiles = Chem.MolToSmiles(rdmol_woh, canonical=False)
        smiles_atom_order = rdmol_woh.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
        rdmol_woh = Chem.RenumberAtoms(rdmol_woh, list(smiles_atom_order))
    elif rand_smiles == "canonical":
        output_smiles = Chem.MolToSmiles(rdmol_woh, canonical=True)
        smiles_atom_order = rdmol_woh.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
        rdmol_woh = Chem.RenumberAtoms(rdmol_woh, list(smiles_atom_order))
    else:
        raise NotImplementedError(f"rand_smiles={rand_smiles!r} not supported")

    rdmol_wh2rdmol_woh = np.full(rdmol.GetNumAtoms(), invalid_int)
    for i, atom in enumerate(rdmol_woh.GetAtoms()):
        rdmol_wh2rdmol_woh[int(atom.GetProp("atom_index"))] = atom.GetIdx()

    symbols = []
    for atom in rdmol_woh.GetAtoms():
        symbol = atom.GetSymbol()
        if len(symbol) == 2:
            symbols.append(two_letter_atoms[symbol])
        else:
            symbols.append(symbol)
    atoms_in_rdmol_woh = "".join(symbols).upper()
    atoms_in_smiles, output2input = obtain_atoms_from_smiles(output_smiles, filter_regex)

    if atoms_in_rdmol_woh == atoms_in_smiles:
        rdmol_woh2smiles = output2input
    else:
        add_h = 0
        rdmol_woh2smiles = []
        for j in range(len(atoms_in_smiles)):
            if j == len(atoms_in_smiles) - 1 and (j - add_h) == len(atoms_in_rdmol_woh):
                continue
            if atoms_in_smiles[j].upper() != atoms_in_rdmol_woh[j - add_h].upper():
                add_h += 1
            else:
                rdmol_woh2smiles.append(output2input[j])
        rdmol_woh2smiles = np.asarray(rdmol_woh2smiles)
        assert len(rdmol_woh2smiles) == len(atoms_in_rdmol_woh)

    rdmol_wh2smiles = []
    for i, j in enumerate(rdmol_wh2rdmol_woh):
        j = int(j)
        if j == invalid_int:
            rdmol_wh2smiles.append(invalid_int)
        else:
            rdmol_wh2smiles.append(rdmol_woh2smiles[j])
    return np.asarray(rdmol_wh2smiles), output_smiles
