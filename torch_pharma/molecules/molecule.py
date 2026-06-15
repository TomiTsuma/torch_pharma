from rdkit.Chem.rdForceFieldHelpers import UFFHasAllMoleculeParams, UFFOptimizeMolecule

from torch_pharma.data.utils import x_map as additional_node_map

class Molecule:
    def __init__(
        self,
        atom_types,
        bond_types,
        positions,
        charges,
        dataset_info,
        atom_types_pocket=None,
        positions_pocket=None,
        context=None,
        is_aromatic=None,
        hybridization=None,
        build_mol_with_addfeats=False,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=True,
        check_validity=False,
        build_obabel_mol=False,
    ):
        """
        atom_types: n      LongTensor
        charges: n         LongTensor
        bond_types: n x n  LongTensor
        positions: n x 3   FloatTensor
        atom_decoder: extracted from dataset_infos.
        """
        assert atom_types.dim() == 1 and atom_types.dtype == torch.long, (
            f"shape of atoms {atom_types.shape} " f"and dtype {atom_types.dtype}"
        )
        assert bond_types.dim() == 2 and bond_types.dtype == torch.long, (
            f"shape of bonds {bond_types.shape} --" f" {bond_types.dtype}"
        )
        assert len(atom_types.shape) == 1
        assert len(bond_types.shape) == 2
        assert len(positions.shape) == 2

        self.relax_mol = relax_mol
        self.max_relax_iter = max_relax_iter
        self.sanitize = sanitize
        self.check_validity = check_validity

        self.dataset_info = dataset_info
        self.atom_decoder = (
            dataset_info["atom_decoder"]
            if isinstance(dataset_info, dict)
            else self.dataset_info.atom_decoder
        )

        self.atom_types = atom_types.long()
        self.bond_types = bond_types.long()
        self.positions = positions
        self.positions_pocket = positions_pocket
        self.atom_types_pocket = atom_types_pocket
        self.charges = charges
        self.context = context

        if isinstance(is_aromatic, torch.Tensor):
            assert len(is_aromatic.shape) == 1
            assert (
                is_aromatic.max().item() <= len(additional_node_map["is_aromatic"]) - 1
            )
            self.is_aromatic = is_aromatic
        else:
            self.is_aromatic = None

        if isinstance(hybridization, torch.Tensor):
            assert len(hybridization.shape) == 1
            assert (
                hybridization.max().item()
                <= len(additional_node_map["hybridization"]) - 1
            )
            self.hybridization = hybridization
        else:
            self.hybridization = None

        self.additional_feats = isinstance(
            self.is_aromatic, torch.Tensor
        ) and isinstance(self.hybridization, torch.Tensor)
        self.build_mol_with_addfeats = build_mol_with_addfeats

        self.rdkit_mol = (
            self.build_molecule_openbabel()
            if build_obabel_mol
            else self.build_molecule()
        )
        self.num_nodes = len(atom_types)
        self.num_atom_types = len(self.atom_decoder)

    def build_molecule(self, verbose=False):
        """If positions is None,"""
        if verbose:
            print("building new molecule")

        mol = Chem.RWMol()

        if self.additional_feats and self.build_mol_with_addfeats:
            for atom, charge, is_aromatic, sp_hybridization in zip(
                self.atom_types, self.charges, self.is_aromatic, self.hybridization
            ):
                if atom == -1:
                    continue
                try:
                    a = Chem.Atom(self.atom_decoder[int(atom.item())])
                except:
                    continue
                if charge.item() != 0:
                    a.SetFormalCharge(charge.item())
                a.SetIsAromatic(additional_node_map["is_aromatic"][is_aromatic.item()])
                a.SetHybridization(
                    additional_node_map["hybridization"][sp_hybridization.item()]
                )
                mol.AddAtom(a)
                if verbose:
                    print("Atom added: ", atom.item(), self.atom_decoder[atom.item()])
        else:
            for atom, charge in zip(self.atom_types, self.charges):
                if atom == -1:
                    continue
                try:
                    a = Chem.Atom(self.atom_decoder[int(atom.item())])
                except:
                    a = Chem.Atom("H")
                if charge.item() != 0:
                    a.SetFormalCharge(charge.item())
                mol.AddAtom(a)
                if verbose:
                    print("Atom added: ", atom.item(), self.atom_decoder[atom.item()])

        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(
                    bond[0].item(),
                    bond[1].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
                if verbose:
                    print(
                        "bond added:",
                        bond[0].item(),
                        bond[1].item(),
                        edge_types[bond[0], bond[1]].item(),
                        bond_dict[edge_types[bond[0], bond[1]].item()],
                    )

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None

        # Set coordinates
        positions = self.positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                i,
                Point3D(
                    positions[i][0].item(),
                    positions[i][1].item(),
                    positions[i][2].item(),
                ),
            )
        mol.AddConformer(conf)

        if self.relax_mol:
            mol_uff = mol
            try:
                if self.sanitize:
                    Chem.SanitizeMol(mol_uff)
                self.uff_relax(mol_uff, self.max_relax_iter)
                if self.sanitize:
                    Chem.SanitizeMol(mol_uff)
                return mol_uff
            except (RuntimeError, ValueError) as e:
                if self.check_validity:
                    return self.compute_validity(mol)
                else:
                    return mol
        else:
            if self.check_validity:
                return self.compute_validity(mol)
            else:
                return mol

    def build_molecule_openbabel(self):
        """
        Build an RDKit molecule using openbabel for creating bonds
        Args:
            positions: N x 3
            atom_types: N
            atom_decoder: maps indices to atom types
        Returns:
            rdkit molecule
        """
        atom_types = [self.atom_decoder[a] for a in self.atom_types]

        try:
            with tempfile.NamedTemporaryFile() as tmp:
                tmp_file = tmp.name

                # Write xyz file
                write_xyz_file(self.positions, atom_types, tmp_file)

                # Convert to sdf file with openbabel
                # openbabel will add bonds
                obConversion = openbabel.OBConversion()
                obConversion.SetInAndOutFormats("xyz", "sdf")
                ob_mol = openbabel.OBMol()
                obConversion.ReadFile(ob_mol, tmp_file)

                obConversion.WriteFile(ob_mol, tmp_file)

                # Read sdf file with RDKit
                tmp_mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

            # Build new molecule. This is a workaround to remove radicals.
            mol = Chem.RWMol()
            for atom in tmp_mol.GetAtoms():
                mol.AddAtom(Chem.Atom(atom.GetSymbol()))
            mol.AddConformer(tmp_mol.GetConformer(0))

            for bond in tmp_mol.GetBonds():
                mol.AddBond(
                    bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
                )
            mol = self.process_obabel_molecule(mol, sanitize=True, largest_frag=True)
        except:
            return None

        return mol

    def process_obabel_molecule(
        self,
        rdmol,
        add_hydrogens=False,
        sanitize=False,
        relax_iter=0,
        largest_frag=False,
    ):
        """
        Apply filters to an RDKit molecule. Makes a copy first.
        Args:
            rdmol: rdkit molecule
            add_hydrogens
            sanitize
            relax_iter: maximum number of UFF optimization iterations
            largest_frag: filter out the largest fragment in a set of disjoint
                molecules
        Returns:
            RDKit molecule or None if it does not pass the filters
        """

        # Create a copy
        mol = Chem.Mol(rdmol)

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                warnings.warn("Sanitization failed. Returning None.")
                return None

        if add_hydrogens:
            mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

        if largest_frag:
            mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if sanitize:
                # sanitize the updated molecule
                try:
                    Chem.SanitizeMol(mol)
                except ValueError:
                    return None

        if relax_iter > 0:
            if not UFFHasAllMoleculeParams(mol):
                warnings.warn(
                    "UFF parameters not available for all atoms. " "Returning None."
                )
                return None

            try:
                self.uff_relax(mol, relax_iter)
                if sanitize:
                    # sanitize the updated molecule
                    Chem.SanitizeMol(mol)
            except (RuntimeError, ValueError) as e:
                return None

        return mol

    def uff_relax(self, mol, max_iter=200):
        """
        Uses RDKit's universal force field (UFF) implementation to optimize a
        molecule.
        """
        more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
        if more_iterations_required:
            warnings.warn(
                f"Maximum number of FF iterations reached. "
                f"Returning molecule after {max_iter} relaxation steps."
            )
        return more_iterations_required

    def compute_validity(self, mol, strict=False):
        if mol is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=False
                )
                if len(mol_frags) > 1:
                    return None
                else:
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    initial_adj = Chem.GetAdjacencyMatrix(
                        largest_mol, useBO=True, force=True
                    )
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol)

                    if sum([a.GetNumImplicitHs() for a in largest_mol.GetAtoms()]) > 0:
                        return None
                    if strict:
                        # sanitization changes bond order without throwing exceptions for certain cases
                        # https://github.com/rdkit/rdkit/blob/master/Docs/Book/RDKit_Book.rst#molecular-sanitization
                        # only consider change in BO to be wrong when difference is > 0.5 (not just kekulization difference)
                        adj2 = Chem.GetAdjacencyMatrix(
                            largest_mol, useBO=True, force=True
                        )
                        if not np.all(np.abs(initial_adj - adj2) < 1):
                            return None
                        # atom valencies are only correct when unpaired electrons are added
                        # when training data does not contain open shell systems, this should be considered an error
                        if (
                            sum(
                                [
                                    a.GetNumRadicalElectrons()
                                    for a in largest_mol.GetAtoms()
                                ]
                            )
                            > 0
                        ):
                            return None
            except:
                return None

        return mol

    def build_molecule_edm(self, positions, atom_types, dataset_info):
        atom_decoder = dataset_info["atom_decoder"]
        X, A, E = self.build_xae_molecule(positions, atom_types, dataset_info)
        mol = Chem.RWMol()
        for atom in X:
            a = Chem.Atom(atom_decoder[atom.item()])
            mol.AddAtom(a)

        all_bonds = torch.nonzero(A)
        for bond in all_bonds:
            mol.AddBond(
                bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()]
            )
        return mol

    def build_xae_molecule(self, positions, atom_types, dataset_info):
        """Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
        """
        atom_decoder = dataset_info["atom_decoder"]
        n = positions.shape[0]
        X = atom_types
        A = torch.zeros((n, n), dtype=torch.bool)
        E = torch.zeros((n, n), dtype=torch.int)

        pos = positions.unsqueeze(0)
        dists = torch.cdist(pos, pos, p=2).squeeze(0)
        for i in range(n):
            for j in range(i):
                pair = sorted([atom_types[i], atom_types[j]])
                if (
                    dataset_info["name"] == "qm9"
                    or dataset_info["name"] == "qm9_second_half"
                    or dataset_info["name"] == "qm9_first_half"
                ):
                    order = get_bond_order(
                        atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j]
                    )
                elif dataset_info["name"] == "drugs" or dataset_info["name"] == "aqm":
                    order = geom_predictor(
                        (atom_decoder[pair[0]], atom_decoder[pair[1]]),
                        dists[i, j],
                        limit_bonds_to_one=True,
                    )
                # TODO: a batched version of get_bond_order to avoid the for loop
                if order > 0:
                    # Warning: the graph should be DIRECTED
                    A[i, j] = 1
                    E[i, j] = order
        return X, A, E

