"""
From https://github.com/ehoogeboom/e3_diffusion_for_molecules/
"""
import tempfile
import pickle
import openbabel
import torch
import os
import warnings

import numpy as np

from pathlib import Path
from rdkit import Chem
from typing import Any, Dict, List, Optional, Tuple
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from torch_pharma.data.datasets.utils import download_qm9, TORCH_PHARMA_HOME, process_xyz_files, process_xyz_gdb9, get_dataset_info
from torch_pharma.data.datasets.base import BaseDataset
from torch_pharma.molecules.chemistry import mol2smiles, build_molecule, process_molecule


patch_typeguard()  # use before @typechecked


# Check if a string can be converted to an int, without throwing an error.
def is_int(str):
    try:
        int(str)
        return True
    except:
        return False


def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass


class QM9Dataset(BaseDataset):
    def __init__(self, calculate_thermo=True):
        super().__init__()
        self.gdb9_dir = os.path.join(TORCH_PHARMA_HOME, "qm9")
        self.calculate_thermo = calculate_thermo

    def download(self):
        download_qm9()

    def gen_splits_gdb9(self, cleanup=True):
        """
        Generate GDB9 training/validation/test splits used.

        First, use the file "uncharacterized.txt" in the GDB9 figshare to find a
        list of excluded molecules.

        Second, create a list of molecule ids, and remove the excluded molecule
        indices.

        Third, assign 100k molecules to the training set, 10% to the test set,
        and the remaining to the validation set.

        Finally, generate torch.tensors which give the molecule ids for each
        set.
        """
        gdb9_txt_excluded = os.path.join(self.gdb9_dir, "uncharacterized.txt")
        excluded_strings = []

        with open(gdb9_txt_excluded) as f:
            lines = f.readlines()
            excluded_strings = [line.split()[0]
                                for line in lines if len(line.split()) > 0]

        excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

        assert len(excluded_idxs) == 3054, "There should be exactly 3054 excluded atoms. Found {}".format(
            len(excluded_idxs))

        # Now, create a list of indices
        Ngdb9 = 133885
        Nexcluded = 3054

        included_idxs = np.array(
            sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

        # Now generate random permutations to assign molecules to training/validation/test sets.
        Nmols = Ngdb9 - Nexcluded

        Ntrain = 100000
        Ntest = int(0.1*Nmols)
        Nvalid = Nmols - (Ntrain + Ntest)

        # Generate random permutation
        np.random.seed(0)
        data_perm = np.random.permutation(Nmols)

        # Now use the permutations to generate the indices of the dataset splits.
        # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

        train, valid, test, extra = np.split(
            data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

        assert(len(extra) == 0), "Split was inexact {} {} {} {}".format(
            len(train), len(valid), len(test), len(extra))

        train = included_idxs[train]
        valid = included_idxs[valid]
        test = included_idxs[test]

        splits = {"train": train, "valid": valid, "test": test}

        return splits
            
    def get_thermo_dict(self):
        """
        Get dictionary of thermochemical energy to subtract off from
        properties of molecules.

        Probably would be easier just to just precompute this and enter it explicitly.
        """
        gdb9_txt_thermo = os.path.join(self.gdb9_dir, "atomref.txt") # FIXED: self.gdb9dir -> self.gdb9_dir

        # Loop over file of thermochemical energies
        therm_targets = ["zpve", "U0", "U", "H", "G", "Cv"]

        # Dictionary that
        id2charge = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

        # Loop over file of thermochemical energies
        therm_energy = {target: {} for target in therm_targets}
        with open(gdb9_txt_thermo) as f:
            for line in f:
                # If line starts with an element, convert the rest to a list of energies.
                split = line.split()

                # Check charge corresponds to an atom
                if len(split) == 0 or split[0] not in id2charge.keys():
                    continue

                # Loop over learning targets with defined thermochemical energy
                for therm_target, split_therm in zip(therm_targets, split[1:]):
                    therm_energy[therm_target][id2charge[split[0]]
                                            ] = float(split_therm)

        # Cleanup file when finished.
        # cleanup_file(gdb9_txt_thermo, cleanup)

        return therm_energy

    def add_thermo_targets(self, data, therm_energy_dict):
        """
        Adds a new molecular property, which is the thermochemical energy.

        Parameters
        ----------
        data : ?????
            QM9 dataset split.
        therm_energy : dict
            Dictionary of thermochemical energies for relevant properties found using :get_thermo_dict:
        """
        # Get the charge and number of charges
        charge_counts = self.get_unique_charges(data["charges"])

        # Now, loop over the targets with defined thermochemical energy
        for target, target_therm in therm_energy_dict.items():
            thermo = np.zeros(len(data[target]))

            # Loop over each charge, and multiplicity of the charge
            for z, num_z in charge_counts.items():
                if z == 0:
                    continue
                # Now add the thermochemical energy per atomic charge * the number of atoms of that type
                thermo += target_therm[z] * num_z

            # Now add the thermochemical energy as a property
            data[target + "_thermo"] = thermo

        return data


    def get_unique_charges(self, charges):
        """
        Get count of each charge for each molecule.
        """
        # Create a dictionary of charges
        charge_counts = {z: np.zeros(len(charges), dtype=np.int64)
                        for z in np.unique(charges)}

        # Loop over molecules, for each molecule get the unique charges
        for idx, mol_charges in enumerate(charges):
            # For each molecule, get the unique charge and multiplicity
            for z, num_z in zip(*np.unique(mol_charges, return_counts=True)):
                # Store the multiplicity of each charge in charge_counts
                charge_counts[z][idx] = num_z

        return charge_counts

    def process(self):
        self.download()
        gdb9_tar_data = os.path.join(self.gdb9_dir, "dsgdb9nsd.xyz.tar.bz2")
        splits = self.gen_splits_gdb9()

        #Process GDB9 dataset and return a dictionary of splits
        gdb9_data = {}
        for split, split_idx in splits.items():
            gdb9_data[split] = process_xyz_files(
                gdb9_tar_data, process_xyz_gdb9, file_idx_list=split_idx, stack=True
            )

        # Subtract thermochemical energy if it is so required
        if self.calculate_thermo:
            # Process thermochemical energy information into a dictionary
            therm_energy = self.get_thermo_dict()

            # For each of train/validation/test split, add the thermochemical energy
            for split_idx, split_data in gdb9_data.items():
                gdb9_data[split_idx] = self.add_thermo_targets(split_data, therm_energy)

        for split, data in gdb9_data.items():
            savedir = os.path.join(self.gdb9_dir, split+".npz")
            np.savez_compressed(savedir, **data)

    def compute_smiles(self, dataset, remove_h):
        class StaticArgs:
            def __init__(self, dataset, remove_h):
                self.dataset = dataset
                self.batch_size = 1
                self.num_workers = 1
                self.filter_n_atoms = None
                self.data_dir = TORCH_PHARMA_HOME
                self.remove_h = remove_h
                self.include_charges = True
                self.subtract_thermo = True
                self.force_download = False
                self.create_pyg_graphs = False
                self.num_radials = 1
                self.device = "cpu"
                self.num_train = -1
                self.num_valid = -1
                self.num_test = -1
                self.shuffle = True
                self.drop_last = True
        args_dataset = StaticArgs(dataset, remove_h)
        dataloaders, _ = dataset.retrieve_loaders(args_dataset)
        dataset_info = get_dataset_info(args_dataset.dataset, args_dataset.remove_h)
        n_types = 4 if remove_h else 5
        mols_smiles = []

        for i, data in enumerate(dataloaders['train']):
            positions = data['positions'][0].view(-1,3).numpy()
            one_hot = data['one_hot'][0].view(-1, n_types).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()
            charges  = data["charges"][0].squeeze(-1)

            mol = build_molecule(torch.tensor(positions), torch.tensor(atom_type), dataset_info, charges)
            mol_str = mol2smiles(mol)
            if mol_str is not None:
                mols_smiles.append(mol_str)
            if i % 1000 == 0:
                print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i)/len(dataloaders["train"])))
        
        return mols_smiles



