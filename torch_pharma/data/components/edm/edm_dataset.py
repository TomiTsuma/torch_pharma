# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch
import os
import numpy as np
import logging
import torch_geometric

from typing import Any, Dict, Optional, Tuple, Union
from torch_geometric.data import Data, Dataset, Batch

from omegaconf import DictConfig
from functools import partial
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from torch_pharma.data.components.edm.helper import _normalize
from torch_pharma.data.components.edm.protein_graph_dataset import ProteinGraphDataset
from torch_pharma.data.components.edm.collate import PreprocessQM9
from torch_pharma.utils.logging import get_pylogger

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

log = get_pylogger(__name__)

SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def set_worker_sharing_strategy(worker_id: int):
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


@typechecked
def _edge_features(
    batch: Batch,
    coords_key: str = "x"
) -> Tuple[
    TensorType["num_edges", "num_edge_scalar_features"],
    TensorType["num_edges", "num_edge_vector_features", 3]
]:
    coords = batch[coords_key]
    E_vectors = coords[batch.edge_index[0]] - coords[batch.edge_index[1]]
    radial = torch.sum(E_vectors ** 2, dim=1).unsqueeze(-1)

    edge_s = radial
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


@typechecked
def _node_features(
    batch: Batch,
    coords_key: str = "x",
    edm_sampling: bool = False
) -> Tuple[
    Union[
        Dict[
            str,
            Union[
                TensorType["num_nodes", "num_atom_types"],
                torch.Tensor  # note: for when `include_charges=False`
            ]
        ],
        TensorType["num_nodes", "num_node_scalar_features"],
        Optional[torch.Tensor]
    ],
    TensorType["num_nodes", "num_node_vector_features", 3]
]:
    # construct invariant node features
    if hasattr(batch, "h"):
        node_s = batch.h
    elif edm_sampling:
        node_s = None
    else:
        node_s = {"categorical": batch.one_hot, "integer": batch.charges}
        node_s["categorical"], node_s["integer"] = (
            map(torch.nan_to_num, (node_s["categorical"], node_s["integer"]))
        )

    # build equivariant node features
    coords = batch[coords_key]
    orientations = ProteinGraphDataset._orientations(coords)
    node_v = torch.nan_to_num(orientations)

    return node_s, node_v


class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Geometric Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    create_pyg_graphs: bool, optional
        If true, return PyTorch Geometric graphs when requesting dataset examples
    num_radials: int, optional
        Number of radial (distance) features to compute for each edge
    device: Union[torch.device, str], optional
        On which device to create graph features
    remove_zero_charge_molecules: bool, optional
        Whether to filter out from the dataset molecules with a total of zero charge
    """

    def __init__(
        self,
        data,
        included_species=None,
        num_pts=-1,
        shuffle=True,
        subtract_thermo=True,
        create_pyg_graphs=True,
        num_radials=1,
        device="cpu",
        remove_zero_charge_molecules=True
    ):

        self.data = data
        self.num_radials = num_radials
        self.device = device

        if remove_zero_charge_molecules:
            nonzero_charge_molecule_mask = self.data["charges"].sum(-1) > 0
            self.data = {key: val[nonzero_charge_molecule_mask] for key, val in self.data.items()}

        if num_pts < 0:
            self.num_pts = len(data["charges"])
        else:
            if num_pts > len(data["charges"]):
                logging.warning("Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!".format(
                    num_pts, len(data["charges"])))
                self.num_pts = len(data["charges"])
            else:
                self.num_pts = num_pts

        if included_species is None:
            # i.e., if included species is not specified
            included_species = torch.unique(self.data["charges"], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split("_")[0] for key in self.data.keys() if key.endswith("_thermo")]
            if len(thermo_targets) == 0:
                logging.warning("No thermochemical targets included! Try reprocessing dataset with --force-download!")
            else:
                logging.info("Removing thermochemical energy from targets {}".format(" ".join(thermo_targets)))
            for key in thermo_targets:
                self.data[key] -= self.data[key + "_thermo"].to(self.data[key].dtype)

        self.included_species = included_species

        self.data["one_hot"] = self.data["charges"].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)

        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {"num_species": self.num_species, "max_charge": self.max_charge}

        # get a dictionary of statistics for all properties that are one-dimensional tensors
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data["charges"]))[:self.num_pts]
        else:
            self.perm = None

        # determine which featurization method to use for requested dataset examples
        self.create_pyg_graphs = create_pyg_graphs

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val)
                      is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    @typechecked
    def _featurize_as_graph(self, molecule: Dict[str, Any], dtype: torch.dtype = torch.float32) -> Data:
        with torch.no_grad():
            mol_index = molecule["index"].unsqueeze(-1)
            coords = molecule["positions"].type(dtype)

            mask = molecule["charges"] > 0
            coords[~mask] = 0.0  # ensure missing nodes are assigned no edges

            # derive edges without self-loops
            edge_mask = mask.unsqueeze(0) * mask.unsqueeze(1)
            diag_mask = ~torch.eye(edge_mask.shape[0], dtype=torch.bool, device=edge_mask.device)
            edge_mask *= diag_mask
            edge_index = torch.stack(torch.where(edge_mask))

            one_hot = molecule["one_hot"].type(torch.float32)
            charges = molecule["charges"].type(torch.float32)

            conditional_properties = {
                key: value.reshape(1).type(dtype) for key, value in molecule.items()
                if key not in ["num_atoms", "charges", "positions", "index", "one_hot", "atom_mask"]
            }
            return torch_geometric.data.Data(
                one_hot=one_hot,
                charges=charges,
                x=coords,
                mol_index=mol_index,
                edge_index=edge_index,
                mask=mask,
                **conditional_properties
            )

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        item = {key: val[idx] for key, val in self.data.items()}
        return self._featurize_as_graph(item) if self.create_pyg_graphs else item


def initialize_datasets(args, data_dir, dataset, subset=None, splits=None,
                        force_download=False, subtract_thermo=False,
                        remove_h=False, create_pyg_graphs=False,
                        num_radials=1, device="cpu"):
    """
    Initialize datasets from .npz files.
    """
    from torch_pharma.data.datasets.utils import TORCH_PHARMA_HOME
    from torch_pharma.data.datasets.qm9 import QM9Dataset

    # Set the number of points based upon the arguments
    num_pts = {"train": getattr(args, "num_train", -1), 
               "test": getattr(args, "num_test", -1), 
               "valid": getattr(args, "num_valid", -1)}

    gdb9_dir = os.path.join(TORCH_PHARMA_HOME, "QM9")
    
    # Check if files exist, otherwise process
    splits_to_check = ["train", "valid", "test"]
    files_exist = all(os.path.exists(os.path.join(gdb9_dir, f"{split}.npz")) for split in splits_to_check)
    
    if not files_exist or force_download:
        log.info("Processing QM9 dataset...")
        qm9 = QM9Dataset()
        qm9.process()

    # Load datasets
    datasets = {}
    for split in splits_to_check:
        datafile = os.path.join(gdb9_dir, f"{split}.npz")
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(val) for key, val in f.items()}

    # Handle dataset halves if specified
    if dataset != "QM9":
        np.random.seed(42)
        fixed_perm = np.random.permutation(len(datasets["train"]["num_atoms"]))
        if dataset == "QM9_second_half":
            sliced_perm = fixed_perm[len(datasets["train"]["num_atoms"])//2:]
        elif dataset == "QM9_first_half":
            sliced_perm = fixed_perm[0:len(datasets["train"]["num_atoms"]) // 2]
        else:
            raise Exception(f"Unknown dataset name: {dataset}")
        for key in datasets["train"]:
            datasets["train"][key] = datasets["train"][key][sliced_perm]

    # Remove hydrogens if specified
    if remove_h:
        for key, split_data in datasets.items():
            pos = split_data["positions"]
            charges = split_data["charges"]
            num_atoms = split_data["num_atoms"]

            assert torch.sum(num_atoms != torch.sum(charges > 0, dim=1)) == 0

            mask = charges > 1
            new_positions = torch.zeros_like(pos)
            new_charges = torch.zeros_like(charges)
            for i in range(new_positions.shape[0]):
                m = mask[i]
                p = pos[i][m]
                if p.shape[0] > 0:
                    p = p - torch.mean(p, dim=0)
                c = charges[i][m]
                n = torch.sum(m)
                new_positions[i, :n, :] = p
                new_charges[i, :n] = c

            split_data["positions"] = new_positions
            split_data["charges"] = new_charges
            split_data["num_atoms"] = torch.sum(split_data["charges"] > 0, dim=1)

    all_species = _get_species(datasets)

    datasets = {
        split: ProcessedDataset(data,
                                num_pts=num_pts.get(split, -1),
                                included_species=all_species,
                                subtract_thermo=subtract_thermo,
                                create_pyg_graphs=create_pyg_graphs,
                                num_radials=num_radials,
                                device=device)
        for split, data in datasets.items()
    }

    num_species = datasets["train"].num_species
    max_charge = datasets["train"].max_charge

    # Update args if it's an object we can modify
    if hasattr(args, "num_train"): args.num_train = datasets["train"].num_pts
    if hasattr(args, "num_valid"): args.num_valid = datasets["valid"].num_pts
    if hasattr(args, "num_test"): args.num_test = datasets["test"].num_pts

    return args, datasets, num_species, max_charge


def _get_species(datasets, ignore_check=False):
    all_species = torch.cat([data["charges"].unique()
                             for data in datasets.values()]).unique(sorted=True)

    split_species = {split: data["charges"].unique(sorted=True) 
                     for split, data in datasets.items()}

    if all_species[0] == 0:
        all_species = all_species[1:]

    split_species = {split: species[1:] if species[0] == 0 else species 
                     for split, species in split_species.items()}

    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        if ignore_check:
            log.error("The number of species is not the same in all datasets!")
        else:
            raise ValueError("Not all datasets have the same number of species!")

    return all_species


def retrieve_dataloaders(dataloader_cfg: DictConfig):
    if "QM9" in dataloader_cfg.dataset:
        batch_size = dataloader_cfg.batch_size
        num_workers = dataloader_cfg.num_workers
        filter_n_atoms = dataloader_cfg.filter_n_atoms
        
        cfg_, datasets, _, charge_scale = initialize_datasets(
            dataloader_cfg,
            dataloader_cfg.data_dir,
            dataloader_cfg.dataset,
            subtract_thermo=dataloader_cfg.subtract_thermo,
            force_download=dataloader_cfg.force_download,
            remove_h=dataloader_cfg.remove_h,
            create_pyg_graphs=dataloader_cfg.create_pyg_graphs,
            num_radials=dataloader_cfg.num_radials,
            device=dataloader_cfg.device
        )
        
        qm9_to_eV = {
            "U0": 27.2114, "U": 27.2114, "G": 27.2114, "H": 27.2114,
            "zpve": 27211.4, "gap": 27.2114, "homo": 27.2114, "lumo": 27.2114
        }

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            log.info(f"Retrieving molecules with only {filter_n_atoms} atoms")
            datasets = filter_atoms(datasets, filter_n_atoms)

        preprocess = PreprocessQM9(load_charges=dataloader_cfg.include_charges)
        prefetch = 100 if num_workers > 0 else None
        dataloader_class = (
            partial(PyGDataLoader, prefetch_factor=prefetch, worker_init_fn=set_worker_sharing_strategy)
            if dataloader_cfg.create_pyg_graphs
            else partial(TorchDataLoader, collate_fn=preprocess.collate_fn)
        )
        
        dataloaders = {
            split: dataloader_class(dataset,
                                   num_workers=num_workers,
                                   batch_size=batch_size,
                                   shuffle=(split == "train"),
                                   drop_last=(split == "train"))
            for split, dataset in datasets.items()
        }
    else:
        raise ValueError(f"Unknown dataset {dataloader_cfg.dataset}")

    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data["num_atoms"] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data["one_hot"].size(0)
        datasets[key].perm = None
    return datasets
