import os
import glob
import random
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from rdkit import Chem
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

@typechecked
def save_xyz_file(
    path: str,
    positions: TensorType["batch_num_nodes", 3],
    one_hot: TensorType["batch_num_nodes", "num_atom_types"],
    charges: torch.Tensor,  # TODO: incorporate charges within saved XYZ file
    dataset_info: Dict[str, Any],
    id_from: int = 0,
    name: str = "molecule",
    batch_index: Optional[TensorType["batch_num_nodes"]] = None
):
    try:
        os.makedirs(path)
    except OSError:
        pass

    if batch_index is None:
        batch_index = torch.zeros(len(one_hot))

    for batch_i in torch.unique(batch_index):
        current_batch_index = (batch_index == batch_i)
        num_atoms = int(torch.sum(current_batch_index).item())
        f = open(os.path.join(path, name + "_" + "%03d.xyz" % (batch_i + id_from)), "w")
        f.write("%d\n\n" % num_atoms)
        atoms = torch.argmax(one_hot[current_batch_index], dim=-1)
        batch_pos = positions[current_batch_index]
        for atom_i in range(num_atoms):
            atom = atoms[atom_i]
            atom = dataset_info["atom_decoder"][atom]
            f.write("%s %.9f %.9f %.9f\n" % (atom, batch_pos[atom_i, 0], batch_pos[atom_i, 1], batch_pos[atom_i, 2]))
        f.close()


@typechecked
def write_xyz_file(
    positions: TensorType["num_nodes", 3],
    atom_types: TensorType["num_nodes"],
    filename: str
):
    out = f"{len(positions)}\n\n"
    assert len(positions) == len(atom_types)
    for i in range(len(positions)):
        out += f"{atom_types[i]} {positions[i, 0]:.3f} {positions[i, 1]:.3f} {positions[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


@typechecked
def write_sdf_file(sdf_path: Path, molecules: List[Chem.Mol], verbose: bool = True):
    from torch_pharma.utils.logging import get_pylogger
    log = get_pylogger(__name__)
    
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if m is not None:
            w.write(m)
    if verbose:
        log.info(f"Wrote generated molecules to SDF file {sdf_path}")


@typechecked
def load_molecule_xyz(
    file: str,
    dataset_info: Dict[str, Any]
) -> Tuple[
    TensorType["num_nodes", 3],
    TensorType["num_nodes", "num_atom_types"]
]:
    with open(file, encoding="utf8") as f:
        num_atoms = int(f.readline())
        one_hot = torch.zeros(num_atoms, len(dataset_info["atom_decoder"]))
        positions = torch.zeros(num_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(num_atoms):
            atom = atoms[i].split(" ")
            atom_type = atom[0]
            one_hot[i, dataset_info["atom_encoder"][atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot


@typechecked
def load_files_with_ext(path: str, ext: str, shuffle: bool = True) -> List[str]:
    files = glob.glob(os.path.join(path, f"*.{ext}"))
    if shuffle:
        random.shuffle(files)
    return files


@typechecked
def num_nodes_to_batch_index(
    num_samples: int,
    num_nodes: Union[int, TensorType["batch_size"]],
    device: Union[torch.device, str]
) -> TensorType["batch_num_nodes"]:
    assert isinstance(num_nodes, int) or len(num_nodes) == num_samples
    sample_inds = torch.arange(num_samples, device=device)
    return torch.repeat_interleave(sample_inds, num_nodes)
