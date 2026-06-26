"""Inference script for QM9 DDPM molecule generation (arXiv 2302.04313).

Loads a checkpoint produced by qm9_mol_gen_ddpm_train.py and samples new
molecules using the trained EquivariantVariationalDiffusion model.

Usage (unconditional with mean property context):
    python qm9_mol_gen_ddpm_inference.py \\
        --checkpoint checkpoints/model_0.pt \\
        --num_samples 100 \\
        --output generated.smi

Usage (conditional — samples alpha from QM9 training distribution):
    python qm9_mol_gen_ddpm_inference.py \\
        --checkpoint checkpoints/model_0.pt \\
        --num_samples 100 \\
        --load_data \\
        --output generated.smi
"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from rdkit import Chem

from torch_pharma.data.components.edm import get_bond_length_arrays
from torch_pharma.data.datasets.utils import QM9_WITH_H, QM9_WITHOUT_H
from torch_pharma.models.diffusion.variational_diffusion import EquivariantVariationalDiffusion
from torch_pharma.models.gnn.egnn import EGNNDynamics
from torch_pharma.models.gnn.gcpnet import GCPNetDynamics
from torch_pharma.molecules.chemistry import build_molecule, process_molecule
from torch_pharma.utils.logging import get_pylogger, setup_logging
from torch_pharma.utils.math import batch_tensor_to_list

log = get_pylogger(__name__)


class QM9DDPMInference(nn.Module):
    """Thin inference wrapper around EquivariantVariationalDiffusion.

    Mirrors the model structure saved by qm9_mol_gen_ddpm_train.py so that
    ``load_state_dict(strict=False)`` correctly restores the DDPM weights.
    """

    def __init__(
        self,
        dynamics_network_type: str = "gcpnet",
        remove_h: bool = True,
        num_timesteps: int = 1000,
        loss_type: str = "l2",
        include_charges: bool = True,
        num_x_dims: int = 3,
        conditioning: List[str] = None,
    ):
        super().__init__()
        conditioning = conditioning if conditioning is not None else ["alpha"]

        num_atom_types = 4 if remove_h else 5
        dataset_info = QM9_WITHOUT_H if remove_h else QM9_WITH_H

        dynamics_cls = {"gcpnet": GCPNetDynamics, "egnn": EGNNDynamics}[dynamics_network_type]
        # Use the same kwargs that qm9_mol_gen_ddpm_train.py passes — no conditioning arg
        # is forwarded to the dynamics network; it relies on the default conditioning=["alpha"].
        dynamics = dynamics_cls(
            num_atom_types=num_atom_types,
            include_charges=include_charges,
            num_x_dims=num_x_dims,
        )

        self.ddpm = EquivariantVariationalDiffusion(
            dynamics_network=dynamics,
            dataset_info=dataset_info,
            num_atom_types=num_atom_types,
            num_x_dims=num_x_dims,
            num_timesteps=num_timesteps,
            loss_type=loss_type,
            include_charges=include_charges,
        )

        # Ensure bond-length look-up tables are populated in dataset_info
        if not dataset_info.get("bonds1"):
            bonds = get_bond_length_arrays(dataset_info["atom_encoder"])
            dataset_info["bonds1"], dataset_info["bonds2"], dataset_info["bonds3"] = bonds

        self.dataset_info = dataset_info
        self.num_x_dims = num_x_dims
        self.include_charges = include_charges
        self.num_atom_types = num_atom_types
        self.condition_on_context = len(conditioning) > 0
        self.num_context_features = len(conditioning)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        num_samples: int,
        batch_size: int = 64,
        num_timesteps: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        sanitize: bool = False,
        relax_iter: int = 0,
        largest_frag: bool = False,
        add_hydrogens: bool = False,
        device: Optional[torch.device] = None,
    ) -> List[Chem.Mol]:
        """Generate ``num_samples`` molecules from the trained DDPM.

        Args:
            num_samples: Number of molecules to generate.
            batch_size: How many molecules to sample in one forward pass.
            num_timesteps: Denoising steps; uses training value when None.
            context: Per-sample context tensor ``(num_samples, num_context_features)``.
                     If the model was trained with conditioning and this is None,
                     zeros are used (equivalent to conditioning on the mean property).
            sanitize: Run RDKit sanitization on generated molecules.
            relax_iter: Number of MMFF force-field optimization steps (0 = skip).
            largest_frag: Keep only the largest connected fragment.
            add_hydrogens: Explicit hydrogens in output molecules.
            device: Device to run inference on.
        Returns:
            List of valid RDKit Mol objects.
        """
        device = device or next(self.parameters()).device
        molecules: List[Chem.Mol] = []

        for start in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - start)

            num_nodes = self.ddpm.num_nodes_distribution.sample(n)

            # Build per-node context: zeros = mean normalized property value
            ctx = None
            if self.condition_on_context:
                if context is not None:
                    ctx = context[start : start + n].to(device)
                else:
                    # Use zero-context (mean property in normalized space)
                    ctx = torch.zeros(n, self.num_context_features, device=device)

            xh, batch_index, _ = self.ddpm.mol_gen_sample(
                num_samples=n,
                num_nodes=num_nodes,
                context=ctx,
                device=device,
                num_timesteps=num_timesteps,
            )

            x = xh[:, : self.num_x_dims].detach().cpu()
            atom_types = (
                xh[:, self.num_x_dims : -1].argmax(-1).detach().cpu()
                if self.include_charges
                else xh[:, self.num_x_dims :].argmax(-1).detach().cpu()
            )

            for pos, atype in zip(
                batch_tensor_to_list(x, batch_index.cpu()),
                batch_tensor_to_list(atom_types, batch_index.cpu()),
            ):
                mol = build_molecule(pos, atype, dataset_info=self.dataset_info, add_coords=True)
                mol = process_molecule(
                    rdmol=mol,
                    add_hydrogens=add_hydrogens,
                    sanitize=sanitize,
                    relax_iter=relax_iter,
                    largest_frag=largest_frag,
                )
                if mol is not None:
                    molecules.append(mol)

        return molecules


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="QM9 DDPM molecule generation inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_0.pt",
        help="Path to .pt checkpoint saved by qm9_mol_gen_ddpm_train.py",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=None,
        help="Denoising steps (default: value used during training)",
    )
    parser.add_argument(
        "--dynamics_network",
        type=str,
        default="gcpnet",
        choices=["gcpnet", "egnn"],
    )
    parser.add_argument(
        "--no_remove_h",
        action="store_true",
        default=False,
        help="Include hydrogen atoms (default: remove H, matching training defaults)",
    )
    parser.add_argument("--no_charges", action="store_true", default=False)
    parser.add_argument(
        "--load_data",
        action="store_true",
        default=False,
        help="Load the QM9 training set to sample conditioning context from the "
        "real property distribution. Recommended for conditional models.",
    )
    parser.add_argument("--sanitize", action="store_true", default=False)
    parser.add_argument("--relax_iter", type=int, default=0)
    parser.add_argument("--largest_frag", action="store_true", default=False)
    parser.add_argument("--add_hydrogens", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="generated_molecules.smi")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level, run_name="ddpm_inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    remove_h = not args.no_remove_h
    include_charges = not args.no_charges

    # ── Build model ─────────────────────────────────────────────────────────────
    model = QM9DDPMInference(
        dynamics_network_type=args.dynamics_network,
        remove_h=remove_h,
        include_charges=include_charges,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(
            "%d keys missing from checkpoint (expected for inference wrappers): %s ...",
            len(missing),
            missing[:3],
        )
    log.info(
        "Loaded checkpoint: %s  (missing=%d, unexpected=%d)",
        args.checkpoint,
        len(missing),
        len(unexpected),
    )

    model = model.to(device)
    model.eval()

    # ── Optional: sample context from QM9 training distribution ─────────────────
    context = None
    if args.load_data and model.condition_on_context:
        log.info("Loading QM9 dataloaders to sample conditioning context …")
        from omegaconf import OmegaConf

        from torch_pharma.data.components.edm import retrieve_dataloaders
        from torch_pharma.data.datasets.utils import TORCH_PHARMA_HOME
        from torch_pharma.models.transformers import PropertiesDistribution, compute_mean_mad

        dataloader_cfg = OmegaConf.create(
            {
                "dataset": "QM9",
                "batch_size": 64,
                "num_workers": 2,
                "filter_n_atoms": None,
                "data_dir": str(TORCH_PHARMA_HOME),
                "subtract_thermo": True,
                "force_download": False,
                "remove_h": remove_h,
                "create_pyg_graphs": True,
                "num_radials": 1,
                "device": "cpu",
                "include_charges": include_charges,
            }
        )
        dataloaders, _ = retrieve_dataloaders(dataloader_cfg)
        props_norms = compute_mean_mad(dataloaders, ["alpha"], "QM9")
        props_distr = PropertiesDistribution(
            dataloaders["train"], ["alpha"], device=device
        )
        props_distr.set_normalizer(props_norms)

        # Pre-sample enough context for all molecules
        num_nodes_all = model.ddpm.num_nodes_distribution.sample(args.num_samples)
        context = props_distr.sample_batch(num_nodes_all)
        log.info("Context sampled from QM9 alpha distribution: shape=%s", context.shape)

    # ── Generate ────────────────────────────────────────────────────────────────
    log.info("Generating %d molecules …", args.num_samples)
    molecules = model.generate(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_timesteps=args.num_timesteps,
        context=context,
        sanitize=args.sanitize,
        relax_iter=args.relax_iter,
        largest_frag=args.largest_frag,
        add_hydrogens=args.add_hydrogens,
        device=device,
    )

    smiles = [Chem.MolToSmiles(mol) for mol in molecules]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(smiles))
    log.info(
        "Generated %d valid molecules (%.1f%% valid) → %s",
        len(smiles),
        100.0 * len(smiles) / max(args.num_samples, 1),
        out_path,
    )


if __name__ == "__main__":
    main()
