import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from torch_geometric.data import Batch
from torch_scatter import scatter

from torch_pharma.models.distributions import CategoricalDistribution, NumNodesDistribution, PropertiesDistribution
from torch_pharma.models.utils import Queue, get_grad_norm, batch_tensor_to_list, reverse_tensor, inflate_batch_array
from torch_pharma.utils.logging import get_pylogger
from torch_pharma.data.components.edm import check_molecular_stability
from torch_pharma.data.components.edm.rdkit_utils import BasicMolecularMetrics, build_molecule
from torch_pharma.models.diffusion.variational_diffusion import EquivariantVariationalDiffusion

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

log = get_pylogger(__name__)

class MoleculeDDPM(nn.Module):
    """
    DDPM model for generating 3D molecules.
    Refactored from bio-diffusion's QM9MoleculeGenerationDDPM.
    """
    def __init__(
        self,
        dynamics_network: nn.Module,
        diffusion_cfg: Dict[str, Any],
        dataloader_cfg: Dict[str, Any],
        dataset_info: Dict[str, Any],
        conditioning: List[str] = [],
        device: Union[torch.device, str] = "cpu"
    ):
        super().__init__()
        self.device = device
        self.dataset_info = dataset_info
        self.conditioning = conditioning
        self.condition_on_context = len(conditioning) > 0
        
        self.T = diffusion_cfg.get("num_timesteps", 1000)
        self.num_atom_types = dataloader_cfg.get("num_atom_types", 5)
        self.num_x_dims = dataloader_cfg.get("num_x_dims", 3)
        self.include_charges = dataloader_cfg.get("include_charges", False)
        
        # Diffusion model
        self.ddpm = EquivariantVariationalDiffusion(
            dynamics_network=dynamics_network,
            diffusion_cfg=diffusion_cfg,
            dataloader_cfg=dataloader_cfg,
            dataset_info=dataset_info
        )
        
        # Distributions (to be initialized via set_distributions)
        self.node_type_distribution = None
        self.num_nodes_distribution = None
        self.props_distr = None
        self.props_norms = None

    def set_distributions(
        self, 
        node_type_hist: Dict[int, int], 
        num_nodes_hist: Dict[int, int],
        props_distr: Optional[PropertiesDistribution] = None,
        props_norms: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ):
        self.node_type_distribution = CategoricalDistribution(
            node_type_hist, self.dataset_info["atom_encoder"]
        )
        self.num_nodes_distribution = NumNodesDistribution(num_nodes_hist)
        self.props_distr = props_distr
        self.props_norms = props_norms

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # This mirrors the forward pass logic in the original LightningModule
        # but expects the batch to be pre-processed if necessary.
        nll, loss_info = self.ddpm(batch, return_loss_info=True)
        return nll, loss_info

    @torch.inference_mode()
    @typechecked
    def sample(
        self,
        num_samples: int,
        num_nodes: Optional[TensorType["batch_size"]] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        fix_noise: bool = False,
        num_timesteps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_nodes is None:
            if self.num_nodes_distribution is None:
                raise ValueError("num_nodes_distribution must be set or num_nodes provided.")
            num_nodes = self.num_nodes_distribution.sample(num_samples)

        if self.condition_on_context and context is None:
            if self.props_distr is None:
                raise ValueError("props_distr must be set or context provided.")
            context = self.props_distr.sample_batch(num_nodes)

        xh, batch_index, _ = self.ddpm.mol_gen_sample(
            num_samples=num_samples,
            num_nodes=num_nodes,
            node_mask=node_mask,
            context=context,
            fix_noise=fix_noise,
            device=self.device,
            num_timesteps=num_timesteps
        )

        x = xh[:, :self.num_x_dims]
        one_hot = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
        charges = xh[:, -1:] if self.include_charges else torch.zeros(0, device=self.device)

        return x, one_hot, charges, batch_index
