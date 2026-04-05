# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

from torch_pharma.utils.io import num_nodes_to_batch_index, save_xyz_file
print("io imported", flush=True)

print("SCRIPT STARTING...", flush=True)
import math
print("math imported", flush=True)
import os
print("os imported", flush=True)
import torch
print("torch imported", flush=True)
import torchmetrics
print("torchmetrics imported", flush=True)


import numpy as np
print("numpy imported", flush=True)
import torch.nn.functional as F
print("F imported", flush=True)

from torch import nn
print("nn imported", flush=True)
from time import time, strftime
from pathlib import Path
from rdkit import Chem
print("rdkit imported", flush=True)

from torch_geometric.data import Batch
print("torch_geometric imported", flush=True)
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from omegaconf import DictConfig
from torch_scatter import scatter
print("torch_scatter imported", flush=True)

try:
    from . import utils as qm9utils
except (ImportError, ValueError):
    import utils as qm9utils
print("utils imported", flush=True)

from torch_pharma.utils.io import num_nodes_to_batch_index, save_xyz_file
print("io imported", flush=True)
from torch_pharma.features.geometry import centralize
print("geometry imported", flush=True)
from torch_pharma.utils.visualize import visualize_mol, visualize_mol_chain
print("visualize imported", flush=True)
from torch_pharma.utils.logging import rank_zero_debug, rank_zero_info, rank_zero_warn, rank_zero_only
print("logging imported", flush=True)

from torch_pharma.molecules.featurizers import BasicMolecularMetrics
print("featurizers imported", flush=True)
from torch_pharma.molecules.chemistry import build_molecule, process_molecule
print("chemistry imported", flush=True)
from torch_pharma.data.datasets.utils import QM9_SECOND_HALF, QM9_WITH_H, QM9_WITHOUT_H
print("datasets utils imported", flush=True)
from torch_pharma.data.components.edm import check_molecular_stability, get_bond_length_arrays, retrieve_dataloaders
print("edm imported", flush=True)
from torch_pharma.models.dynamics.egnn import EGNNDynamics
print("egnn imported", flush=True)
from torch_pharma.models.diffusion.variational_diffusion import EquivariantVariationalDiffusion
print("diffusion imported", flush=True)

from torch_pharma.models.dynamics.gcpnet import GCPNetDynamics
print("gcpnet imported", flush=True)
from torch_pharma.models.transformers import HALT_FILE_EXTENSION, CategoricalDistribution, PropertiesDistribution, Queue, compute_mean_mad, get_grad_norm, log_grad_flow_lite
print("transformers imported", flush=True)
from torch_pharma.utils.math import batch_tensor_to_list, reverse_tensor
print("math utils imported", flush=True)
from torch_pharma.data.datasets.utils import TORCH_PHARMA_HOME
print("dataset utils again imported", flush=True)

from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
print("typeguard/torchtyping imported", flush=True)


patch_typeguard()  # use before @typechecked

from torch_pharma.utils.tracking import track_gnn_activations
from torch_pharma.utils.tracking.loggers import WandbActivationLogger, MlflowActivationLogger




@track_gnn_activations(
            track_layers=True,
            track_nodes=True,
            track_edges=True,
            verbose=False,
            # Filter specifically to core graph messaging layers to avoid OOM or API request timeouts
            layer_filter=lambda name, mod: "interaction" in name.lower() or "conv" in name.lower(),
            loggers=[
                WandbActivationLogger(
                    prefix="QM9MoleculeGenerationDDPM", 
                    log_raw_tensors=False, # True would upload HUNDREDS of histograms per batch, slowing down tracking
                    project="torch-pharma-QM9MoleculeGenerationDDPM", 
                    name="demo-run"
                ),
                MlflowActivationLogger(
                    prefix="QM9MoleculeGenerationDDPM", 
                    tracking_uri="http://localhost:5000", # Route data to explicit loc
                    experiment_name="torch-pharma-QM9MoleculeGenerationDDPM", 
                    run_name="demo-run"
                )
            ]
        )
class QM9MoleculeGenerationDDPM(nn.Module):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_eval_samples = 1000,
        conditioning = ['alpha'],
        remove_h = True,
        ddpm_mode = "unconditional",
        dynamics_network = "gcpnet",
        num_timesteps = 1000,
        loss_type = "l2",
        num_atom_types = 5,
        num_x_dims = 3,
        include_charges = True,
        dataset = "QM9",
        clip_gradients = True,
        smiles_filepath = f"{TORCH_PHARMA_HOME}/QM9/QM9_smiles.pickle",
        **kwargs
    ):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.conditioning = conditioning
        self.num_eval_samples = num_eval_samples
        self.remove_h = remove_h
        self.dynamics_network_type = dynamics_network
        self.num_timesteps = num_timesteps
        self.dataset_name = dataset
        self.clip_gradients = clip_gradients
        self.smiles_filepath = smiles_filepath
        self.kwargs = kwargs

        # placeholders for state usually handled by Lightning
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_output_dir: Optional[Path] = None
        self.props_distr: Optional[PropertiesDistribution] = None
        self.props_norms: Optional[Dict[str, Any]] = None
        self.ddpm: nn.Module

        num_eval_samples = (
            num_eval_samples // 2 if len(conditioning) > 0 else num_eval_samples
        )

        num_atom_types = (
            num_atom_types - 1
            if remove_h
            else num_atom_types
        )

        ddpm_modes = {
            "unconditional": EquivariantVariationalDiffusion,
            "inpainting": EquivariantVariationalDiffusion
        }

        self.ddpm_mode = ddpm_mode
        assert self.ddpm_mode in ddpm_modes, f"Selected DDPM mode {self.ddpm_mode} is currently not supported."

        dynamics_networks = {
            "gcpnet": GCPNetDynamics,
            "egnn": EGNNDynamics
        }

        assert dynamics_network in dynamics_networks, f"Selected dynamics network {dynamics_network} is currently not supported."

        self.T = num_timesteps
        self.loss_type = loss_type
        self.num_atom_types = num_atom_types
        self.num_x_dims = num_x_dims
        self.include_charges = include_charges
        self.condition_on_context = len(conditioning) > 0

        dataset_info_mapping = {
            "QM9": QM9_WITHOUT_H if remove_h else QM9_WITH_H,
            "QM9_second_half": QM9_SECOND_HALF
        }

        self.dataset_info = dataset_info_mapping[dataset]

        if dataset == "QM9_second_half" and remove_h:
            raise NotImplementedError(f"Missing config for dataset {dataset} without hydrogen atoms")


        ## Look into passing parameters here
        # dynamics network instantiation
        dynamics_kwargs = {
            "num_atom_types": num_atom_types,
            "include_charges": include_charges,
            "num_x_dims": num_x_dims,
        }
        if self.dynamics_network_type == "gcpnet":
            # GCPNetDynamics specific kwargs could go here if needed
            pass
        
        dynamics_network = dynamics_networks[self.dynamics_network_type](**dynamics_kwargs)

        self.ddpm = ddpm_modes[self.ddpm_mode](
            dynamics_network = dynamics_network,
            dataset_info = self.dataset_info,
            num_atom_types = num_atom_types,
            num_x_dims = num_x_dims,
            num_timesteps = num_timesteps,
            loss_type = loss_type,
            include_charges = include_charges,
            **kwargs.get("ddpm", {})
        )

        # initialize distributions #
        self.node_type_distribution = CategoricalDistribution(
            self.dataset_info["atom_types"],
            self.dataset_info["atom_encoder"]
        )

        # training #
        if clip_gradients:
            self.gradnorm_queue = Queue()
            self.gradnorm_queue.add(3000)  # add large value that will be flushed

        # metrics #
        self.train_phase, self.val_phase, self.test_phase = "train", "val", "test"
        self.phases = [self.train_phase, self.val_phase, self.test_phase]
        self.metrics_to_monitor = [
            "loss", "loss_t", "SNR_weight", "loss_0",
            "kl_prior", "delta_log_px", "neg_log_const_0", "log_pN",
            "eps_hat_x", "eps_hat_h"
        ]
        self.eval_metrics_to_monitor = self.metrics_to_monitor + ["log_SNR_max", "log_SNR_min"]
        for phase in self.phases:
            metrics_to_monitor = (
                self.metrics_to_monitor
                if phase == self.train_phase
                else self.eval_metrics_to_monitor
            )
            for metric in metrics_to_monitor:
                # note: individual metrics e.g., for averaging loss across batches
                setattr(self, f"{phase}_{metric}", torchmetrics.MeanMetric())

        # sample metrics
        if smiles_filepath and os.path.exists(smiles_filepath):
            import pickle
            with open(smiles_filepath, "rb") as f:
                smiles_list = pickle.load(f)
        else:
            smiles_list = None

        self.molecular_metrics = BasicMolecularMetrics(
            self.dataset_info,
            data_dir=str(TORCH_PHARMA_HOME),
            dataset_smiles_list=smiles_list
        )

        # placeholder for dynamically derived properties
        self.props_norms: Optional[Dict[str, Any]] = None
        self.props_distr: Optional[PropertiesDistribution] = None
        self.num_context_node_feats: Optional[int] = None
        self.sampling_output_dir: Optional[Path] = None

        self.on_train_start()

    @typechecked
    def forward(
        self,
        batch: Batch,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[
        torch.Tensor,
        Dict[str, Any]
    ]:
        """
        Compute the loss (type L2 or negative log-likelihood (NLL)) if `training`.
        If `eval`, then always compute NLL.
        """

        # centralize node positions to make them translation-invariant
        _, batch.x = centralize(
            batch,
            key="x",
            batch_index=batch.batch,
            node_mask=batch.mask,
            edm=True
        )

        # construct invariant node features
        batch.h = {"categorical": batch.one_hot, "integer": batch.charges}

        # derive property contexts (i.e., conditionals)
        if self.condition_on_context:
            batch.props_context = qm9utils.prepare_context(
                list(self.conditioning),
                batch,
                self.props_norms
            ).type(dtype)
        else:
            batch.props_context = None

        # derive node counts per batch
        num_nodes = scatter(batch.mask.int(), batch.batch, dim=0, reduce="sum")
        batch.num_nodes_present = num_nodes

        # note: `L` terms in e.g., the GCDM paper represent log-likelihoods,
        # while our loss terms are negative (!) log-likelihoods
        (
            delta_log_px, error_t, SNR_weight,
            loss_0_x, loss_0_h, neg_log_const_0,
            kl_prior, log_pN, t_int, loss_info
        ) = self.ddpm(batch, return_loss_info=True)

        # support L2 loss training step
        if self.training and self.loss_type == "l2":
            # normalize `loss_t`
            effective_num_nodes = (
                num_nodes.max()
                if self.kwargs.get("diffusion_cfg", {}).get("norm_training_by_max_nodes", False)
                else num_nodes
            )
            denom = (self.num_x_dims + self.ddpm.num_node_scalar_features) * effective_num_nodes
            error_t = error_t / denom
            loss_t = 0.5 * error_t

            # normalize `loss_0` via `loss_0_x` normalization
            loss_0_x = loss_0_x / denom
            loss_0 = loss_0_x + loss_0_h
        
        # support VLB objective or evaluation step
        else:
            loss_t = self.T * 0.5 * SNR_weight * error_t
            loss_0 = loss_0_x + loss_0_h
            loss_0 = loss_0 + neg_log_const_0

        # combine loss terms
        nll = loss_t + loss_0 + kl_prior

        # correct for normalization on `x`
        nll = nll - delta_log_px

        # transform conditional `nll` into joint `nll`
        # note: loss = -log p(x,h|N) and log p(x,h,N) = log p(x,h|N) + log p(N);
        # therefore, log p(x,h,N) = -loss + log p(N)
        # => loss_new = -log p(x,h,N) = loss - log p(N)
        nll = nll - log_pN

        # collect all metrics' batch-averaged values
        local_variables = locals()
        for metric in self.metrics_to_monitor:
            if metric in ["eps_hat_x", "eps_hat_h"]:
                continue
            if metric != "loss":
                loss_info[metric] = local_variables[metric].mean(0)

        return nll, loss_info

    def step(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # make a forward pass and score it
        nll, loss_info = self.forward(batch)
        return nll, loss_info

    def on_train_start(self, dtype: torch.dtype = torch.float32):
        # note: by default, Lightning executes validation step sanity checks before training starts,
        # so we need to make sure that val_`metric` doesn't store any values from these checks
        for metric in self.eval_metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.val_phase}_{metric}")
            torchmetric.reset()

        # ensure valid bond lengths have been added to each dataset's metadata collection (i.e., `self.dataset_info`)
        if any([
            not getattr(self.dataset_info, "bonds1", None),
            not getattr(self.dataset_info, "bonds2", None),
            not getattr(self.dataset_info, "bonds3", None)
        ]):
            bonds = get_bond_length_arrays(self.dataset_info["atom_encoder"])
            self.dataset_info["bonds1"], self.dataset_info["bonds2"], self.dataset_info["bonds3"] = (
                bonds[0], bonds[1], bonds[2]
            )

        # ensure directory for storing sampling outputs is defined
        if self.sampling_output_dir is None:
            self.sampling_output_dir = Path("sampling_output")
            self.sampling_output_dir.mkdir(exist_ok=True)

        # if not already loaded, derive distribution information
        # regarding node counts and possibly properties for model conditioning
        if self.condition_on_context and self.props_distr is None:
            print("On train start: Computing mean and mad from dataloaders")
            dataloaders = self.kwargs.get("dataloaders")
            if dataloaders is not None:
                self.props_norms = compute_mean_mad(
                    dataloaders, list(self.conditioning), self.dataset_name
                )
                self.props_distr = PropertiesDistribution(
                    dataloaders.get("train"), self.conditioning, device=self.device
                )
                if self.props_distr is not None:
                    self.props_distr.set_normalizer(self.props_norms)

            # derive number of property context features (i.e., conditionals)
            if self.condition_on_context and self.props_norms is not None:
                dummy_data = next(iter(dataloaders["train"]))
                dummy_props_context = qm9utils.prepare_context(
                    list(self.conditioning),
                    dummy_data,
                    self.props_norms
                ).type(dtype)
                self.num_context_node_feats = dummy_props_context.shape[-1]
            else:
                self.num_context_node_feats = None

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        self.global_step += 1
        
        try:
            nll, metrics_dict = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise(e)
            torch.cuda.empty_cache()
            rank_zero_info(f"Skipping training batch with index {batch_idx} due to OOM error...")
            return

        # ensure all intermediate losses to be logged as metrics have their gradients ignored
        metrics_dict = {key: value.detach() for key, value in metrics_dict.items()}

        # calculate standard NLL from forward KL-divergence while preserving its gradients
        metrics_dict["loss"] = nll.mean(0)

        # update metrics
        for metric in self.metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.train_phase}_{metric}")
            torchmetric(metrics_dict[metric])

        return metrics_dict

    def on_train_epoch_end(self):
        # log metrics
        for metric in self.metrics_to_monitor:
            # e.g., logging loss that has been averaged across batches
            torchmetric = getattr(self, f"{self.train_phase}_{metric}")
            print(f"{self.train_phase}/{metric}: {torchmetric.compute()}")

    def on_validation_start(self, dtype: torch.dtype = torch.float32):
        # ensure directory for storing sampling outputs is defined
        if self.sampling_output_dir is None:
            sampling_output_dir = Path("sampling_output")
            sampling_output_dir.mkdir(exist_ok=True)
            self.sampling_output_dir = sampling_output_dir

        # ensure valid bond lengths have been added to each dataset's metadata collection (i.e., `self.dataset_info`)
        if any([
            not getattr(self.dataset_info, "bonds1", None),
            not getattr(self.dataset_info, "bonds2", None),
            not getattr(self.dataset_info, "bonds3", None)
        ]):
            bonds = get_bond_length_arrays(self.dataset_info["atom_encoder"])
            self.dataset_info["bonds1"], self.dataset_info["bonds2"], self.dataset_info["bonds3"] = (
                bonds[0], bonds[1], bonds[2]
            )

        # if not already loaded, derive distribution information
        # regarding node counts and possibly properties for model conditioning
        if self.condition_on_context and not getattr(self, "props_distr", None):
            dataloaders = self.kwargs.get("dataloaders")
            if dataloaders is not None:
                self.props_norms = compute_mean_mad(
                    dataloaders, list(self.conditioning), self.dataset_name
                )
                self.props_distr = PropertiesDistribution(
                    dataloaders.get("train"), self.conditioning, device=self.device
                )
                if self.props_distr is not None:
                    self.props_distr.set_normalizer(self.props_norms)

            # derive number of property context features (i.e., conditionals)
            if self.condition_on_context and self.props_norms is not None:
                dataloaders = self.kwargs.get("dataloaders")
                if dataloaders is not None and "valid" in dataloaders:
                    dummy_data = next(iter(dataloaders["valid"]))
                    dummy_props_context = qm9utils.prepare_context(
                        list(self.conditioning),
                        dummy_data,
                        self.props_norms
                    ).type(dtype)
                    self.num_context_node_feats = dummy_props_context.shape[-1]
                else:
                    self.num_context_node_feats = None
            else:
                self.num_context_node_feats = None

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        try:
            nll, metrics_dict = self.step(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise(e)
            torch.cuda.empty_cache()
            rank_zero_info(f"Skipping validation batch with index {batch_idx} due to OOM error...")
            return

        # ensure all intermediate losses to be logged as metrics have their gradients ignored
        metrics_dict = {key: value.detach() for key, value in metrics_dict.items()}

        # calculate standard NLL from forward KL-divergence while preserving its gradients
        metrics_dict["loss"] = nll.mean(0)

        # collect additional loss information
        gamma_0 = self.ddpm.gamma(torch.zeros((1, 1), device=self.device)).squeeze()
        gamma_1 = self.ddpm.gamma(torch.ones((1, 1), device=self.device)).squeeze()
        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1
        metrics_dict["log_SNR_max"] = log_SNR_max
        metrics_dict["log_SNR_min"] = log_SNR_min

        # update metrics
        for metric in self.eval_metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.val_phase}_{metric}")
            torchmetric(metrics_dict[metric])

        return metrics_dict

    def log_evaluation_metrics(
        self,
        metrics_dict: Dict[str, Any],
        phase: str,
        batch_size: Optional[int] = None,
        sync_dist: bool = False,
        **kwargs
    ):
        for m, value in metrics_dict.items():
            print(f"{phase}/{m}: {value}")

    @rank_zero_only
    def evaluate_sampling(self):
        suffix_mapping = {
            "unconditional": "",
            "conditional": "_conditional",
            "simple_conditional": "_simple_conditional"
        }
        suffix = suffix_mapping[self.ddpm_mode]
        diffusion_cfg = self.kwargs.get("diffusion_cfg", {})

        if (self.current_epoch + 1) % diffusion_cfg.get("eval_epochs", 10) == 0:
            ticker = time()

            sampler = getattr(self, "sample_and_analyze" + suffix)
            sampling_results = sampler(
                num_samples=diffusion_cfg.get("num_eval_samples", 1000),
                batch_size=diffusion_cfg.get("eval_batch_size", 64)
            )
            self.log_evaluation_metrics(sampling_results, phase="val")

            rank_zero_info(f"on_validation_epoch_end(): Sampling evaluation took {time() - ticker:.2f} seconds")

        if (self.current_epoch + 1) % diffusion_cfg.get("visualize_sample_epochs", 10) == 0:
            ticker = time()
            sampler = getattr(self, "sample_and_save" + suffix)
            sampler(num_samples=diffusion_cfg.get("num_visualization_samples", 10))
            rank_zero_info(f"on_validation_epoch_end(): Sampling visualization took {time() - ticker:.2f} seconds")

        if (self.current_epoch + 1) % diffusion_cfg.get("visualize_chain_epochs", 10) == 0:
            ticker = time()
            sampler = getattr(self, "sample_chain_and_save" + suffix)
            sampler(keep_frames=diffusion_cfg.get("keep_frames", 100))
            rank_zero_info(f"on_validation_epoch_end(): Chain visualization took {time() - ticker:.2f} seconds")

    def on_validation_epoch_end(self):
        # log metrics
        for metric in self.eval_metrics_to_monitor:
            # e.g., logging loss that has been averaged across batches
            torchmetric = getattr(self, f"{self.val_phase}_{metric}")
            print(f"{self.val_phase}/{metric}: {torchmetric.compute()}")

        # make a backup checkpoint before (potentially) sampling from the model
        # NOTE: Lightning save_checkpoint removed. Implement manual saving if needed.
        pass

        # perform sampling evaluation on the first device (i.e., rank zero) only
        diffusion_cfg = self.kwargs.get("diffusion_cfg", {})
        intervals = [
            diffusion_cfg.get("eval_epochs", 10),
            diffusion_cfg.get("visualize_sample_epochs", 10),
            diffusion_cfg.get("visualize_chain_epochs", 10)
        ]
        time_to_evaluate_sampling = (
            diffusion_cfg.get("sample_during_training", False)
            and any([(self.current_epoch + 1) % interval == 0 for interval in intervals])
        )
        if time_to_evaluate_sampling:
            self.evaluate_sampling()

    def test_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        nll, metrics_dict = self.step(batch)

        # ensure all intermediate losses to be logged as metrics have their gradients ignored
        metrics_dict = {key: value.detach() for key, value in metrics_dict.items()}

        # calculate standard NLL from forward KL-divergence while preserving its gradients
        metrics_dict["loss"] = nll.mean(0)

        # collect additional loss information
        gamma_0 = self.ddpm.gamma(torch.zeros((1, 1), device=self.device)).squeeze()
        gamma_1 = self.ddpm.gamma(torch.ones((1, 1), device=self.device)).squeeze()
        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1
        metrics_dict["log_SNR_max"] = log_SNR_max
        metrics_dict["log_SNR_min"] = log_SNR_min

        # update metrics
        for metric in self.eval_metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.test_phase}_{metric}")
            torchmetric(metrics_dict[metric])

        return metrics_dict

    def on_test_epoch_end(self):
        # log metrics
        for metric in self.eval_metrics_to_monitor:
            # e.g., logging loss that has been averaged across batches
            torchmetric = getattr(self, f"{self.test_phase}_{metric}")
            print(f"{self.test_phase}/{metric}: {torchmetric.compute()}")

    def on_after_backward(self) -> None:
        # periodically log gradient flow
        if (
            (self.global_step + 1) % self.kwargs.get("module_cfg", {}).get("log_grad_flow_steps", 100) == 0
        ):
            # NOTE: Logging grad flow simplified for print-based logging.
            pass

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
        # node count-conditioning
        if num_nodes is None:
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
            max_num_nodes = (
                self.dataset_info["max_n_nodes"]
                if "max_n_nodes" in self.dataset_info
                else num_nodes.max().item()
            )
            assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        if self.condition_on_context:
            if context is None:
                assert self.props_distr is not None, "props_distr must be initialized for conditional sampling"
                context = self.props_distr.sample_batch(num_nodes)
        else:
            context = None

        # sampling
        xh, batch_index, _ = self.ddpm.mol_gen_sample(
            num_samples=num_samples,
            num_nodes=num_nodes,
            node_mask=node_mask,
            context=context,
            fix_noise=fix_noise,
            fix_self_conditioning_noise=fix_noise,
            device=self.device,
            num_timesteps=num_timesteps
        )

        x = xh[:, :self.num_x_dims]
        one_hot = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
        charges = xh[:, -1:] if self.include_charges else torch.zeros(0, device=self.device)

        return x, one_hot, charges, batch_index

    @torch.inference_mode()
    @typechecked
    def optimize(
        self,
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
        num_timesteps: int,
        num_nodes: TensorType["batch_size"],
        context: TensorType["batch_size", "num_context_features"],
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        sampling_output_dir: Optional[str] = None,
        optim_property: Optional[str] = None,
        iteration_index: Optional[int] = None,
        return_frames: int = 1,
        id_from: int = 0,
        chain_viz_batch_element_idx: int = 0,
        name: str = os.sep + "chain",
        norm_with_original_timesteps: bool = False,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # context-conditioning
        if self.condition_on_context:
            if context is None:
                assert self.props_distr is not None, "props_distr must be initialized for conditional optimization"
                context = self.props_distr.sample_batch(num_nodes)
        else:
            raise Exception("Optimization requires a context conditional to optimize (e.g., `alpha`).")

        # sampling
        xh, batch_index, _ = self.ddpm.mol_gen_optimize(
            samples=samples,
            num_nodes=num_nodes,
            node_mask=node_mask,
            context=context,
            device=self.device,
            num_timesteps=num_timesteps,
            return_frames=return_frames,
            norm_with_original_timesteps=norm_with_original_timesteps,
        )

        # visualize optimized samples
        if return_frames > 1:
            assert all([p is not None for p in [sampling_output_dir, optim_property, iteration_index]]), \
                "Required parameters must be provided to visualize optimized molecules."
            
            # choose which molecule (i.e., the first) in the current batch to visualize
            xh_sample = xh[:, (batch_index == chain_viz_batch_element_idx), :]
            chain = reverse_tensor(xh_sample)

            # repeat last frame to see final sample better
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)

            # check stability of the generated molecule
            x_final = chain[-1, :, :self.num_x_dims].cpu().detach()
            one_hot_final = chain[-1, :, self.num_x_dims:-1] if self.include_charges else chain[-1, :, self.num_x_dims:]
            one_hot_final = torch.argmax(one_hot_final, dim=-1).cpu().detach()

            mol_stable = check_molecular_stability(
                positions=x_final,
                atom_types=one_hot_final,
                dataset_info=self.dataset_info
            )[0]

            # prepare entire chain
            x = chain[:, :, :self.num_x_dims]
            one_hot = chain[:, :, self.num_x_dims:-1] if self.include_charges else chain[:, :, self.num_x_dims:]
            one_hot = F.one_hot(
                torch.argmax(one_hot, dim=-1),
                num_classes=self.num_atom_types
            )
            charges = (
                torch.round(chain[:, :, -1:]).long()
                if self.include_charges
                else torch.zeros(0, dtype=torch.long, device=self.device)
            )

            if mol_stable and verbose:
                rank_zero_info("Found stable molecule to visualize :)")
            elif verbose:
                rank_zero_info("Did not find stable molecule to visualize :(")

            # flatten (i.e., treat frame (chain dimension) as batch for visualization)
            x_flat = x.view(-1, x.size(-1))
            one_hot_flat = one_hot.view(-1, one_hot.size(-1))
            charges_flat = torch.tensor([])
            batch_index_flat = torch.arange(x.size(0)).repeat_interleave(x.size(1))

            output_dir = Path(sampling_output_dir, optim_property, strftime("%Y%m%d-%H%M%S"), f"iteration_{iteration_index}", "chain")
            save_xyz_file(
                path=str(output_dir),
                positions=x_flat,
                one_hot=one_hot_flat,
                charges=charges_flat,
                dataset_info=self.dataset_info,
                id_from=id_from,
                name=name,
                batch_index=batch_index_flat
            )

            visualize_mol_chain(str(output_dir), dataset_info=self.dataset_info)

            x = xh[0, :, :self.num_x_dims]
            one_hot = xh[0, :, self.num_x_dims:-1] if self.include_charges else xh[0, :, self.num_x_dims:]
        
        # directly score optimize samples
        else:
            x = xh[:, :self.num_x_dims]
            one_hot = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
            charges = xh[:, -1:] if self.include_charges else torch.zeros(0, device=self.device)

        return x, one_hot, charges, batch_index

    @torch.inference_mode()
    @typechecked
    def sample_and_analyze(
        self,
        num_samples: int,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        batch_size: Optional[int] = None,
        max_num_nodes: Optional[int] = 100,
        num_timesteps: Optional[int] = None,
        id_from: int = 0,
        name: str = "molecule",
        save_molecules: bool = False,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        rank_zero_info(f"Analyzing molecule stability at epoch {self.current_epoch}...")
        diffusion_cfg = self.kwargs.get("diffusion_cfg", {})

        max_num_nodes = (
            self.dataset_info["max_n_nodes"]
            if "max_n_nodes" in self.dataset_info
            else max_num_nodes
        )
        if save_molecules and output_dir is None:
            if self.sampling_output_dir is None:
                sampling_output_dir = Path("sampling_and_analysis_output")
                sampling_output_dir.mkdir(exist_ok=True)
                self.sampling_output_dir = sampling_output_dir
            output_dir = self.sampling_output_dir / f"epoch_{self.current_epoch}"

        batch_size = diffusion_cfg.get("eval_batch_size", 64) if batch_size is None else batch_size
        batch_size = min(batch_size, num_samples)

        # note: each item in `molecules` is a tuple of (`position`, `atom_type_encoded`)
        molecules, atom_types, atom_one_hots, charges = [], [], [], []
        for _ in range(math.ceil(num_samples / batch_size)):
            # node count-conditioning
            num_samples_batch = min(batch_size, num_samples - len(molecules))
            assert self.ddpm.num_nodes_distribution is not None, "num_nodes_distribution must be initialized"
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples_batch)

            assert int(num_nodes.max()) <= max_num_nodes

            # context-conditioning
            if self.condition_on_context:
                if context is None:
                    assert self.props_distr is not None, "props_distr must be initialized for conditional sampling"
                    context = self.props_distr.sample_batch(num_nodes)
            else:
                context = None

            xh, batch_index, _ = self.ddpm.mol_gen_sample(
                num_samples=num_samples_batch,
                num_nodes=num_nodes,
                node_mask=node_mask,
                context=context,
                device=self.device,
                num_timesteps=num_timesteps,
            )

            x_ = xh[:, :self.num_x_dims].detach().cpu()
            atom_types_ = (
                xh[:, self.num_x_dims:-1].argmax(-1).detach().cpu()
                if self.include_charges
                else xh[:, self.num_x_dims:].argmax(-1).detach().cpu()
            )
            atom_one_hot_ = (
                xh[:, self.num_x_dims:-1].detach().cpu()
                if self.include_charges
                else xh[:, self.num_x_dims:].detach().cpu()
            )
            charges_ = (
                xh[:, -1]
                if self.include_charges
                else torch.zeros(0, device=self.device)
            )

            molecules.extend(
                list(
                    zip(
                        batch_tensor_to_list(x_, batch_index),
                        batch_tensor_to_list(atom_types_, batch_index)
                    )
                )
            )

            atom_types.extend(atom_types_.tolist())
            atom_one_hots.extend(atom_one_hot_.tolist())
            charges.extend(charges_.tolist())

        if save_molecules:
            save_xyz_file(
                path=str(output_dir) + "/",
                positions=torch.cat([pos for pos, _ in molecules], dim=0),
                one_hot=torch.tensor(atom_one_hots) if self.include_charges else torch.zeros(0),
                charges=torch.tensor(charges) if self.include_charges else torch.zeros(0),
                dataset_info=self.dataset_info,
                id_from=id_from,
                name=name,
                batch_index=torch.repeat_interleave(torch.arange(len(molecules)), torch.tensor([len(pos) for pos, _ in molecules])),
            )

        return self.analyze_samples(molecules, atom_types, charges)

    @typechecked
    def analyze_samples(
        self,
        molecules: List[Tuple[torch.Tensor, ...]],
        atom_types: List[int],
        charges: List[float]
    ) -> Dict[str, Any]:
        # assess distribution of node types
        kl_div_atom = (
            self.node_type_distribution.kl_divergence(atom_types)
            if self.node_type_distribution is not None
            else -1
        )

        # measure molecular stability
        molecule_stable, nr_stable_bonds, num_atoms = 0, 0, 0
        for pos, atom_type in molecules:
            validity_results = check_molecular_stability(
                positions=pos,
                atom_types=atom_type,
                dataset_info=self.dataset_info
            )
            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            num_atoms += int(validity_results[2])

        fraction_mol_stable = molecule_stable / float(len(molecules))
        fraction_atm_stable = nr_stable_bonds / float(num_atoms)

        # collect other basic molecular metrics
        metrics = self.molecular_metrics.evaluate(molecules)
        validity, uniqueness, novelty = metrics[0], metrics[1], metrics[2]

        return {
            "kl_div_atom_types": kl_div_atom,
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty
        }

    @torch.inference_mode()
    @typechecked
    def sample_and_save(
        self,
        num_samples: int,
        num_nodes: Optional[TensorType["batch_size"]] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        num_timesteps: Optional[int] = None,
        id_from: int = 0,
        name: str = "molecule",
        sampling_output_dir: Optional[Path] = None,
        norm_with_original_timesteps: bool = False,
    ):
        # node count-conditioning
        if num_nodes is None:
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
        max_num_nodes = (
            self.dataset_info["max_n_nodes"]
            if "max_n_nodes" in self.dataset_info
            else num_nodes.max().item()
        )
        assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        if self.condition_on_context:
            if context is None:
                context = self.props_distr.sample_batch(num_nodes)
        else:
            context = None

        # sampling
        xh, batch_index, _ = self.ddpm.mol_gen_sample(
            num_samples=num_samples,
            num_nodes=num_nodes,
            node_mask=node_mask,
            context=context,
            device=self.device,
            num_timesteps=num_timesteps,
            norm_with_original_timesteps=norm_with_original_timesteps,
        )

        x = xh[:, :self.num_x_dims]
        one_hot = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
        charges = xh[:, -1:] if self.include_charges else torch.zeros(0, device=self.device)

        if sampling_output_dir is not None:
            output_dir = sampling_output_dir
        else:
            if self.sampling_output_dir is None:
                self.sampling_output_dir = Path("sampling_output")
                self.sampling_output_dir.mkdir(exist_ok=True)
            output_dir = self.sampling_output_dir / f"epoch_{self.current_epoch}"

        save_xyz_file(
            path=str(output_dir) + "/",
            positions=x,
            one_hot=one_hot,
            charges=charges,
            dataset_info=self.dataset_info,
            id_from=id_from,
            name=name,
            batch_index=batch_index
        )

        if getattr(self, "logger", None) is not None and getattr(self.logger, "experiment", None) is not None and type(self.logger.experiment).__name__ == "Run":
            experiment = self.logger.experiment
        else:
            experiment = None

        visualize_mol(str(output_dir), dataset_info=self.dataset_info, wandb_run=experiment)

    @typechecked
    def sample_chain_and_save(
        self,
        keep_frames: int,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        id_from: int = 0,
        num_tries: int = 1,
        name: str = os.sep + "chain",
        verbose: bool = True
    ):
        # fixed hyperparameter(s)
        num_samples = 1

        # node count-conditioning
        if "QM9" in self.dataset_info["name"]:
            num_nodes = torch.tensor([19], dtype=torch.long, device=self.device)
        else:
            if verbose:
                rank_zero_info(f"Sampling `num_nodes` for dataset {self.dataset_info['name']}")
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
            max_num_nodes = (
                self.dataset_info["max_n_nodes"]
                if "max_n_nodes" in self.dataset_info
                else num_nodes.max().item()
            )
            assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        if self.condition_on_context:
            if context is None:
                context = self.props_distr.sample_batch(num_nodes)
        else:
            context = None

        one_hot, x = [None] * 2
        for i in range(num_tries):
            chain, _, _ = self.ddpm.mol_gen_sample(
                num_samples=num_samples,
                num_nodes=num_nodes,
                node_mask=node_mask,
                context=context,
                return_frames=keep_frames,
                device=self.device
            )

            chain = reverse_tensor(chain)

            # repeat last frame to see final sample better
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)

            # check stability of the generated molecule
            x_final = chain[-1, :, :self.num_x_dims].cpu().detach()
            one_hot_final = chain[-1, :, self.num_x_dims:-1] if self.include_charges else chain[-1, :, self.num_x_dims:]
            one_hot_final = torch.argmax(one_hot_final, dim=-1).cpu().detach()

            mol_stable = check_molecular_stability(
                positions=x_final,
                atom_types=one_hot_final,
                dataset_info=self.dataset_info
            )[0]

            # prepare entire chain
            x = chain[:, :, :self.num_x_dims]
            one_hot = chain[:, :, self.num_x_dims:-1] if self.include_charges else chain[:, :, self.num_x_dims:]
            one_hot = F.one_hot(
                torch.argmax(one_hot, dim=-1),
                num_classes=self.num_atom_types
            )
            charges = (
                torch.round(chain[:, :, -1:]).long()
                if self.include_charges
                else torch.zeros(0, dtype=torch.long, device=self.device)
            )

            if mol_stable and verbose:
                rank_zero_info("Found stable molecule to visualize :)")
                break
            elif i == num_tries - 1 and verbose:
                rank_zero_info("Did not find stable molecule :( -> showing last sample")

        # flatten (i.e., treat frame (chain dimension) as batch for visualization)
        x_flat = x.view(-1, x.size(-1))
        one_hot_flat = one_hot.view(-1, one_hot.size(-1))
        charges_flat = charges.view(-1, charges.size(-1)) if charges.numel() > 0 else charges
        batch_index_flat = torch.arange(x.size(0)).repeat_interleave(x.size(1))

        if self.sampling_output_dir is None:
            self.sampling_output_dir = Path("sampling_output")
            self.sampling_output_dir.mkdir(exist_ok=True)
        
        output_dir = self.sampling_output_dir / f"epoch_{self.current_epoch}" / "chain"
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_xyz_file(
            path=str(output_dir),
            positions=x_flat,
            one_hot=one_hot_flat,
            charges=charges_flat,
            dataset_info=self.dataset_info,
            id_from=id_from,
            name=name,
            batch_index=batch_index_flat
        )

        if getattr(self, "logger", None) is not None and getattr(self.logger, "experiment", None) is not None and type(self.logger.experiment).__name__ == "Run":
            experiment = self.logger.experiment
        else:
            experiment = None

        visualize_mol_chain(str(output_dir), dataset_info=self.dataset_info, wandb_run=experiment)

    @typechecked
    def generate_molecules(
        self,
        ddpm_mode: Literal["unconditional", "inpainting"],
        num_samples: int,
        num_nodes: Optional[TensorType["batch_size"]] = None,
        sanitize: bool = False,
        largest_frag: bool = False,
        add_hydrogens: bool = False,
        sample_chain: bool = False,
        relax_iter: int = 0,
        num_timesteps: Optional[int] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        **kwargs
    ) -> List[Chem.Mol]:
        """
        Generate molecules, with inpainting as an option
        Args:
            ddpm_mode: the method by which to generate molecules
            num_samples: number of samples to generate
            num_nodes: number of molecule nodes for each sample; sampled randomly if `None`
            sanitize: whether to sanitize molecules
            largest_frag: whether to return only the largest molecular fragment
            add_hydrogens: whether to include hydrogen atoms in the generated molecule
            sample_chain: whether to sample a chain of molecules
            relax_iter: number of force field optimization steps
            num_timesteps: number of denoising steps; will use training value instead if `None`
            node_mask: mask indicating which nodes are to be ignored during model generation
                    NOTE: `True` here means to fix a node's type and 3D position when `ddpm_mode=inpainting`;
                          `False` means to ignore a node when `ddpm_mode=unconditional`
            context: a batch of contextual features with which to condition the model's generations
            kwargs: additional e.g., inpainting parameters
        Returns:
            list of generated molecules
        """
        # node count-conditioning
        if num_nodes is None:
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
            max_num_nodes = (
                self.dataset_info["max_n_nodes"]
                if "max_n_nodes" in self.dataset_info
                else num_nodes.max().item()
            )
            assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        if self.condition_on_context:
            if context is None:
                context = self.props_distr.sample_batch(num_nodes)
        else:
            context = None

        # sampling
        if sample_chain:
            assert num_samples == 1, "Chain sampling is only supported for single-molecule batches."
        if ddpm_mode == "unconditional":
            # sample unconditionally
            xh, batch_index, _ = self.ddpm.mol_gen_sample(
                num_samples=num_samples,
                num_nodes=num_nodes,
                device=self.device,
                return_frames=num_timesteps if sample_chain else 1,
                num_timesteps=num_timesteps,
                node_mask=node_mask,
                context=context
            )

        elif ddpm_mode == "inpainting":
            # employ inpainting
            batch_index = num_nodes_to_batch_index(
                num_samples=len(num_nodes),
                num_nodes=num_nodes,
                device=self.device
            )

            molecule = {
                "x": torch.zeros(
                    (len(batch_index), self.num_x_dims),
                    device=self.device,
                    dtype=torch.float
                ),
                "one_hot": torch.zeros(
                    (len(batch_index), self.num_atom_types),
                    device=self.device,
                    dtype=torch.float
                ),
                "charges": torch.zeros(
                    (len(batch_index), 1),
                    device=self.device,
                    dtype=torch.float
                ),
                "num_nodes": num_nodes,
                "batch_index": batch_index
            }

            if node_mask is None:
                # largely disable inpainting by sampling for all but the first node
                node_mask = torch.zeros(len(batch_index), dtype=torch.bool, device=self.device)
                node_mask[0] = True  # note: an arbitrary choice of a generation's fixed point
            else:
                # inpaint requested region as specified in `node_mask`
                pass

            # record molecule's original center of mass
            molecule_com_before = scatter(molecule["x"], batch_index, dim=0, reduce="mean")

            xh = self.ddpm.inpaint(
                molecule=molecule,
                node_mask_fixed=node_mask,
                num_timesteps=num_timesteps,
                context=context,
                **kwargs
            )

            # move generated molecule back to its original center of mass position
            molecule_com_after = scatter(xh[:, :self.num_x_dims], batch_index, dim=0, reduce="mean")
            xh[:, :self.num_x_dims] += (molecule_com_before - molecule_com_after)[batch_index]

        else:
            raise NotImplementedError(f"DDPM type {type(self.ddpm)} is currently not implemented.")

        # build RDKit molecule objects
        molecules = []
        if sample_chain:
            xh = reverse_tensor(xh)
            x = xh[:, :, :self.num_x_dims].detach().cpu()
            atom_types = (
                xh[:, :, self.num_x_dims:-1].argmax(-1).detach().cpu()
                if self.include_charges
                else xh[:, :, self.num_x_dims:].argmax(-1).detach().cpu()
            )
            for atom_pos, atom_types in zip(x, atom_types):
                mol = build_molecule(
                    atom_pos,
                    atom_types,
                    dataset_info=self.dataset_info,
                    add_coords=True
                )
                mol = process_molecule(
                    rdmol=mol,
                    add_hydrogens=add_hydrogens,
                    sanitize=sanitize,
                    relax_iter=relax_iter,
                    largest_frag=largest_frag
                )
                if mol is not None:
                    molecules.append(mol)
        else:
            x = xh[:, :self.num_x_dims].detach().cpu()
            atom_types = (
                xh[:, self.num_x_dims:-1].argmax(-1).detach().cpu()
                if self.include_charges
                else xh[:, self.num_x_dims:].argmax(-1).detach().cpu()
            )
            # TODO: incorporate charges in some meaningful way
            charges = (
                xh[:, -1].detach().cpu()
                if self.include_charges
                else torch.zeros(0, device=self.device)
            )
            for mol_pc in zip(
                batch_tensor_to_list(x, batch_index),
                batch_tensor_to_list(atom_types, batch_index)
            ):

                mol = build_molecule(
                    *mol_pc,
                    dataset_info=self.dataset_info,
                    add_coords=True
                )
                mol = process_molecule(
                    rdmol=mol,
                    add_hydrogens=add_hydrogens,
                    sanitize=sanitize,
                    relax_iter=relax_iter,
                    largest_frag=largest_frag
                )
                if mol is not None:
                    molecules.append(mol)

        return molecules

    @typechecked
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @typechecked
    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        verbose: bool = False
    ):
        if not self.clip_gradients:
            return

        # allow gradient norm to be 150% + 2 * stdev of recent gradient history
        max_grad_norm = (
            1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
        )

        # get current `grad_norm`
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = get_grad_norm(params)

        # note: Gradient clipping must be handled manually or via a different framework
        nn.utils.clip_grad_norm_(
            params,
            max_norm=max_grad_norm,
            norm_type=2.0
        )

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if verbose:
            rank_zero_info(f"Current gradient norm: {grad_norm}")

        if float(grad_norm) > max_grad_norm:
            rank_zero_info(
                f"Clipped gradient with value {grad_norm:.1f}, since the maximum value currently allowed is {max_grad_norm:.1f}")

    def on_fit_end(self):
        """Callback refactored for manual usage."""
        path_cfg = self.kwargs.get("path_cfg")
        if path_cfg is not None and getattr(path_cfg, "grid_search_script_dir", None) is not None:
            # uniquely record when model training is concluded
            grid_search_script_dir = path_cfg.grid_search_script_dir
            # run_id would need to be tracked manually
            run_id = "manual_run" 
            fit_end_indicator_filename = f"{run_id}.{HALT_FILE_EXTENSION}"
            fit_end_indicator_filepath = os.path.join(grid_search_script_dir, fit_end_indicator_filename)
            os.makedirs(grid_search_script_dir, exist_ok=True)
            with open(fit_end_indicator_filepath, "w") as f:
                f.write("`on_fit_end` has been called.")
        return


if __name__ == "__main__":
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch_geometric.data import Data, Batch
    from omegaconf import OmegaConf
    from torch_pharma.data.datasets.utils import TORCH_PHARMA_HOME

    # initialization
    def optimizer_fn(params): return AdamW(params, lr=1e-4)
    def scheduler_fn(optimizer): return CosineAnnealingLR(optimizer, T_max=100)

    # initialize dataloaders
    print("Initializing dataloaders...")
    dataloader_cfg = OmegaConf.create({
        "dataset": "QM9",
        "batch_size": 2, # small batch size for quick test
        "num_workers": 0,
        "filter_n_atoms": None,
        "data_dir": str(TORCH_PHARMA_HOME),
        "subtract_thermo": True,
        "force_download": False,
        "remove_h": True,
        "create_pyg_graphs": True,
        "num_radials": 1,
        "device": "cpu",
        "include_charges": True
    })
    dataloaders, charge_scale = retrieve_dataloaders(dataloader_cfg)

    print("Initializing model...")
    model = QM9MoleculeGenerationDDPM(
        optimizer=optimizer_fn,
        scheduler=scheduler_fn,
        conditioning=['alpha'],
        remove_h=True,
        dataloaders=dataloaders
    )
    model.to("cuda")

    # run a training step
    print("Running training step with real data...")
    # dataset_iter = iter(dataloaders["train"])
    # batch = next(dataset_iter).to(model.device)
    
    # model.train()
    # try:
    #     metrics = model.training_step(batch, 0)
    #     print(f"Training metrics: {metrics}")
    # except Exception as e:
    #     print(f"Training step failed: {e}")
    #     import traceback
    #     traceback.print_exc()

    # # run a validation step
    # print("Running validation step with real data...")
    # val_batch = next(iter(dataloaders["valid"])).to(model.device)
    # model.eval()
    # try:
    #     metrics = model.validation_step(val_batch, 0)
    #     print(f"Validation metrics: {metrics}")
    # except Exception as e:
    #     print(f"Validation step failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    from tqdm import tqdm
    for epoch in range(100):

        print(f"Epoch {epoch+1} training start...")
        for batch in tqdm(dataloaders['train']):
            batch = batch.to(model.device)
            metrics = model.training_step(batch, 0)
            print(f"Training metrics: {metrics}")

        print(f"Epoch {epoch+1} validation start...")
        for batch in tqdm(dataloaders['valid']):
            batch = batch.to(model.device)
            metrics = model.validation_step(batch, 0)
            print(f"Validation metrics: {metrics}")

        torch.save(model.state_dict(), f"examples/molecule_generation/2302_04313/checkpoints/model_{epoch}.pt")

    # run sampling
    print("Running sampling...")
    try:
        x, one_hot, charges, batch_index = model.sample(num_samples=2)
        print(f"Sampled {len(torch.unique(batch_index))} molecules")
        print(f"Positions shape: {x.shape}")
    except Exception as e:
        print(f"Sampling failed: {e}")
        import traceback
        traceback.print_exc()

            
