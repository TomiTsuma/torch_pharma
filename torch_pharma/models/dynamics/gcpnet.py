import torch
from torch import nn
from torch_pharma.features import safe_norm, get_nonlinearity, ScalarVector
from typing import Tuple, Optional, Union
from torchtyping import patch_typeguard, TensorType
from typeguard import typechecked
from torch_pharma.features import is_identity, vectorize, scalarize

patch_typeguard()  # use before @typechecked

class GCP(nn.Module):
    def __init__(
            self,
            input_dims: ScalarVector,
            output_dims: ScalarVector,
            nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
            scalar_out_nonlinearity: Optional[str] = "silu",
            scalar_gate: int = 0,
            vector_gate: bool = True,
            frame_gate: bool = False,
            sigma_frame_gate: bool = False,
            feedforward_out: bool = False,
            bottleneck: int = 1,
            vector_residual: bool = False,
            vector_frame_residual: bool = False,
            ablate_frame_updates: bool = False,
            ablate_scalars: bool = False,
            ablate_vectors: bool = False,
            scalarization_vectorization_output_dim: int = 3,
            **kwargs
    ):
        super().__init__()

        if nonlinearities is None:
            nonlinearities = (None, None)

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_nonlinearity(nonlinearities[0], return_functional=True),
            get_nonlinearity(nonlinearities[1], return_functional=True)
        )
        self.scalar_gate, self.vector_gate, self.frame_gate, self.sigma_frame_gate = (
            scalar_gate, vector_gate, frame_gate, sigma_frame_gate
        )
        self.vector_residual, self.vector_frame_residual = vector_residual, vector_frame_residual
        self.ablate_frame_updates = ablate_frame_updates
        self.ablate_scalars, self.ablate_vectors = ablate_scalars, ablate_vectors

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                self.vector_input_dim % bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"

            self.hidden_dim = self.vector_input_dim // bottleneck if bottleneck > 1 else max(self.vector_input_dim,
                                                                                             self.vector_output_dim)

            self.vector_down = nn.Linear(self.vector_input_dim, self.hidden_dim, bias=False)
            self.scalar_out = nn.Sequential(
                nn.Linear(self.hidden_dim + self.scalar_input_dim, self.scalar_output_dim),
                get_nonlinearity(scalar_out_nonlinearity),
                nn.Linear(self.scalar_output_dim, self.scalar_output_dim)
            ) if feedforward_out else nn.Linear(self.hidden_dim + self.scalar_input_dim, self.scalar_output_dim)

            if self.vector_output_dim:
                self.vector_up = nn.Linear(self.hidden_dim, self.vector_output_dim, bias=False)
                if self.vector_gate:
                    self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)

            if not self.ablate_frame_updates:
                vector_down_frames_input_dim = self.hidden_dim if not self.vector_output_dim else self.vector_output_dim
                self.vector_down_frames = nn.Linear(vector_down_frames_input_dim,
                                                    scalarization_vectorization_output_dim, bias=False)
                self.scalar_out_frames = nn.Linear(
                    self.scalar_output_dim + scalarization_vectorization_output_dim * 3, self.scalar_output_dim)

                if self.vector_output_dim and self.sigma_frame_gate:
                    self.vector_out_scale_sigma_frames = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
                elif self.vector_output_dim and self.frame_gate:
                    self.vector_out_scale_frames = nn.Linear(
                        self.scalar_output_dim, scalarization_vectorization_output_dim * 3)
                    self.vector_up_frames = nn.Linear(
                        scalarization_vectorization_output_dim, self.vector_output_dim, bias=False)
        else:
            self.scalar_out = nn.Sequential(
                nn.Linear(self.scalar_input_dim, self.scalar_output_dim),
                get_nonlinearity(scalar_out_nonlinearity),
                nn.Linear(self.scalar_output_dim, self.scalar_output_dim)
            ) if feedforward_out else nn.Linear(self.scalar_input_dim, self.scalar_output_dim)

    @typechecked
    def process_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "m"],
        vector_hidden_rep: TensorType["batch_num_entities", 3, "n"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        vector_rep = self.vector_up(vector_hidden_rep)
        if self.vector_residual:
            vector_rep = vector_rep + v_pre
        vector_rep = vector_rep.transpose(-1, -2)
        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def create_zero_vector(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"]
    ) -> TensorType["batch_num_entities", "o", 3]:
        return torch.zeros(scalar_rep.shape[0], self.vector_output_dim, 3, device=scalar_rep.device)

    @typechecked
    def process_vector_frames(
        self,
        scalar_rep: TensorType["batch_num_entities", "merged_scalar_dim"],
        v_pre: TensorType["batch_num_entities", 3, "o"],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> TensorType["batch_num_entities", "p", 3]:
        vector_rep = v_pre.transpose(-1, -2)
        if self.sigma_frame_gate:
            # bypass vectorization in favor of row-wise gating
            gate = self.vector_out_scale_sigma_frames(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif self.frame_gate:
            # apply elementwise gating between localized frame vectors and vector residuals
            gate = self.vector_out_scale_frames(self.vector_nonlinearity(scalar_rep))
            # perform frame-gating, where edges must be present
            gate_vector = vectorize(
                gate,
                edge_index,
                frames,
                node_inputs=node_inputs,
                dim_size=scalar_rep.shape[0],
                node_mask=node_mask
            )
            # ensure the channels for `coordinates` are being left-multiplied
            gate_vector_rep = self.vector_up_frames(gate_vector.transpose(-1, -2)).transpose(-1, -2)
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(gate_vector_rep, dim=-1, keepdim=True))
            if self.vector_frame_residual:
                vector_rep = vector_rep + v_pre.transpose(-1, -2)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @typechecked
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                TensorType["batch_num_entities", "scalar_dim"],
                TensorType["batch_num_entities", "m", "vector_dim"]
            ],
            TensorType["batch_num_entities", "merged_scalar_dim"]
        ],
        edge_index: TensorType[2, "batch_num_edges"],
        frames: TensorType["batch_num_edges", 3, 3],
        node_inputs: bool = False,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_entities", "new_scalar_dim"],
            TensorType["batch_num_entities", "n", "vector_dim"]
        ],
        TensorType["batch_num_entities", "new_scalar_dim"]
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)
        else:
            merged = s_maybe_v
            merged = torch.zeros_like(merged) if self.ablate_scalars else merged

        scalar_rep = self.scalar_out(merged)

        if self.vector_input_dim and self.vector_output_dim:
            vector_rep = self.process_vector(scalar_rep, v_pre, vector_hidden_rep)

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        vector_rep = self.create_zero_vector(
            scalar_rep
        ) if self.vector_output_dim and not self.vector_input_dim else vector_rep

        if self.ablate_frame_updates:
            return ScalarVector(scalar_rep, vector_rep) if self.vector_output_dim else scalar_rep

        # GCP: update scalar features using complete local frames
        v_pre = vector_rep.transpose(-1, -2)
        vector_hidden_rep = self.vector_down_frames(v_pre)
        scalar_hidden_rep = scalarize(
            vector_hidden_rep.transpose(-1, -2),
            edge_index,
            frames,
            node_inputs=node_inputs,
            dim_size=vector_hidden_rep.shape[0],
            node_mask=node_mask
        )
        merged = torch.cat((scalar_rep, scalar_hidden_rep), dim=-1)

        scalar_rep = self.scalar_out_frames(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using complete local frames (e.g., in the case of a final layer)
            scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
            return self.scalar_nonlinearity(scalar_rep)

        # GCP: update vector features using complete local frames
        if self.vector_input_dim and self.vector_output_dim:
            vector_rep = self.process_vector_frames(
                scalar_rep,
                v_pre,
                edge_index,
                frames,
                node_inputs=node_inputs,
                node_mask=node_mask
            )

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        scalar_rep = torch.zeros_like(scalar_rep) if self.ablate_scalars else scalar_rep
        vector_rep = torch.zeros_like(vector_rep) if self.ablate_vectors else vector_rep
        return ScalarVector(scalar_rep, vector_rep)
