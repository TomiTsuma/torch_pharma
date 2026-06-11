
import torch
from torch import nn
from torch_pharma.models.diffusion.utils import coord2dist, modulate, uncompilable_dropout
from torch_pharma.models.message_passing.update_layer import ConditionalEquivariantUpdate
from torch_scatter import scatter
from torch_pharma.models.message_passing.transformation_layer import (
    TransformLayer,
    TransformLayerOptim,
    TransformLayer as TransLayer,
    TransformLayerOptim as TransLayerOptim,
)
from torch_pharma.models.diffusion.noise import GaussianLayer
from torch_pharma.utils.device import is_amd_gpu

disable_compile = is_amd_gpu()


class EquivariantBlock(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(self, node_dim, edge_dim, time_dim, num_heads,
                 cond_time=True, mlp_ratio=4, act=nn.GELU, dropout=0.1, dist_emb=False, equi_pos=False, pair_update=True, fuse_qkv=False):
        super().__init__()

        self.dropout = dropout
        self.act1 = act()
        self.act2 = act()
        self.cond_time = cond_time
        dist_dim = edge_dim
        self.dist_emb = dist_emb
        self.pair_update = pair_update
        if dist_emb:
            self.edge_emb = nn.Linear(edge_dim + dist_dim, edge_dim)
            self.dist_layer = GaussianLayer(dist_dim)
        else:
            if self.pair_update:
                self.edge_emb = nn.Linear(edge_dim, edge_dim)
            else:
                self.edge_emb = nn.Sequential(
                    nn.Linear(edge_dim, edge_dim * 2),
                    nn.GELU(),
                    nn.Linear(edge_dim * 2, edge_dim),
                    nn.LayerNorm(edge_dim),
                )

        # message passing layer
        if fuse_qkv:
            self.attn_mpnn = TransLayerOptim(node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout)
        else:
            self.attn_mpnn = TransLayer(node_dim, node_dim // num_heads, num_heads, edge_dim=edge_dim, dropout=dropout)

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)

        if pair_update:
            self.node2edge_lin = nn.Linear(node_dim, edge_dim)
        # Feed forward block -> edge.
        self.ff_linear3 = nn.Linear(edge_dim, edge_dim * mlp_ratio)
        self.ff_linear4 = nn.Linear(edge_dim * mlp_ratio, edge_dim)

        # equivariant edge update layer
        self.equi_pos = equi_pos
        if self.equi_pos:
            self.equi_update = ConditionalEquivariantUpdate(node_dim, edge_dim, dist_dim, time_dim)

        if self.cond_time:
            self.node_time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, node_dim * 6)
            )
            # Normalization for MPNN
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=False, eps=1e-6)

            if self.pair_update:
                self.edge_time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_dim, edge_dim * 6)
                )
                self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)
            self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            self.norm2_node = nn.LayerNorm(node_dim, elementwise_affine=True, eps=1e-6)
            if self.pair_update:
                self.norm1_edge = nn.LayerNorm(edge_dim, elementwise_affine=True, eps=1e-6)
            self.norm2_edge = nn.LayerNorm(edge_dim, elementwise_affine=True, eps=1e-6)


    def _ff_block_node(self, x):
        x = uncompilable_dropout(self.act1(self.ff_linear1(x)), p=self.dropout, training=self.training)
        return uncompilable_dropout(self.ff_linear2(x), p=self.dropout, training=self.training)

    def _ff_block_edge(self, x):
        x = uncompilable_dropout(self.act2(self.ff_linear3(x)), p=self.dropout, training=self.training)
        return uncompilable_dropout(self.ff_linear4(x), p=self.dropout, training=self.training)

    def forward_old(self, pos, h, edge_attr, edge_index, node_mask, node_time_emb=None, edge_time_emb=None):
        """
        Params:
            pos: [B*N, 3]
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
            node_mask: [B*N, 1]
            extra_heads: [N_edge, extra_heads]
        """
        h_in_node = h
        h_in_edge = edge_attr

        # obtain distance feature
        if self.dist_emb:
            distance = coord2dist(pos, edge_index)
            distance = self.dist_layer(distance, edge_time_emb)
            edge_attr = self.edge_emb(torch.cat([distance, edge_attr], dim=-1))
        else:
            edge_attr = self.edge_emb(edge_attr)

        # time (noise level) condition
        if self.cond_time:
            node_shift_msa, node_scale_msa, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp = \
                self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            edge_shift_msa, edge_scale_msa, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp = \
                self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)

            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
            edge_attr = modulate(self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa)
        else:
            h = self.norm1_node(h)
            edge_attr = self.norm1_edge(edge_attr)

        # apply transformer-based message passing, update node features and edge features (FFN + norm)
        h_node = self.attn_mpnn(h, edge_index, edge_attr)
        h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
        h_edge = self.node2edge_lin(h_edge)

        h_node = h_in_node + node_gate_msa * h_node if self.cond_time else h_in_node + h_node
        _h_node = modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp) * node_mask if self.cond_time else \
                self.norm2_node(h_node) * node_mask
        h_out = (h_node + node_gate_mlp * self._ff_block_node(_h_node)) * node_mask if self.cond_time else \
                (h_node + self._ff_block_node(_h_node)) * node_mask

        h_edge = h_in_edge + edge_gate_msa * h_edge if self.cond_time else h_in_edge + h_edge
        _h_edge = modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp) if self.cond_time else \
                self.norm2_edge(h_edge)
        h_edge_out = h_edge + edge_gate_mlp * self._ff_block_edge(_h_edge) if self.cond_time else \
                    h_edge + self._ff_block_edge(_h_edge)

        # apply equivariant coordinate update
        if self.equi_pos:
            pos = self.equi_update(h_out, pos, edge_index, h_edge_out, distance, edge_time_emb)

        return h_out, h_edge_out, pos


    def forward(self, pos, h, edge_attr, edge_index, node_mask, node_time_emb=None, edge_time_emb=None):
        """
        A more optimized version of forward_old using torch.compile
        Params:
            pos: [B*N, 3]
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
            node_mask: [B*N, 1]
            extra_heads: [N_edge, extra_heads]
        """
        h_in_node = h
        h_in_edge = edge_attr

        # obtain distance feature
        if self.dist_emb:
            distance = coord2dist(pos, edge_index)
            distance = self.dist_layer(distance, edge_time_emb)
            edge_attr = self.edge_emb(torch.cat([distance, edge_attr], dim=-1))
        else:
            edge_attr = self.edge_emb(edge_attr)

        # time (noise level) condition
        if self.cond_time:
            node_shift_msa, node_scale_msa, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp = \
                self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
            if self.pair_update:
                edge_shift_msa, edge_scale_msa, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp = \
                    self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)
                edge_attr = modulate(self.norm1_edge(edge_attr), edge_shift_msa, edge_scale_msa)
        else:
            h = self.norm1_node(h)
            if self.pair_update:
                edge_attr = self.norm1_edge(edge_attr)

        # apply transformer-based message passing, update node features and edge features (FFN + norm)
        h_node = self.attn_mpnn(h, edge_index, edge_attr)
        h_out = self.node_update(h_in_node, h_node, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp, node_mask)

        if self.pair_update:
            h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
            h_edge_out = self.edge_update(h_in_edge, h_edge, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp)
        else:
            h_edge_out = h_in_edge

        # apply equivariant coordinate update
        if self.equi_pos:
            pos = self.equi_update(h_out, pos, edge_index, h_edge_out, distance, edge_time_emb)

        return h_out, h_edge_out, pos

    @torch.compile(dynamic=True, disable=disable_compile)
    def node_update(self, h_in_node, h_node, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp, node_mask):
        h_node = h_in_node + node_gate_msa * h_node if self.cond_time else h_in_node + h_node
        _h_node = modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp) * node_mask if self.cond_time else \
                self.norm2_node(h_node) * node_mask
        h_out = (h_node + node_gate_mlp * self._ff_block_node(_h_node)) * node_mask if self.cond_time else \
                (h_node + self._ff_block_node(_h_node)) * node_mask
        return h_out

    @torch.compile(dynamic=True, disable=disable_compile)
    def edge_update(self, h_in_edge, h_edge, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp):
        h_edge = self.node2edge_lin(h_edge)
        h_edge = h_in_edge + edge_gate_msa * h_edge if self.cond_time else h_in_edge + h_edge
        _h_edge = modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp) if self.cond_time else \
                self.norm2_edge(h_edge)
        h_edge_out = h_edge + edge_gate_mlp * self._ff_block_edge(_h_edge) if self.cond_time else \
                    h_edge + self._ff_block_edge(_h_edge)
        return h_edge_out
