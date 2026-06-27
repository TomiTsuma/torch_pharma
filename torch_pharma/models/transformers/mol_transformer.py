from torch import nn
import torch
from torch_pharma.models.radial_basis import *
from torch_pharma.models.layer.feed_forward import GVPFFNLayer
from torch_pharma.features.utils.tools import _unit_edges_from_block_edges 
from torch_pharma.models.layer.ept.ept import EPTLayer
from torch_scatter import scatter_sum, scatter_mean
from torch_pharma.features.utils.nn_utils import graph_to_batch_nx


try:
    from xformers.ops import memory_efficient_attention as attn_func
    xformers_enable = True
except:
    xformers_enable = False

class Transformer(nn.Module):
    '''Equivariant Adaptive Block Transformer'''

    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            n_layers,
            n_rbf,
            d_edge,
            cutoff=7.0,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            use_edge_feat = False,
            local_mask = False,
            attn_bias = True,
            sparse_k=None,
            efficient=False,
            vector_act='none',
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.use_edge_feat = use_edge_feat
        self.sparse_k = sparse_k
        self.efficient = efficient
        self._local_mask = local_mask
        if self.efficient and not xformers_enable:
            print("xformers are not downloaded, change into custom attention mechanism. "
                  "Please install xformers via 'pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121',"
                  "or seek 'https://github.com/facebookresearch/xformers' for more details.")
            self.efficient = False

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge + n_rbf, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden * 2)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden)
        )
        
        self.final_v = GVPFFNLayer(
            d_hidden, d_ffn, act_fn, d_output=1
        )

        self.n_rbf = n_rbf
        if n_rbf > 1:
            self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

        if self.use_edge_feat:
            self.rbf_mapping = nn.Linear(n_rbf + d_edge, n_layers)    
        else:
            self.rbf_mapping = nn.Linear(n_rbf, n_layers)

        if self.layer_norm == 'pre':
            self.ln = nn.LayerNorm(d_hidden)

        
        for i in range(0, n_layers):
            self.add_module(f'layer_{i}', EPTLayer(
                d_hidden, d_ffn, n_heads, i, act_fn, layer_norm, residual, self.efficient, vector_act, attn_bias
            ))

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None):
        with torch.no_grad():
            if topo_edges is not None:
                # first delete self-loop of 3D edges. Otherwise there might be two same atom-level edges overwriting each other
                not_self_loop = edges[0] != edges[1]
                edges = edges.T[not_self_loop].T
                if edge_attr is not None: edge_attr = edge_attr[not_self_loop]
            (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = _unit_edges_from_block_edges(block_id, edges.T, Z, k=self.sparse_k) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j
        
        if edge_attr is not None: edge_attr = edge_attr[block_edge_id]
        
        # concat 3D and 2D edges
        if topo_edges is not None:
            unit_row = torch.cat([unit_row, topo_edges[0]], dim=0)
            unit_col = torch.cat([unit_col, topo_edges[1]], dim=0)
        if topo_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, topo_edge_attr], dim=0) # [E1 + E2, d]

        # vector init
        Z = Z.view(-1, 3)
        edge_vec = Z[unit_row] - Z[unit_col] # [Ne, 3]
        edge_dis = torch.norm(edge_vec, dim=-1)
        dis_feat = self.rbf(edge_dis)
        edge_feat = torch.cat([H[unit_row], H[unit_col], dis_feat, edge_attr], dim=-1)
        edge_scaler = self.edge_mlp(edge_feat) # [Ne, d_hidden]
        inv_feat, equiv_feat = torch.split(edge_scaler, self.d_hidden, dim=-1)
        edge_scas = H[unit_col] * inv_feat
        edge_vecs = edge_vec.unsqueeze(-1) * equiv_feat.unsqueeze(-2) # [Ne, 3, d_hidden]
        
        H = self.node_mlp(torch.cat([H, scatter_sum(edge_scas, unit_row, dim_size=H.shape[0], dim=0)], dim=-1))
        V = scatter_mean(edge_vecs, unit_row, dim_size=H.shape[0], dim=0)

        # graph to batch
        batch_to_nodes = batch_id[block_id]
        H_batch, H_mask = graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8)
        bs, max_n = H_batch.shape[0], H_batch.shape[1]
        V_batch = torch.zeros((bs, max_n, *V.shape[1:]), dtype=V.dtype, device=V.device)
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, max_n, *Z.shape[1:]), dtype=Z.dtype, device=Z.device)
        Z_batch[H_mask] = Z

        # rbf to all layer & heads
        if self.use_edge_feat:
            dis_feat = torch.cat([dis_feat, edge_attr], dim=-1)
        rbf_feat = self.rbf_mapping(dis_feat)
        lengths = torch.zeros(bs, dtype=batch_id.dtype, device=batch_id.device)
        lengths[1:] = torch.cumsum(scatter_sum(torch.ones_like(batch_to_nodes), batch_to_nodes), dim=-1)[:-1]  # [bs]
        lengths = lengths[batch_to_nodes]
        tot_idx = torch.cumsum(torch.ones_like(batch_to_nodes), dim=-1) - 1
        self_idx = tot_idx - lengths
        if self._local_mask:
            rbf_feat_batch = torch.ones((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device) * float('-inf')
            rbf_feat_batch[~H_mask] = 0.0 # to prevent nan in padding which will lead to 0 * nan = nan (broadcast to other positions)
        else:
            rbf_feat_batch = torch.zeros((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device)
        if attn_mask is not None:
            rbf_feat_batch[~attn_mask] = float('-inf')
        rbf_feat_batch[batch_to_nodes[unit_row], self_idx[unit_row], self_idx[unit_col]] = rbf_feat
        rbf_feat_batch = rbf_feat_batch.reshape(bs, max_n, max_n, self.n_layers, -1).permute(3,0,4,1,2) # [l, bs, h, n, n]

        # svd init
        D_batch = torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1) # [bs, n, n]
        D_batch = -D_batch

        cached_info = (D_batch.detach(), rbf_feat_batch, H_mask)    

        for i in range(self.n_layers):
            H_batch, V_batch = self._modules[f'layer_{i}'](
                H_batch, V_batch, cached_info
            )
        
        if self.layer_norm == 'pre':
            H_batch = self.ln(H_batch)


        H_graph = H_batch[H_mask]
        V_graph = V_batch[H_mask]

        V_graph = V_graph / (V_graph.norm(dim=-2, keepdim=True) + 1e-5)

        _, V_graph = self.final_v(H_graph, V_graph)
        return H_graph, V_graph
