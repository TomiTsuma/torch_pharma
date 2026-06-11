import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from typing import Union
from types import SimpleNamespace

from torch_pharma.models.dynamics.equivariance import EquivariantBlock
from torch_pharma.models.dynamics.projector import ExtendedProjector
from torch_pharma.models.embedding.learned_sinusoidal_pos_embedding import LearnedSinusodialposEmb
from torch_pharma.models.message_passing.update_layer import ConditionalEquivariantUpdate
from torch_pharma.models.message_passing.transformation_layer import (
    TransformLayer as TransLayer,
    TransformLayerOptim as TransLayerOptim,
)
from torch_pharma.models.diffusion.utils import coord2dist, remove_mean_with_mask, remove_mean
from torch_pharma.models.diffusion.noise import GaussianLayer
from torch_pharma.models.diffusion.config import DGTDiffusionConfig

__all__ = ["DGTDiffusion", "DGTDiffusionConfig", "TransLayer", "TransLayerOptim"]



class DGTDiffusion(nn.Module):
    """V2: Predict pos noise in the last block."""

    @staticmethod
    def add_args(parser):
        parser.add_argument('--in_node_features', type=int, default=44)
        parser.add_argument('--in_edge_features', type=int, default=4)
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--n_blocks', type=int, default=10)
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--enable_equiv', action='store_true', default=False)
        parser.add_argument('--use_original_dgt', action='store_true', default=False)
        parser.add_argument('--pred_noise', action='store_true', default=True)
        parser.add_argument('--mlp_ratio', type=int, default=4)
        parser.add_argument('--disable_com', action='store_true', default=True)
        parser.add_argument('--trans_linear', action='store_true', default=True)
        parser.add_argument('--disable_extra_gelu', action='store_true', default=False)
        parser.add_argument('--not_pair_update', action='store_true', default=False)
        parser.add_argument('--fuse_qkv', action='store_true', default=False)

    def __init__(self, config: Union[DGTDiffusionConfig, SimpleNamespace], in_dim=None):
        super().__init__()
        if isinstance(config, DGTDiffusionConfig):
            args = SimpleNamespace(**config.__dict__)
        else:
            args = config
        self.args = args
        self.pred_noise = args.pred_noise
        self.use_original_dgt = args.use_original_dgt
        self.disable_com = args.disable_com
        self.disable_extra_gelu = args.disable_extra_gelu
        self.pair_update = not args.not_pair_update

        time_dim = args.hidden_size
        hidden_dim = args.hidden_size
        edge_dim = hidden_dim // 4
        self.n_blocks = args.n_blocks

        # noise level conditioning embedding
        learned_dim = 16
        sinu_pos_emb = LearnedSinusodialposEmb(learned_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(learned_dim + 1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Conditional MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_lin = nn.Linear(hidden_dim, time_dim) # multiple condition: hidden_dim -> cond_ch * hidden_dim

        # distance GBF embedding
        self.dist_gbf = GaussianLayer(edge_dim)

        self.enable_equiv = args.enable_equiv
        self.delta_train = self.args.delta_train
        self.use_llm = self.args.use_llm
        # initial mapping
        if self.enable_equiv:
            self.node_emb = nn.Linear(args.in_node_features, hidden_dim)
        else:
            if self.use_llm and not self.delta_train:
                self.node_emb = nn.Sequential(
                    nn.Linear(in_dim + args.in_node_features + 3, 2 * hidden_dim),
                    nn.GELU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
            else:
                self.node_emb = nn.Sequential(
                        nn.Linear(args.in_node_features + 3, hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, hidden_dim)
                    )

        self.edge_emb = nn.Linear(args.in_edge_features + edge_dim, edge_dim)
        for i in range(self.n_blocks):
            self.add_module(f'block_{i}', EquivariantBlock(hidden_dim, edge_dim, time_dim,
                            args.n_heads, dropout=args.dropout, dist_emb=self.use_original_dgt, equi_pos=self.use_original_dgt, mlp_ratio=args.mlp_ratio, act=nn.GELU, pair_update=self.pair_update, fuse_qkv=args.fuse_qkv))

        if self.use_original_dgt:
            assert self.enable_equiv
            assert not self.pred_noise
        else:
            if self.enable_equiv:
                # last block for predicting pos noise
                self.dist_gbf2 = GaussianLayer(edge_dim)
                self.pred_pos_noise = ConditionalEquivariantUpdate(hidden_dim, edge_dim, edge_dim, time_dim, residual=False)
            else:
                self.final_linear = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 3, bias=False)
                )


        self.llm_cond = self.args.llm_cond
        if self.use_llm:
            if self.llm_cond:
                self.projector = nn.Sequential(
                    nn.Linear(in_dim, 4 * hidden_dim),
                    nn.GELU(),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                )
            else:
                if self.delta_train:
                    self.extended_node_emb = ExtendedProjector(self.node_emb, in_dim, hidden_dim, disable_extra_gelu=self.disable_extra_gelu)

    def forward(self, data, lm_x=None, context=None):
        # sparse to dense format: node_h, node_mask, pos, t_cond, edge_h, edge_mask
        if self.enable_equiv:
            node_h, node_mask = pyg_utils.to_dense_batch(data.x, data.batch, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N, node_nf], [B, N]
            pos, _ = pyg_utils.to_dense_batch(data.pos, data.batch, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N, 3]
            t_cond, _ = pyg_utils.to_dense_batch(data.t_cond, data.batch, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N]
            edge_h = pyg_utils.to_dense_adj(data.edge_index, data.batch, data.edge_attr, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N, N, edge_nf]
        else:
            x = torch.cat((data.x, data.pos, data.t_cond.reshape(-1, 1)), dim=-1)
            dense_x, node_mask = pyg_utils.to_dense_batch(x, data.batch, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N, node_nf], [B, N]
            node_h, pos, t_cond = dense_x[:, :, :-1], dense_x[:, :, -4:-1], dense_x[:, :, -1]
            edge_h = pyg_utils.to_dense_adj(data.edge_index, data.batch, data.edge_attr, batch_size=len(data['smiles']), max_num_nodes=data.max_seqlen)  # [B, N, N, edge_nf]

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) # [B, N, N]
        bs, n_nodes = node_mask.size()
        dense_index = edge_mask.nonzero(as_tuple=True)
        edge_h = edge_h[dense_index]
        edge_index, _ = pyg_utils.dense_to_sparse(edge_mask)

        # obtain conditional feature (noise level)
        time_emb = self.time_mlp(t_cond[:,0])  # [B, time_dim]

        if context is not None:
            condition = context.unsqueeze(-1)
            condition = self.cond_lin(self.cond_mlp(condition).reshape(bs, -1))
            time_emb = time_emb + condition

        node_time_emb = time_emb.unsqueeze(1).expand(bs, n_nodes, -1).reshape(bs*n_nodes, -1)
        edge_batch_id = torch.div(edge_index[0], n_nodes, rounding_mode='floor')
        edge_time_emb = time_emb[edge_batch_id]  # only keep valid edge

        # add distance to edge feature
        pos = pos.reshape(bs * n_nodes, -1)
        distance = coord2dist(pos, edge_index)
        dist_emb = self.dist_gbf(distance)
        edge_h = torch.cat([edge_h, dist_emb], dim=-1)

        if self.use_llm:
            if self.llm_cond:
                node_h = self.node_emb(node_h).reshape(bs * n_nodes, -1)
                node_cond = self.projector(lm_x).reshape(bs * n_nodes, -1)
                edge_cond = edge_time_emb
                node_cond = node_cond + node_time_emb
                # edge_cond = edge_cond + edge_time_emb
            else:
                if self.delta_train:
                    node_h = self.extended_node_emb(node_h, lm_x).reshape(bs * n_nodes, -1)
                else:
                    node_h = self.node_emb(torch.cat([node_h, lm_x], dim=-1)).reshape(bs * n_nodes, -1)
                node_cond = node_time_emb
                edge_cond = edge_time_emb
        else:
            node_h = self.node_emb(node_h).reshape(bs * n_nodes, -1)
            node_cond = node_time_emb
            edge_cond = edge_time_emb

        edge_h = self.edge_emb(edge_h)

        # run the equivariant block
        for i in range(self.n_blocks):
            node_h, edge_h, pos = self._modules[f'block_{i}'](pos, node_h, edge_h, edge_index, node_mask.reshape(-1, 1),
                                                              node_cond, edge_cond)

        if self.use_original_dgt:
            pos = remove_mean_with_mask(pos.reshape(bs, n_nodes, -1), node_mask.unsqueeze(-1))
            pos = pos.reshape(bs * n_nodes, -1)[node_mask.reshape(-1)]
            pred_noise = (data.pos - pos.detach() * data.alpha_t) / data.sigma_t
            return pos, pred_noise

        # last block for predicting pos noise
        if self.enable_equiv:
            dist_last = self.dist_gbf2(distance)
            pred_noise = self.pred_pos_noise(node_h, pos, edge_index, edge_h, dist_last, edge_time_emb)
        else:
            pred_noise = self.final_linear(node_h)

        # pyg dense to sparse
        pred_noise = pred_noise.reshape(bs * n_nodes, -1)[node_mask.reshape(-1)]
        if not self.disable_com:
            pred_noise = remove_mean(pred_noise, data.batch)
        pred_pos = (data.pos - pred_noise.detach() * data.sigma_t) / data.alpha_t

        if context is not None:
            atom_type, _ = pyg_utils.to_dense_batch(data.atom_type, data.batch) # [B, N]
            h0 = F.one_hot(atom_type, num_classes=5).float()

            charges, _ = pyg_utils.to_dense_batch(data.charge, data.batch) # [B, N]

            atom_mask = charges > 0 # [B, N]
            node_mask = atom_mask
            edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2) # [B, N, N]
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0).to(pos.device)
            edge_mask *= diag_mask

            included_species = torch.unique(charges, sorted=True)
            # if included_species[0] == 0:
            #     included_species = included_species[1:]
            one_hot = charges.unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
            nodes = one_hot.to(pos.device, torch.float32)
            edges_dic = {}
            def get_adj_matrix(n_nodes, batch_size, device):
                if n_nodes in edges_dic:
                    edges_dic_b = edges_dic[n_nodes]
                    if batch_size in edges_dic_b:
                        return edges_dic_b[batch_size]
                    else:
                        # get edges for a single sample
                        rows, cols = [], []
                        for batch_idx in range(batch_size):
                            for i in range(n_nodes):
                                for j in range(n_nodes):
                                    rows.append(i + batch_idx * n_nodes)
                                    cols.append(j + batch_idx * n_nodes)
                else:
                    edges_dic[n_nodes] = {}
                    return get_adj_matrix(n_nodes, batch_size, device)

                edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
                return edges
            full_edges = get_adj_matrix(n_nodes, bs, pos.device)
            return pred_pos, pred_noise, [nodes.reshape(bs * n_nodes, -1), pos, full_edges, None, node_mask.reshape(bs * n_nodes, -1), edge_mask.view(bs * n_nodes * n_nodes, 1), n_nodes]

        return pred_pos, pred_noise
