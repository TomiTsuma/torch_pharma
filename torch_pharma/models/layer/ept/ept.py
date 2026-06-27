from torch import nn
from torch_pharma.models.layer.wrapper import SubLayerWrapper
from torch_pharma.models.layer.feed_forward import GVPFFNLayer
from torch_pharma.models.transformers.attention import SelfAttnLayer
class EPTLayer(nn.Module):
    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            layer_idx=-1,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            efficient = False,
            vector_act = 'none',
            attn_bias = True
        ):
        super(EPTLayer, self).__init__()
        self.attn_layer = SubLayerWrapper(
            SelfAttnLayer(d_hidden, n_heads, layer_idx, efficient, attn_bias = attn_bias),
            d_hidden,
            layer_norm,
            residual
        )
        self.ffn_layer = SubLayerWrapper(
            GVPFFNLayer(d_hidden, d_ffn, act_fn, vector_act = vector_act),
            d_hidden,
            layer_norm,
            residual
        )
        self.layer_idx = layer_idx

    def forward(self, H, V, cached_info=None):

        H, V = self.attn_layer(H, V, cached_info=cached_info)
        H, V = self.ffn_layer(H, V)

        return H, V

  
import torch
from torch_pharma.models.transformers.mol_transformer import Transformer

if __name__ == '__main__':
    d_hidden = 64
    d_ffn = 16
    d_edge = 16
    n_rbf = 16
    n_heads = 4
    n_layers = 3
    device = torch.device('cuda:0')

    # d_hidden, d_ffn, n_heads, n_layers, n_rbf, d_edge, cutoff=7.0, act_fn=nn.SiLU(), layer_norm = 'pre', residual = True, sparse_k=3, svd_k=128

    model = Transformer(d_hidden, d_ffn, n_heads, n_layers, n_rbf, d_edge=d_edge, use_ieconv=True, use_edge_feat=True, efficient_ieconv=True, ieconv_share_edge_feat=False)
    model.to(device)
    model.eval()
    
    block_id = torch.tensor([0,0,1,1,1,1,2,2,2,3,4,4,5,6,6,6,6,7,7], dtype=torch.long).to(device)
    batch_id = torch.tensor([0,0,0,0,0,1,1,1], dtype=torch.long).to(device)
    src_dst = torch.tensor([[0,1], [2,3], [1,3], [2,4], [3, 0], [3, 3], [5,7], [7,6], [5,6], [6,7]], dtype=torch.long).to(device)
    src_dst = src_dst.T
    edge_attr = torch.randn(len(src_dst[0]), d_edge).to(device)
    n_unit = block_id.shape[0]

    H = torch.randn(n_unit, d_hidden, device=device)
    Z = torch.randn(n_unit, 3, device=device)

    H1, V1 = model(H, Z, block_id, batch_id, src_dst, edge_attr)

    # random rotaion matrix
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q1, t1 = U.mm(V), torch.randn(3, device=device)
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q2, t2 = U.mm(V), torch.randn(3, device=device)

    unit_batch_id = batch_id[block_id]
    Z[unit_batch_id == 0] = torch.matmul(Z[unit_batch_id == 0], Q1) + t1
    Z[unit_batch_id == 1] = torch.matmul(Z[unit_batch_id == 1], Q2) + t2
    # Z = torch.matmul(Z, Q) + t

    H2, V2 = model(H, Z, block_id, batch_id, src_dst, edge_attr)

    print(f'invariant feature: {torch.abs(H1 - H2).sum()}')
    V1[unit_batch_id == 0] = torch.einsum('nih, ij -> njh', V1[unit_batch_id == 0], Q1)
    V1[unit_batch_id == 1] = torch.einsum('nih, ij -> njh', V1[unit_batch_id == 1], Q2)
    print(f'equivariant feature: {torch.abs(V1 - V2).sum()}')