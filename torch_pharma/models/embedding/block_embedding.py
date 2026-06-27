import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_sum


# embedding of blocks (for proteins, it is residue).
class BlockEmbedding(nn.Module):
    '''
    [atom embedding + block embedding]
    '''
    def __init__(self, num_block_type, num_atom_type, embed_size):
        super().__init__()
        self.block_embedding = nn.Embedding(num_block_type, embed_size)
        self.atom_embedding = nn.Embedding(num_atom_type, embed_size)
    
    def forward(self, S, A, block_id):
        '''
        :param S: [Nb], block (residue) types
        :param A: [Nu], unit (atom) types
        :param block_id: [Nu], block id of each unit
        '''
        atom_embed = self.atom_embedding(A)
        block_embed = self.block_embedding(S[block_id])
        return atom_embed + block_embed
    

class AtomTopoEmbedding(nn.Module):
    '''
    TODO: atom embedding based on 2D chemical interactions
    '''
    def __init__(self, num_atom_type, num_bond_type, embed_size) -> None:
        super().__init__()

