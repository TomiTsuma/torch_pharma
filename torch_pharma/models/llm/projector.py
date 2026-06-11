"""LLM-to-atom projection for NExT-Mol cross-modal conditioning."""

import torch
from torch import nn


class LLMProjector(nn.Module):
    """Maps LLM hidden states to per-atom embeddings via rdmol2selfies alignment."""

    def __init__(
        self,
        in_dim: int,
        hidden_size: int,
        llm_jk: str = "last",
        use_self_att_proj: bool = False,
        llm_num_layers: int = 22,
    ):
        super().__init__()
        self.llm_jk = llm_jk
        self.use_self_att_proj = use_self_att_proj
        if self.llm_jk == "mean":
            self.mean_weight = nn.Parameter(torch.zeros(1, llm_num_layers))
            self.mean_ln = nn.LayerNorm(in_dim)

        if self.use_self_att_proj:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                batch_first=True,
                norm_first=True,
                dropout=0.0,
            )
            self.self_att_proj = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.linear_proj = nn.Linear(in_dim, hidden_size)

    def forward(self, hidden_states, rdmol2selfies, selfies_batch):
        if self.llm_jk == "last":
            lm_embeds = hidden_states[-1]
        elif self.llm_jk == "mean":
            lm_embeds = torch.stack(hidden_states[1:], dim=2)
            lm_embeds = (self.mean_weight.softmax(dim=-1) @ lm_embeds).squeeze(2)
            lm_embeds = self.mean_ln(lm_embeds)
        else:
            raise NotImplementedError(f"llm_jk={self.llm_jk}")

        if self.use_self_att_proj:
            lm_embeds = self.self_att_proj(
                self.linear_proj(lm_embeds),
                src_key_padding_mask=~selfies_batch.attention_mask.bool(),
            )

        lm_x = torch.bmm(rdmol2selfies.to(lm_embeds.dtype), lm_embeds)
        norm = torch.clamp(torch.sum(rdmol2selfies, dim=-1, keepdim=True), min=1)
        return lm_x / norm
