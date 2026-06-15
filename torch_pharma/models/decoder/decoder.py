import torch
import torch.nn as nn
from typing import Optional
from torch_pharma.models.normalization.normalizer import LlamaRMSNorm
from torch_pharma.models.transformers.attention import LlamaAttention
from torch_pharma.models.mlp.llama_mlp import LlamaMLP

class LlamaDecoder(nn.Module):
    def __init__(
            self,
            hidden_size: int = 2048,
            num_attention_heads: int = 8,
            num_key_value_heads: int = 2,
            head_dim: int = 256,
            intermediate_size: int = 256,
            rms_norm_eps: float = 1e-5,
            max_position_embeddings: int = 2048
    ):
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings
        )
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states