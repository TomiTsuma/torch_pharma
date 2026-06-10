import torch
import math
from torch.nn import functional as F
from torch import nn
from torch_pharma.models.embedding.rotary_embedding import LlamaRotaryEmbedding, apply_rotary_pos_emb
from typing import Optional

class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 8,       # query heads
        num_key_value_heads: int = 2,        # KV heads (GQA)
        head_dim: int = 256,
        max_position_embeddings: int = 2048,
        rope_base: int = 10000,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads  # = 4

        assert hidden_size == num_attention_heads * head_dim, \
            "hidden_size must equal num_attention_heads * head_dim"

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
        )

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Expand KV heads to match query head count for GQA.
        Input:  (batch, num_kv_heads, seq_len, head_dim)
        Output: (batch, num_attention_heads, seq_len, head_dim)
        """
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,                    # (B, T, D)
        attention_mask: Optional[torch.Tensor] = None,  # (B, 1, T, T) causal mask
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)  # (B, T, num_heads * head_dim)
        key_states   = self.k_proj(hidden_states)  # (B, T, num_kv_heads * head_dim)
        value_states = self.v_proj(hidden_states)  # (B, T, num_kv_heads * head_dim)

        # Reshape to (B, num_heads, T, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(query_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand KV heads to match query heads (GQA repeat)
        key_states   = self._repeat_kv(key_states,   self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / scale
        # attn_weights: (B, num_heads, T, T)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask  # mask is -inf for future positions

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)  # (B, num_heads, T, head_dim)

        # Merge heads and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)  # (B, T, D)

        return attn_output
