import torch
from torch import nn
from torch_pharma.models.decoder.decoder import LlamaDecoder
from torch_pharma.models.embedding.rotary_embedding import LlamaRotaryEmbedding
from torch_pharma.models.normalization.RMSNorm import LlamaRMSNorm


class LlamaModel(nn.Module):
    def __init__(
            self,
            vocab_size: int = 192,
            hidden_size: int = 2048,
            num_hidden_layers: int = 22,
            num_attention_heads: int = 8,
            num_key_value_heads: int = 2,
            head_dim: int = 256,
            intermediate_size: int = 5632,
            rms_norm_eps: float = 1e-5,
            max_position_embeddings: int = 2048,
            pad_token_id: int = 1
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.layers = nn.ModuleList([
            LlamaDecoder(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings
            )
            for _ in range(num_hidden_layers)
        ])

        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
        )

    def _make_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask.triu_(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        causal_mask = self._make_causal_mask(seq_len, input_ids.device, hidden_states.dtype)

        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * float("-inf")
            causal_mask = causal_mask + pad_mask

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask, position_ids)

            hidden_states = self.norm(hidden_states)
            return hidden_states 