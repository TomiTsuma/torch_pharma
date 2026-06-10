# MoLLaMA Architecture — PyTorch Implementation Guide

This document provides complete, module-by-module instructions for building the MoLLaMA model in PyTorch. MoLLaMA is a 960M-parameter autoregressive causal language model pre-trained on SELFIES molecular sequences. Its architecture is a compact Llama-style transformer.

---

## Model Summary

| Hyperparameter        | Value   |
|-----------------------|---------|
| Hidden dimension      | 2048    |
| Number of layers      | 22      |
| Vocabulary size       | 192     |
| Query heads           | 8       |
| KV heads              | 2 (GQA) |
| Head dimension        | 256     |
| Intermediate size     | 5632    |
| Activation            | SiLU (SwiGLU) |
| Normalization         | RMSNorm (eps=1e-5) |
| Position encoding     | Rotary (RoPE) |
| Padding token index   | 1       |

---

## Required Imports

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
```

---

## Module 1: LlamaRMSNorm

Root Mean Square Layer Normalization. Replaces LayerNorm — no mean subtraction, no bias.

**Formula:** `output = x / RMS(x) * weight`  
where `RMS(x) = sqrt(mean(x^2) + eps)`

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

**Shape:** Input and output are both `(batch, seq_len, hidden_size)`.  
`self.weight` is a learnable scale vector of shape `(hidden_size,)` initialized to ones.

---

## Module 2: LlamaRotaryEmbedding

Rotary Position Embedding (RoPE). Encodes relative position information by rotating query and key vectors using sine/cosine frequencies. No learnable parameters.

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute inverse frequencies: shape (dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _compute_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        # positions: shape (seq_len,)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # outer product → (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)
        # concatenate to get (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, seq_len: int):
        cos, sin = self._compute_cos_sin(seq_len, x.device, x.dtype)
        # Return (1, 1, seq_len, dim) for broadcasting
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to implement RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply RoPE to query and key tensors.
    q, k: (batch, num_heads, seq_len, head_dim)
    cos, sin: (1, 1, seq_len, head_dim)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**Key points:**
- `dim` here is the per-head dimension (256 for MoLLaMA).
- RoPE is applied after projecting to Q and K, before the attention score computation.
- `rotate_half` splits the last dimension in half and performs `[-x2, x1]` rotation.

---

## Module 3: LlamaAttention (Grouped Query Attention)

Multi-head attention with Grouped Query Attention (GQA). Queries use 8 heads; keys and values use 2 heads. Head dimension is 256 for all groups.

**Dimension accounting:**
- `q_proj`: `(2048, 2048)` → 8 heads × 256 = 2048 ✓
- `k_proj`: `(2048, 256)` → 2 heads × 128... wait — the architecture shows `k_proj` out=256, meaning 1 head of 256 per KV group shared across 4 query heads each.
- `v_proj`: `(2048, 256)` → same as k_proj
- `o_proj`: `(2048, 2048)` → projects concatenated query outputs back

```python
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
```

**GQA note:** With `num_kv_heads=2` and `num_heads=8`, each KV head is shared by 4 query heads. `_repeat_kv` replicates each KV head 4 times before the attention computation.

---

## Module 4: LlamaMLP (SwiGLU Feed-Forward Network)

A gated feed-forward block. Uses SwiGLU activation: `SiLU(gate_proj(x)) * up_proj(x)`.

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int = 2048, intermediate_size: int = 5632):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn    = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: element-wise gate before down-projection
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**Shape flow:**
1. `x`: `(B, T, 2048)`
2. `gate_proj(x)` and `up_proj(x)`: `(B, T, 5632)`
3. Element-wise product after SiLU: `(B, T, 5632)`
4. `down_proj(...)`: `(B, T, 2048)`

---

## Module 5: LlamaDecoderLayer

One full transformer decoder block. Pre-norm architecture (RMSNorm applied before attention and MLP, residual added after).

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        intermediate_size: int = 5632,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
        )
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.input_layernorm        = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # --- Self-attention sublayer (pre-norm + residual) ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # --- MLP sublayer (pre-norm + residual) ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

---

## Module 6: LlamaModel (Trunk)

The core transformer: token embedding → 22 decoder layers → final RMSNorm.

```python
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
        pad_token_id: int = 1,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ])

        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
        )

    def _make_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Build an upper-triangular causal mask filled with -inf above the diagonal.
        Shape: (1, 1, seq_len, seq_len)
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.LongTensor,                    # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) padding mask, 1=keep 0=mask
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)    # (B, T, D)

        # Build causal attention mask
        causal_mask = self._make_causal_mask(seq_len, input_ids.device, hidden_states.dtype)

        # Optionally incorporate padding mask
        if attention_mask is not None:
            # Convert (B, T) → (B, 1, 1, T) and set padded positions to -inf
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * float("-inf")
            causal_mask = causal_mask + pad_mask

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Pass through all decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask, position_ids)

        # Final normalization
        hidden_states = self.norm(hidden_states)        # (B, T, D)
        return hidden_states
```

---

## Module 7: LlamaForCausalLM (Full Model)

Wraps `LlamaModel` with a language model head. Uses weight tying between `embed_tokens` and `lm_head` if desired (common in Llama models).

```python
class LlamaForCausalLM(nn.Module):
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
        pad_token_id: int = 1,
    ):
        super().__init__()
        self.model = LlamaModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying: share embedding and lm_head weights (optional but standard in Llama)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            loss   : cross-entropy loss if labels provided, else None
            logits : (B, T, vocab_size)
        """
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)    # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Shift for causal LM: predict token t+1 from token t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.model.embed_tokens.num_embeddings),
                shift_labels.view(-1),
                ignore_index=-100,  # standard convention for masked positions
            )

        return loss, logits
```

---

## Instantiation and Verification

```python
def build_mollama() -> LlamaForCausalLM:
    model = LlamaForCausalLM(
        vocab_size=192,
        hidden_size=2048,
        num_hidden_layers=22,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        intermediate_size=5632,
        rms_norm_eps=1e-5,
        max_position_embeddings=2048,
        pad_token_id=1,
    )
    return model


if __name__ == "__main__":
    model = build_mollama()

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Forward pass sanity check
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 192, (batch_size, seq_len))
    labels    = torch.randint(0, 192, (batch_size, seq_len))
    labels[labels == 1] = -100   # mask padding

    loss, logits = model(input_ids, labels=labels)
    print(f"Loss:    {loss.item():.4f}")
    print(f"Logits:  {logits.shape}")   # (2, 64, 192)
    print(f"Arch:\n{model}")
```

---

## Parameter Budget Breakdown

| Component                                       | Parameters per layer | × Layers | Total        |
|-------------------------------------------------|----------------------|-----------|--------------|
| `embed_tokens` (192 × 2048)                     | 393,216              | 1         | 393,216      |
| `q_proj` (2048 × 2048)                          | 4,194,304            | 22        | 92,274,688   |
| `k_proj` (2048 × 256)                           | 524,288              | 22        | 11,534,336   |
| `v_proj` (2048 × 256)                           | 524,288              | 22        | 11,534,336   |
| `o_proj` (2048 × 2048)                          | 4,194,304            | 22        | 92,274,688   |
| `gate_proj` (2048 × 5632)                       | 11,534,336           | 22        | 253,755,392  |
| `up_proj` (2048 × 5632)                         | 11,534,336           | 22        | 253,755,392  |
| `down_proj` (5632 × 2048)                       | 11,534,336           | 22        | 253,755,392  |
| `input_layernorm` + `post_attn_layernorm` (×2)  | 4,096                | 22        | 180,224      |
| `norm` (final RMSNorm)                          | 2,048                | 1         | 2,048        |
| `lm_head` (tied with embed_tokens)              | 0 (shared)           | 1         | 0            |
| **Total**                                       |                      |           | **~969M**    |

---

## Critical Implementation Notes

### 1. No biases anywhere
Every `nn.Linear` uses `bias=False`. This is standard for Llama-family models and critical for weight tying to work correctly.

### 2. RMSNorm, not LayerNorm
Use the custom `LlamaRMSNorm` above. Do **not** use `nn.LayerNorm` — LayerNorm subtracts the mean and has a bias parameter; RMSNorm does neither.

### 3. Pre-norm residual connections
Normalization is applied **before** the sublayer, and the residual is added **after**. The pattern in each decoder layer is:
```
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

### 4. RoPE is applied per attention layer, not once globally
Each `LlamaAttention` module contains its own `LlamaRotaryEmbedding`. The `LlamaModel.rotary_emb` attribute in the original printout is present but the actual embedding computation happens inside each attention block.

### 5. GQA repeat_kv
With `num_kv_heads=2` and `num_heads=8`, `num_key_value_groups=4`. The `_repeat_kv` function must interleave-expand, not just concatenate, so that head group boundaries align correctly.

### 6. Causal mask dtype
Always cast the causal mask to the same dtype as `hidden_states` (e.g. `float16` during mixed-precision training). Adding `-inf` in `float32` then converting causes no issues, but a dtype mismatch in the mask addition will cause errors.

### 7. Weight tying
`lm_head.weight = model.embed_tokens.weight` makes both share the same underlying tensor. This reduces parameters by `192 × 2048 = 393,216` and is the standard Llama configuration. If you do **not** want tying, simply initialize `lm_head` without this assignment.

### 8. Causal LM loss shift
When computing cross-entropy, shift logits left by one and labels right by one:
- `logits[..., :-1, :]` predicts positions 1 through T
- `labels[..., 1:]` are the targets at positions 1 through T
Use `ignore_index=-100` to mask padding tokens in the loss.

---

## SELFIES Tokenization Context

MoLLaMA operates on SELFIES strings with a vocabulary of 192 tokens. When using this model:

- Token index `1` is the padding token (`padding_idx=1` in `nn.Embedding`).
- SELFIES tokens map to indices `0–191`.
- The model's `embed_tokens` layer will zero-out the gradient for the padding embedding automatically due to `padding_idx=1`.

---

## Optional: Flash Attention Integration

For training efficiency, replace the manual scaled dot-product attention block in `LlamaAttention.forward` with PyTorch's built-in implementation:

```python
# Replace the manual attention computation with:
attn_output = F.scaled_dot_product_attention(
    query_states,
    key_states,
    value_states,
    attn_mask=attention_mask,   # pass None for causal; set is_causal=True instead
    is_causal=True,
)
```

This uses Flash Attention under the hood on supported hardware (CUDA with PyTorch >= 2.0) and avoids materializing the full `(B, H, T, T)` attention matrix.