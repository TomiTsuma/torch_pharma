import torch
from torch import nn
from typing import Optional, Tuple
from torch_pharma.models.llama.llama import LlamaModel
from torch import functional as F

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