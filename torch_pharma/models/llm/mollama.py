"""MoLlama HuggingFace integration for NExT-Mol stage-1 / conditioning."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from torch_pharma.utils.device import get_half_precision_dtype
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


def set_embed_tokens_trainable(model) -> None:
    for name, param in model.named_parameters():
        if "embed_tokens" in name:
            param.requires_grad = True


def load_mollama(
    model_name: str = "acharkq/MoLlama",
    llm_tune: str = "freeze",
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    tune_embedding: bool = False,
    use_flash_attention: bool = False,
):
    """Load MoLlama from HuggingFace with optional PEFT LoRA."""
    from transformers import AutoModelForCausalLM

    log.info("Loading MoLlama from %s (llm_tune=%s)", model_name, llm_tune)
    kwargs = {"torch_dtype": get_half_precision_dtype()}
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if llm_tune == "freeze":
        for param in model.parameters():
            param.requires_grad = False
    elif llm_tune == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif llm_tune in ("lora", "mid_lora"):
        from peft import LoraConfig, get_peft_model

        target = (
            ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            if llm_tune == "mid_lora"
            else ["q_proj", "v_proj"]
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target,
        )
        model = get_peft_model(model, lora_config)
        if tune_embedding:
            set_embed_tokens_trainable(model)
    else:
        raise ValueError(f"Unknown llm_tune: {llm_tune}")

    log.debug("MoLlama loaded with llm_tune=%s", llm_tune)
    return model


def init_mollama_tokenizer(model_name: str = "acharkq/MoLlama"):
    from transformers import AutoTokenizer

    log.info("Loading MoLlama tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
    return tokenizer


class MoLlamaConditioning(nn.Module):
    """Wraps MoLlama forward for property-conditioned and plain SELFIES encoding."""

    def __init__(self, llm_model, hidden_size: int):
        super().__init__()
        self.llm_model = llm_model
        self.condition_mlp = nn.Sequential(
            nn.Linear(1, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, 4 * hidden_size),
        )

    def create_condition_prompt(self, context: torch.Tensor) -> torch.Tensor:
        context = context.unsqueeze(1)
        return self.condition_mlp(context)

    def forward(
        self,
        selfies_batch,
        context: Optional[torch.Tensor] = None,
        llm_tune: str = "freeze",
    ):
        if context is not None:
            token_embeds = self.llm_model.get_input_embeddings()(selfies_batch.input_ids)
            condition_embeds = self.create_condition_prompt(context)
            inputs_embeds = torch.cat([condition_embeds, token_embeds], dim=1)
            soft_prompt = torch.ones(
                (selfies_batch.attention_mask.shape[0], 4),
                device=selfies_batch.attention_mask.device,
            )
            attention_mask = torch.cat([soft_prompt, selfies_batch.attention_mask], dim=1)
            ignore_prefix = torch.full(
                (selfies_batch.input_ids.shape[0], 4),
                -100,
                device=selfies_batch.input_ids.device,
            )
            target = torch.cat([ignore_prefix, selfies_batch.input_ids], dim=1)
            target = target.masked_fill(~attention_mask.bool(), -100)
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=target,
                output_hidden_states=True,
            )
            hidden_states = [h[:, 4:] for h in outputs.hidden_states]
        else:
            targets = selfies_batch.input_ids.masked_fill(
                ~selfies_batch.attention_mask.bool(), -100
            )
            outputs = self.llm_model(
                input_ids=selfies_batch.input_ids,
                attention_mask=selfies_batch.attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states

        if llm_tune == "freeze":
            hidden_states = tuple(h.detach() for h in hidden_states)
            lm_loss = torch.tensor(0.0, device=selfies_batch.input_ids.device)
        else:
            lm_loss = outputs.loss

        return hidden_states, lm_loss
