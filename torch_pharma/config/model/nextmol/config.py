"""Configuration dataclasses for DMT (NExT-Mol) diffusion models."""

from dataclasses import dataclass, field


@dataclass
class DGTDiffusionConfig:
    """Configuration for DGTDiffusion (NExT-Mol DMT backbone)."""

    in_node_features: int = 44
    in_edge_features: int = 4
    hidden_size: int = 384
    n_blocks: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    enable_equiv: bool = False
    use_original_dgt: bool = False
    pred_noise: bool = True
    mlp_ratio: int = 4
    disable_com: bool = True
    trans_linear: bool = True
    disable_extra_gelu: bool = False
    not_pair_update: bool = False
    fuse_qkv: bool = False
    use_llm: bool = False
    llm_cond: bool = False
    delta_train: bool = False
    llm_hidden_size: int = 2048

    @classmethod
    def dmt_b(cls) -> "DGTDiffusionConfig":
        return cls(hidden_size=384, n_blocks=6)

    @classmethod
    def dmt_l(cls) -> "DGTDiffusionConfig":
        return cls(hidden_size=768, n_blocks=12)


@dataclass
class NextMolTrainingConfig:
    """Training hyperparameters for NExT-Mol workflows."""

    llm_model: str = "acharkq/MoLlama"
    llm_tune: str = "freeze"
    use_llm: bool = True
    llm_cond: bool = False
    use_llm_projector: bool = False
    llm_jk: str = "last"
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lm_loss: float = 0.0
    diff_loss: float = 1.0
    noise_scheduler: str = "cosine"
    continuous_beta_0: float = 0.1
    continuous_beta_1: float = 20.0
    sampling_steps: int = 100
    pos_std: float = 1.7226
    align_loss: bool = True
    translation_correction: bool = False
    align_prediction: bool = False
    reduce_node_mean: bool = True
    t_cond: str = "t"
    delta_train_epochs: int = 10
    use_self_att_proj: bool = False
    weight_decay: float = 0.05
    init_lr: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    max_epochs: int = 100
    precision: str = "bf16-mixed"
    dataset: str = "QM9-df"
    condition_property: str | None = None
