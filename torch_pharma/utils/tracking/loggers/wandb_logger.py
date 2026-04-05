import warnings
from typing import Optional
from torch_pharma.utils.tracking.store import ActivationStore
from torch_pharma.utils.tracking.loggers.base import ActivationLogger

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

class WandbActivationLogger(ActivationLogger):
    """
    Logs ActivationStore statistics to Weights & Biases.
    """
    def __init__(self, prefix: str = "activations", log_raw_tensors: bool = False, **wandb_init_kwargs):
        """
        Args:
            prefix: String prefix to group all activation metrics in wandb panel.
            log_raw_tensors: If True, dumps raw tensor weights into wandb.Histogram
            **wandb_init_kwargs: Auto-initializes wandb.init() locally if provided.
        """
        if not HAS_WANDB:
            warnings.warn("wandb is not installed. WandbActivationLogger will silently pass.", ImportWarning)
        self.prefix = prefix
        self.log_raw_tensors = log_raw_tensors
        
        # Self-contained initialization
        if HAS_WANDB and wandb_init_kwargs:
            if wandb.run is None:
                wandb.init(**wandb_init_kwargs)
        
    def log(self, store: ActivationStore, step: Optional[int] = None):
        if not HAS_WANDB or wandb.run is None:
            return
            
        max_seq_len = max([len(seq) for seq in store.stats.values()] + [0])
        base_step = (step * max_seq_len) if step is not None else 0
        
        # We loop chronologically so WandB renders a progressive line chart
        for seq_idx in range(max_seq_len):
            metrics = {}
            for layer_name, stats_sequence in store.stats.items():
                if seq_idx >= len(stats_sequence):
                    continue
                    
                stats = stats_sequence[seq_idx]
                
                bucket = "LAYERS"
                if layer_name in store.nodes:
                    bucket = "NODES"
                elif layer_name in store.edges:
                    bucket = "EDGES"
                    
                for stat_key, stat_val in stats.items():
                    if stat_key == "shape":
                        continue
                    metrics[f"{self.prefix}/{bucket}/{layer_name}/{stat_key}"] = stat_val
                    
                # Optionally push the raw progressive activations internally
                if self.log_raw_tensors:
                    tensor_seq = getattr(store, bucket.lower())
                    if layer_name in tensor_seq and seq_idx < len(tensor_seq[layer_name]):
                        raw_tensor = tensor_seq[layer_name][seq_idx]
                        metrics[f"{self.prefix}/{bucket}/{layer_name}/raw"] = wandb.Histogram(raw_tensor.float().numpy())
            
            # Submits the metric slice at this sequence frame
            wandb.log(metrics, step=base_step + seq_idx)
