import warnings
from typing import Optional
from torch_pharma.utils.tracking.store import ActivationStore
from torch_pharma.utils.tracking.loggers.base import ActivationLogger

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

class MlflowActivationLogger(ActivationLogger):
    """
    Logs ActivationStore statistics to MLflow.
    """
    def __init__(self, prefix: str = "activations", tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None, run_name: Optional[str] = None, **mlflow_kwargs):
        """
        Args:
            prefix: String prefix for mlflow metric names.
            tracking_uri: Sets the tracking URI for MLflow prior to run execution.
            experiment_name: Safely executes mlflow.set_experiment if passed.
            run_name: Auto-initializes an mlflow active run locally if passed.
        """
        if not HAS_MLFLOW:
            warnings.warn("mlflow is not installed. MlflowActivationLogger will silently pass.", ImportWarning)
        self.prefix = prefix
        
        # Self-contained initialization
        if HAS_MLFLOW:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            if run_name and mlflow.active_run() is None:
                mlflow.start_run(run_name=run_name, **mlflow_kwargs)
        
    def log(self, store: ActivationStore, step: Optional[int] = None):
        if not HAS_MLFLOW or mlflow.active_run() is None:
            return
            
        max_seq_len = max([len(seq) for seq in store.stats.values()] + [0])
        # If tracking across outer epochs, offset them so overlapping iterations string together cleanly
        base_step = (step * max_seq_len) if step is not None else 0
        
        # Iterate horizontally across the time-dimension to push a line curve naturally
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
                    metrics[f"{self.prefix}_{bucket}_{layer_name}_{stat_key}"] = stat_val
            
            # Submit to MLflow
            mlflow.log_metrics(metrics, step=base_step + seq_idx)
