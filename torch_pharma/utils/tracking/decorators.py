"""
Role: The core user interface/injection mechanism.
Contents: The track_gnn_activations function. 
        It applies the wrapping logic and coordinates applying the hooks (from hooks.py) 
        with allocating a store (from store.py).
"""
import functools
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
 
import torch
import torch.nn as nn

from .store import ActivationStore
from .hooks import _make_forward_hook, _make_message_hook
from .loggers.base import ActivationLogger

def track_gnn_activations(
    track_layers: bool = True,
    track_nodes:  bool = True,
    track_edges:  bool = True,
    layer_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    verbose:      bool = False,
    methods_to_wrap: Union[str, List[str]] = ["forward", "sample", "optimize", "step", "training_step", "validation_step"],
    loggers:      Optional[List[ActivationLogger]] = None,
):
    """
    Class decorator that adds live activation tracking to any PyTorch GNN.
 
    After decoration the class gains:
        model.activations           -> ActivationStore  (populated after forward)
        model.get_layer_activation(name)  -> Tensor | None
        model.get_node_activation(name)   -> Tensor | None
        model.get_edge_activation(name)   -> Tensor | None
        model.activation_summary()        -> str
 
    Parameters
    ----------
    track_layers  : capture every sub-module's output tensor
    track_nodes   : alias; node features come from GNN conv outputs
    track_edges   : capture per-edge messages (requires torch_geometric)
    layer_filter  : optional callable(name, module) -> bool to restrict which
                    layers are tracked (return True to track)
    verbose       : print the summary after every forward pass
 
    Examples
    --------
    Basic usage::
 
        @track_gnn_activations()
        class MyGCN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(16, 32)
                self.conv2 = GCNConv(32, 64)
 
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index)
                return x
 
        model = MyGCN()
        out = model(x, edge_index)
 
        print(model.activation_summary())
        node_acts = model.get_node_activation('conv1')
 
    Filter to conv layers only::
 
        @track_gnn_activations(
            layer_filter=lambda name, mod: 'conv' in name
        )
        class MyGCN(torch.nn.Module):
            ...
    """
 
    def decorator(cls):
 
        original_init = cls.__init__
 
        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.activations = ActivationStore()
            self._tracker_hooks_registered = False
 
        cls.__init__ = new_init
 
        # ------------------------------------------------------------------
        # Hook registration (deferred until first forward so all children
        # have been built even in lazy-init models)
        # ------------------------------------------------------------------
 
        def _register_hooks(self):
            if self._tracker_hooks_registered:
                return
            store = self.activations
 
            try:
                from torch_geometric.nn import MessagePassing
                has_pyg = True
            except ImportError:
                has_pyg = False
                if track_edges:
                    warnings.warn(
                        "torch_geometric not found – edge tracking disabled.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
 
            for name, module in self.named_modules():
                if not name:           # skip the top-level model itself
                    continue
                if layer_filter and not layer_filter(name, module):
                    continue
 
                is_conv = has_pyg and isinstance(module, MessagePassing)
 
                # Node / layer activations via forward hook
                if track_layers or (track_nodes and is_conv):
                    kind = "node" if is_conv else "layer"
                    h = module.register_forward_hook(
                        _make_forward_hook(store, name, kind)
                    )
                    store._hooks.append(h)
 
                # Edge activations via message-method patch
                if track_edges and is_conv and has_pyg:
                    if not hasattr(module, "_original_message"):
                        module._original_message = module.message
 
                        def _patched(mod, *a, **kw):
                            out = mod._original_message(*a, **kw)
                            if isinstance(out, torch.Tensor):
                                store.register(
                                    f"{name}.messages", out, kind="edge"
                                )
                            return out
 
                        # Bind to the specific module instance
                        import types
                        module.message = types.MethodType(
                            lambda mod, *a, **kw: _patched(mod, *a, **kw),
                            module,
                        )
 
            self._tracker_hooks_registered = True
 
        # ------------------------------------------------------------------
        # Wrap methods
        # ------------------------------------------------------------------
        if isinstance(methods_to_wrap, str):
            methods = [methods_to_wrap]
        else:
            methods = methods_to_wrap

        for method_name in methods:
            if not hasattr(cls, method_name):
                continue
            original_method = getattr(cls, method_name)

            @functools.wraps(original_method)
            def new_method(self, *args, _original_method=original_method, **kwargs):
                _register_hooks(self)
                self.activations.clear()
                result = _original_method(self, *args, **kwargs)
                if verbose:
                    print(self.activations.summary())
                    
                # Invoke external loggers
                if loggers:
                    # Provide a global step if it exists to align metrics tracking properly
                    step = getattr(self, "global_step", None)
                    for logger in loggers:
                        logger.log(self.activations, step=step)
                        
                return result

            setattr(cls, method_name, new_method)
 
        # ------------------------------------------------------------------
        # Public API added to the class
        # ------------------------------------------------------------------
 
        def get_layer_activation(self, name: str) -> Optional[List[torch.Tensor]]:
            """Return the stored list of layer activation tensors for *name*, or None."""
            return self.activations.layers.get(name)
 
        def get_node_activation(self, name: str) -> Optional[List[torch.Tensor]]:
            """Return the stored list of node-feature tensors for the named GNN conv layer."""
            # Node features may land in .nodes (conv layers) or .layers
            result = self.activations.nodes.get(name)
            if result is None:
                result = self.activations.layers.get(name)
            return result
 
        def get_edge_activation(self, name: str) -> Optional[List[torch.Tensor]]:
            """Return the stored list of edge-message tensors for the named GNN conv layer."""
            key = f"{name}.messages"
            return self.activations.edges.get(key)
 
        def activation_summary(self) -> str:
            """Pretty-print the activation summary."""
            return self.activations.summary()
 
        def get_activation_stats(self, name: str) -> Optional[List[Dict[str, float]]]:
            """Return the statistics dict sequence for any tracked name."""
            return self.activations.stats.get(name)
 
        def remove_activation_hooks(self):
            """Detach all hooks. Call this to restore the original model."""
            self.activations.remove_hooks()
            self._tracker_hooks_registered = False
 
        cls.get_layer_activation    = get_layer_activation
        cls.get_node_activation     = get_node_activation
        cls.get_edge_activation     = get_edge_activation
        cls.activation_summary      = activation_summary
        cls.get_activation_stats    = get_activation_stats
        cls.remove_activation_hooks = remove_activation_hooks
 
        return cls
 
    return decorator
 
 