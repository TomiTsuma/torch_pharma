"""
Role: PyTorch internals and extraction logic.
Contents: _make_forward_hook and _make_message_hook. 
        Let this module depend on store.py (to write to the ActivationStore). 
        By keeping torch_geometric monkey-patching in here, you isolate the "hacky" 
           logic from the pure classes.
"""

import functools
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
 
import torch
import torch.nn as nn
from torch_pharma.utils.tracking.store import ActivationStore

def _make_forward_hook(store: ActivationStore, name: str, kind: str):
    """Return a standard forward hook that writes the *output* to the store."""
    def hook(module, input, output):
        # output may be a tuple (x, edge_index, ...) for some GNN convs
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(tensor, torch.Tensor):
            store.register(name, tensor, kind=kind)
    return hook
 
 
def _make_message_hook(store: ActivationStore, layer_name: str):
    """
    Monkey-patch the `message` method of a MessagePassing layer so that we
    can capture the per-edge messages (edge activations).
    """
    try:
        from torch_geometric.nn import MessagePassing
    except ImportError as e:
        raise e
 
    def patched_message(module, *args, **kwargs):
        out = module._original_message(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            store.register(f"{layer_name}.messages", out, kind="edge")
        return out
 
    return patched_message