import torch
import torch.nn.functional as F
from functools import partial
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from torch_geometric.data import Batch
from torch import nn
from torch_scatter import scatter
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

patch_typeguard()




from torch_pharma.features.geometry import (
    ScalarVector,
    centralize,
    decentralize,
    localize,
    scalarize,
    vectorize
)
from torch_pharma.utils.math import (
    safe_norm,
    is_identity
)

from torch_pharma.modules.activation import get_nonlinearity