#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_sum, scatter_min

from torch_pharma.data.bioparse import VOCAB, const
from utils.nn_utils import SinusoidalPositionEmbedding, expand_like, SinusoidalTimeEmbeddings, graph_to_batch_nx
from utils.gnn_utils import length_to_batch_id, std_conserve_scatter_mean, scatter_sort
import utils.register as R
from utils.oom_decorator import oom_decorator

from .map import block_to_atom_map
from .tools import _avoid_clash

from ..modules.GET.tools import fully_connect_edges, knn_edges
from ..modules.nn import BlockEmbedding, MLP
from ..modules.create_net import create_net
from ..modules.metrics import batch_accu
