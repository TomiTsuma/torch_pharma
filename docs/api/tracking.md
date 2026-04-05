# Activation Tracking Utilities

The `torch_pharma` package provides native, highly-scalable tools for interpreting internal module activations within Graph Neural Networks (GNNs).

The primary architecture uses a dynamic PyTorch hooking system invoked via decorators. It tracks Layer, Node, and Edge representations and naturally manages repeated evaluations typical of timestepped loops such as Diffusion sampling.

## The Tracking Decorator

::: torch_pharma.utils.tracking.decorators.track_gnn_activations

## Activation Store Memory Core

The core memory state representation uses `defaultdict(list)` internally to avoid overwriting timestepped layers. This ensures chronological tracing.

::: torch_pharma.utils.tracking.store.ActivationStore

## Telemetry Loggers (MLflow & WandB)

To natively dump chronological metrics automatically at the end of each model cycle without cluttering your core execution loops, use the Strategy Pattern loggers included natively.

::: torch_pharma.utils.tracking.loggers.wandb_logger.WandbActivationLogger

::: torch_pharma.utils.tracking.loggers.mlflow_logger.MlflowActivationLogger

::: torch_pharma.utils.tracking.loggers.base.ActivationLogger

## Usage Examples

Here is an example showing how to cleanly implement the tracker onto PyTorch Geometric models inside your codebase and stream data to external loggers seamlessly:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# 1. Import natively from the package!
from torch_pharma.utils.tracking import track_gnn_activations
from torch_pharma.utils.tracking.loggers import WandbActivationLogger

# 2. Decorate your neural network to intercept execution
@track_gnn_activations(
    track_layers=True,
    track_nodes=True,
    track_edges=True,
    verbose=False,
    # 3. Supply our out-of-the-box logger implementation 
    # (safely passes if wandb run hasn't triggered)
    loggers=[WandbActivationLogger(prefix="demo_gcn")]
)
class TwoLayerGCN(nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, out_ch)
        self.bn    = nn.BatchNorm1d(hidden)

    def forward(self, x, edge_index):
        # Tracker natively intercepts continuous representations
        x = F.relu(self.bn(self.conv1(x, edge_index)))
        x = self.conv2(x, edge_index)
        return x

# Initialization and Dummy Data
num_nodes, num_edges = 50, 120
x = torch.randn(num_nodes, 16)
edge_index = torch.randint(0, num_nodes, (2, num_edges))

model = TwoLayerGCN(16, 32, 7)

# Model Invocation (Hooks dynamically record and external loggers push automatically)
out = model(x, edge_index)

# Pull sequences natively!
print(model.activation_summary())

n1 = model.get_node_activation("conv1")
print(f"conv1 node states sequence length: {len(n1)} (last step shape: {n1[-1].shape})")
```
