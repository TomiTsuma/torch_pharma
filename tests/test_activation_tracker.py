"""
Demo mapping for the new GNN Activation Tracker natively embedded in torch_pharma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import natively from the package!
from torch_pharma.utils.tracking import track_gnn_activations
from torch_pharma.utils.tracking.loggers import WandbActivationLogger, MlflowActivationLogger

if __name__ == "__main__":
    # Try to use torch_geometric if available
    try:
        from torch_geometric.nn import GCNConv, GATConv
        HAS_PYG = True
    except ImportError:
        HAS_PYG = False
        print("torch_geometric not installed – running plain-PyTorch demo.\n")

    if HAS_PYG:
        # ----------------------------------------------------------------
        # Example 1: GCN with full tracking
        # ----------------------------------------------------------------
        @track_gnn_activations(
            track_layers=True,
            track_nodes=True,
            track_edges=True,
            verbose=False,
            # Supplying out-of-the box implementation loggers! 
            # They will natively self-initialize because we pass tracking credentials internally.
            loggers=[
                WandbActivationLogger(
                    prefix="demo_gcn", 
                    log_raw_tensors=True, # enables tracking histograms natively 
                    project="torch-pharma-tracker-demo", 
                    name="demo-run"
                ),
                MlflowActivationLogger(
                    prefix="demo_gcn", 
                    tracking_uri="http://100.101.70.112:5000", # Route data to explicit loc
                    experiment_name="torch-pharma-tracker", 
                    run_name="demo-run"
                )
            ]
        )
        class TwoLayerGCN(nn.Module):
            def __init__(self, in_ch, hidden, out_ch):
                super().__init__()
                self.conv1 = GCNConv(in_ch, hidden)
                self.conv2 = GCNConv(hidden, out_ch)
                self.bn    = nn.BatchNorm1d(hidden)

            def forward(self, x, edge_index):
                x = F.relu(self.bn(self.conv1(x, edge_index)))
                x = self.conv2(x, edge_index)
                return x

        num_nodes, num_edges = 50, 120
        x          = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        print("Executing Example 1: TwoLayerGCN Tracker")
        model = TwoLayerGCN(16, 32, 7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        print("Simulating 25 continuous epochs to flush sequential tracking curves...")
        for epoch in range(25):
            model.global_step = epoch   # The decorators inherently watch this!
            out = model(x, edge_index)  # Hooks automatically submit the states progressively
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(model.activation_summary())

        # NOTE: Using the new tracked sequence mechanisms where everything is wrapped natively in Lists!
        n1 = model.get_node_activation("conv1")
        print(f"\n[Validation] conv1 node activations sequence length: {len(n1)} (last eval shape: {n1[-1].shape})")

        e1 = model.get_edge_activation("conv1")
        if e1 is not None:
            print(f"[Validation] conv1 edge messages sequence length: {len(e1)} (last eval shape: {e1[-1].shape})")

        stats = model.get_activation_stats("conv2")
        print(f"[Validation] conv2 stats sequence length: {len(stats)} (last iter mean: {stats[-1]['mean']:.4f})")

        # ----------------------------------------------------------------
        # Example 2: GAT with conv-only filter
        # ----------------------------------------------------------------
        @track_gnn_activations(
            track_edges=True,
            layer_filter=lambda name, mod: "conv" in name,
        )
        class SimpleGAT(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GATConv(16, 8, heads=4, concat=True)
                self.conv2 = GATConv(32, 16, heads=1, concat=False)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x, edge_index):
                x = F.elu(self.conv1(x, edge_index))
                x = self.dropout(x)
                return self.conv2(x, edge_index)

        print("-" * 50)
        print("Executing Example 2: Filtered SimpleGAT Tracker")
        gat = SimpleGAT()
        
        for epoch in range(5):
            gat.global_step = epoch
            gat(x, edge_index)
            
        print("\n" + gat.activation_summary())

    else:
        # Plain PyTorch demo (no pyg)
        @track_gnn_activations(track_layers=True, verbose=True)
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 8)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                return self.fc2(x)

        model = SimpleMLP()
        model(torch.randn(10, 16))
        print(model.activation_summary())

    # Close the metrics logging connections gracefully
    try:
        import wandb 
        if wandb.run is not None:
            wandb.finish()
            
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass