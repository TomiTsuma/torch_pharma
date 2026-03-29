import torch
import numpy as np
from torch_geometric.data import Batch
from torch_pharma.models.diffusion.variational_diffusion import EquivariantVariationalDiffusion
from torch_pharma.models.dynamics.egnn import EGNNDynamics

def test_diffusion_initialization():
    print("Testing EquivariantVariationalDiffusion initialization...")
    
    # Initialize dynamics network
    dynamics = EGNNDynamics(
        num_atom_types=16,
        include_charges=False,
        diffusion_target="atom_types_and_coords",
        h_input_dim=16,
        h_hidden_dim=256,
        condition_on_time=True,
        self_condition=True,
        num_encoder_layers=9
    )
    
    # Initialize diffusion model
    model = EquivariantVariationalDiffusion(
        dynamics_network=dynamics,
        num_atom_types=16,
        num_charge_classes=0,
        include_charges=False,
        T=1000,
        parametrization="eps",
        noise_schedule="prec_linear"
    )
    
    print("Initialization successful!")
    return model, dynamics

def test_diffusion_forward_pass(model):
    print("Testing EquivariantVariationalDiffusion forward pass...")
    
    batch_size = 2
    num_nodes_per_graph = [5, 7]
    total_nodes = sum(num_nodes_per_graph)
    
    # Mock node positions (x) and features (h/one_hot)
    x = torch.randn(total_nodes, 3)
    one_hot = torch.zeros(total_nodes, 16)
    for i in range(total_nodes):
        one_hot[i, i % 16] = 1.0
        
    mask = torch.ones(total_nodes, 1, dtype=torch.bool)
    
    # Create batch index
    batch_idx = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_nodes_per_graph)])
    
    # Create geometric data batch
    data = Batch(
        x=x,
        h={"categorical": one_hot, "integer": torch.zeros(total_nodes, 0)},
        mask=mask,
        batch=batch_idx,
        num_graphs=batch_size
    )
    
    # Forward pass
    loss, loss_info = model(data, return_loss_info=True)
    
    print(f"Forward pass successful! Loss: {loss.item()}")
    print(f"Loss info: {loss_info.keys()}")
    
    assert not torch.isnan(loss)
    return loss, loss_info

if __name__ == "__main__":
    try:
        model, dynamics = test_diffusion_initialization()
        loss, loss_info = test_diffusion_forward_pass(model)
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
