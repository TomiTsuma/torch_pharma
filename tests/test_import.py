import torch_pharma.data.components.edm as edm
try:
    print(f"get_bond_order exists: {hasattr(edm, 'get_bond_order')}")
    from torch_pharma.data.components.edm import get_bond_order
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
