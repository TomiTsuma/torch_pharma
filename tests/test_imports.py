import sys
import os
import time

def test_import(module_name, import_fn):
    print(f"Testing {module_name}...", end="", flush=True)
    start = time.time()
    try:
        import_fn()
        end = time.time()
        print(f" success ({end - start:.2f}s)")
    except Exception as e:
        print(f" FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

sys.path.append(r"c:\Users\tsuma.thomas\Documents\torch_pharma")

test_import("math", lambda: __import__("math"))
test_import("os", lambda: __import__("os"))
test_import("torch", lambda: __import__("torch"))
test_import("torchmetrics", lambda: __import__("torchmetrics"))
test_import("numpy", lambda: __import__("numpy"))
test_import("torch.nn.functional", lambda: __import__("torch.nn.functional"))
test_import("rdkit", lambda: __import__("rdkit"))
test_import("torch_geometric", lambda: __import__("torch_geometric"))
test_import("omegaconf", lambda: __import__("omegaconf"))
test_import("torch_scatter", lambda: __import__("torch_scatter"))
test_import("torch_pharma.utils.io", lambda: __import__("torch_pharma.utils.io"))
test_import("torch_pharma.features.geometry", lambda: __import__("torch_pharma.features.geometry"))
test_import("torch_pharma.utils.visualize", lambda: __import__("torch_pharma.utils.visualize"))
test_import("torch_pharma.utils.logging", lambda: __import__("torch_pharma.utils.logging"))
test_import("torch_pharma.molecules.featurizers", lambda: __import__("torch_pharma.molecules.featurizers"))
test_import("torch_pharma.molecules.chemistry", lambda: __import__("torch_pharma.molecules.chemistry"))
test_import("torch_pharma.data.datasets.utils", lambda: __import__("torch_pharma.data.datasets.utils"))
test_import("torch_pharma.data.components.edm", lambda: __import__("torch_pharma.data.components.edm"))
test_import("torch_pharma.models.dynamics.egnn", lambda: __import__("torch_pharma.models.dynamics.egnn"))
test_import("torch_pharma.models.diffusion.variational_diffusion", lambda: __import__("torch_pharma.models.diffusion.variational_diffusion"))
test_import("torch_pharma.models.dynamics.gcpnet", lambda: __import__("torch_pharma.models.dynamics.gcpnet"))
test_import("torch_pharma.models.transformers", lambda: __import__("torch_pharma.models.transformers"))
test_import("torch_pharma.utils.math", lambda: __import__("torch_pharma.utils.math"))
test_import("typeguard", lambda: __import__("typeguard"))
test_import("torchtyping", lambda: __import__("torchtyping"))

print("All tests finished.")
