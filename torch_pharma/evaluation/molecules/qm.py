import os
import subprocess
from typing import Optional, Dict, Any, Union
from torch_pharma.utils.logging import get_pylogger

try:
    import psi4
    HAS_PSI4 = True
except ImportError:
    HAS_PSI4 = False

log = get_pylogger(__name__)

class QMEvaluator:
    def __init__(
        self,
        dataset: str = "qm9",
        memory: str = "32 GB",
        num_threads: int = 4,
        verbose: bool = True
    ):
        self.dataset = dataset
        self.memory = memory
        self.num_threads = num_threads
        self.verbose = verbose
        
        if dataset == "qm9" and HAS_PSI4:
            psi4.set_memory(memory)
            psi4.set_num_threads(num_threads)
            # Default options for QM9 energy calculation
            psi4.set_options({
                "basis": "6-31G(2df,p)",
                "scf_type": "pk",
                "e_convergence": 1e-8,
                "d_convergence": 1e-8,
            })

    def calculate_properties(self, xyz_filepath: str) -> Dict[str, Any]:
        """
        Calculate QM properties for a molecule in an XYZ file.
        
        :param xyz_filepath: Path to the XYZ file.
        :return: Dictionary of calculated properties.
        """
        if self.dataset == "qm9":
            if not HAS_PSI4:
                raise ImportError("Psi4 is not installed. Please install it with `conda install -c psi4 psi4`.")
            
            with open(xyz_filepath, "r") as f:
                xyz_contents = f.read()
            
            molecule = psi4.geometry(xyz_contents)
            # Calculate polarizability (dipole_polarizabilities)
            energy = psi4.properties("B3LYP", properties=["dipole_polarizabilities"], molecule=molecule)
            
            if self.verbose:
                log.info(f"Final energy of molecule: {energy} (a.u.)")
            
            return {"energy": energy}
            
        elif self.dataset == "drugs":
            # crest-based xTB calculation
            log.info(f"Running xTB calculation for {xyz_filepath} using crest...")
            try:
                subprocess.run(
                    ["crest", xyz_filepath, "--single-point", "GFN2-xTB", "-T", str(self.num_threads), "-quick"],
                    check=True
                )
                # Note: crest outputs property files such as `.out`, we might need to parse them.
                return {"status": "success"}
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                log.error(f"Failed to run crest/xTB: {e}")
                return {"status": "error", "message": str(e)}
        else:
            raise ValueError(f"Dataset '{self.dataset}' not recognized for QM analysis.")
