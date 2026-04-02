import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Any

try:
    from posebusters import PoseBusters
    HAS_POSEBUSTERS = True
except ImportError:
    HAS_POSEBUSTERS = False

from torch_pharma.utils.logging import get_pylogger
from torch_pharma.evaluation.molecules.metrics import calculate_mean_and_conf_int

log = get_pylogger(__name__)

class PoseBustersEvaluator:
    def __init__(self, config: str = "mol", full_report: bool = False):
        if not HAS_POSEBUSTERS:
            raise ImportError("PoseBusters is not installed. Please install it with `pip install posebusters`.")
        self.buster = PoseBusters(config=config, top_n=None)
        self.full_report = full_report

    def evaluate_molecules(self, mol_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate a table of molecules using PoseBusters.
        
        :param mol_df: Pandas DataFrame with columns like 'mol_pred'.
        :return: PoseBusters results as a DataFrame.
        """
        log.info(f"Running PoseBusters evaluation on {len(mol_df)} molecules...")
        results = self.buster.bust_table(mol_df, full_report=self.full_report)
        return results

def create_molecule_table(molecule_paths: List[str]) -> pd.DataFrame:
    """
    Create a molecule table from a list of SDF file paths.
    
    :param molecule_paths: List of paths to SDF files.
    :return: Molecule table as a Pandas DataFrame.
    """
    return pd.DataFrame({"mol_pred": molecule_paths})

def plot_comparative_bust_analysis(
    method_1_name: str,
    method_1_results: pd.DataFrame,
    method_2_name: str,
    method_2_results: pd.DataFrame,
    column_name: str,
    save_path: str,
    ylim: Optional[tuple] = (0, 10)
):
    """
    Compare PoseBusters results from two methods using a boxplot.
    """
    method_1_data = method_1_results[[column_name]].copy()
    method_1_data["source"] = method_1_name
    
    method_2_data = method_2_results[[column_name]].copy()
    method_2_data["source"] = method_2_name
    
    combined_data = pd.concat([method_1_data, method_2_data], ignore_index=True)
    
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="source", y=column_name, data=combined_data)
    if ylim:
        ax.set_ylim(*ylim)
    plt.xlabel("Method")
    plt.ylabel(column_name.replace("_", " ").title())
    plt.savefig(save_path, dpi=300)
    plt.close()
    log.info(f"Comparative bust analysis plot saved to {save_path}")
