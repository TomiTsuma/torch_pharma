import numpy as np
import scipy.stats as st
from typing import Iterable, Tuple, List, Dict, Any
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)

def calculate_mean_and_conf_int(data: Iterable, alpha: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate and report the mean and confidence interval of the data.
    
    :param data: Iterable data to calculate the mean and confidence interval.
    :param alpha: Confidence level (default: 0.95).
    :return: Tuple of the mean and confidence interval.
    """
    data = np.array(data)
    if len(data) < 2:
        return np.mean(data), (np.nan, np.nan)
    
    conf_int = st.t.interval(
        confidence=alpha,
        df=len(data) - 1,
        loc=np.mean(data),
        scale=st.sem(data),
    )
    return np.mean(data), conf_int

def aggregate_posebusters_results(results_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate PoseBusters results across multiple runs.
    
    :param results_list: List of dictionaries containing PoseBusters metrics.
    :return: Dictionary of aggregated means.
    """
    aggregated = {}
    if not results_list:
        return aggregated
    
    keys = results_list[0].keys()
    for key in keys:
        values = [res[key] for res in results_list if key in res]
        mean, conf = calculate_mean_and_conf_int(values)
        aggregated[f"{key}_mean"] = mean
        aggregated[f"{key}_conf_int"] = conf[1] - mean if not np.isnan(conf[0]) else 0.0
        
    return aggregated

def check_pb_validity(pb_results_df) -> float:
    """
    Calculate the percentage of molecules that pass all PoseBusters checks.
    
    :param pb_results_df: Pandas DataFrame containing PoseBusters results.
    :return: Percentage of valid molecules.
    """
    # Assuming the DataFrame has columns as defined in bio-diffusion
    check_columns = [
        "mol_pred_loaded", "sanitization", "all_atoms_connected",
        "bond_lengths", "bond_angles", "internal_steric_clash",
        "aromatic_ring_flatness", "double_bond_flatness",
        "internal_energy", "passes_valence_checks", "passes_kekulization"
    ]
    
    valid_mask = pb_results_df[check_columns].all(axis=1)
    return valid_mask.mean()
