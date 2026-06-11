"""Conformer generation metrics (ported from NExT-Mol conf_gen_cal_metrics)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


def set_rdmol_positions(molecule, pos, removeHs=False, add_conformer=True):
    mol = Chem.Mol(molecule)
    if removeHs:
        mol = Chem.RemoveHs(mol)
    if add_conformer:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, pos[i].tolist())
        mol.AddConformer(conf, assignId=True)
    return mol


def compute_rmsd(mol_pred, mol_gt) -> float:
    try:
        return AllChem.GetBestRMS(Chem.RemoveHs(mol_gt), Chem.RemoveHs(mol_pred))
    except Exception:
        return float("inf")


def conformer_recall(
    predictions: List[Tuple[str, Chem.Mol]],
    threshold: float = 0.5,
) -> dict:
    """Simplified COV-R metric for validation."""
    rmsds = []
    for _, mol in predictions:
        if mol.GetNumConformers() < 2:
            continue
        gt = Chem.Mol(mol)
        gt.RemoveAllConformers()
        gt.AddConformer(mol.GetConformer(0), assignId=True)
        pred = Chem.Mol(mol)
        pred.RemoveAllConformers()
        pred.AddConformer(mol.GetConformer(1), assignId=True)
        rmsds.append(compute_rmsd(pred, gt))

    rmsds = np.asarray(rmsds)
    valid = rmsds[np.isfinite(rmsds) & (rmsds < 1000)]
    if len(valid) == 0:
        log.warning("Conformer recall: no valid RMSD pairs (n_predictions=%d)", len(predictions))
        return {"cov_mean": 0.0, "mat_mean": 0.0}
    metrics = {
        "cov_mean": float(np.mean(valid <= threshold)),
        "mat_mean": float(np.mean(valid)),
        "cov_median": float(np.median(valid <= threshold)),
        "mat_median": float(np.median(valid)),
    }
    log.info(
        "Conformer recall (threshold=%.2f): cov_mean=%.4f mat_mean=%.4f (n=%d)",
        threshold,
        metrics["cov_mean"],
        metrics["mat_mean"],
        len(valid),
    )
    return metrics
