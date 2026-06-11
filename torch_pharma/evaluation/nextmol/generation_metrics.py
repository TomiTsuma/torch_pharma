"""2D/3D generation quality metrics for NExT-Mol."""

from __future__ import annotations

from typing import Iterable, Set

from rdkit import Chem

from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


def validity_rate(smiles_list: Iterable[str]) -> float:
    valid = 0
    total = 0
    for s in smiles_list:
        total += 1
        if Chem.MolFromSmiles(s) is not None:
            valid += 1
    rate = valid / max(total, 1)
    log.info("Validity: %.2f%% (%d/%d)", rate * 100, valid, total)
    return rate


def uniqueness_rate(smiles_list: Iterable[str]) -> float:
    items = list(smiles_list)
    if not items:
        return 0.0
    rate = len(set(items)) / len(items)
    log.info("Uniqueness: %.2f%% (%d unique / %d total)", rate * 100, len(set(items)), len(items))
    return rate


def novelty_rate(smiles_list: Iterable[str], training_set: Set[str]) -> float:
    items = list(smiles_list)
    if not items:
        return 0.0
    novel = sum(1 for s in items if s not in training_set)
    rate = novel / len(items)
    log.info("Novelty: %.2f%% (%d novel / %d total)", rate * 100, novel, len(items))
    return rate
