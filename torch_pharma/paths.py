"""Canonical paths under TORCH_PHARMA_HOME for data and checkpoints."""

from pathlib import Path

TORCH_PHARMA_HOME = Path.home() / ".torch_pharma"
TORCH_PHARMA_HOME.mkdir(parents=True, exist_ok=True)

# Shared data layout (EDM QM9 lives under data/QM9 via datasets/utils.py)
TORCH_PHARMA_DATA = TORCH_PHARMA_HOME / "data"
TORCH_PHARMA_DATA.mkdir(parents=True, exist_ok=True)

# NExT-Mol datasets
NEXTMOL_DATA = TORCH_PHARMA_DATA / "nextmol"
NEXTMOL_QM9_LM = NEXTMOL_DATA / "qm9_lm"
NEXTMOL_QM9_TORDF = NEXTMOL_DATA / "tordf_qm9"
NEXTMOL_GEOM_DRUGS = NEXTMOL_DATA / "geom_drugs"

# Checkpoints
TORCH_PHARMA_CHECKPOINTS = TORCH_PHARMA_HOME / "checkpoints"
NEXTMOL_CHECKPOINTS = TORCH_PHARMA_CHECKPOINTS / "nextmol"
NEXTMOL_MOLLAMA_CHECKPOINTS = NEXTMOL_CHECKPOINTS / "mollama"
NEXTMOL_DMT_CHECKPOINTS = NEXTMOL_CHECKPOINTS / "dmt"

# Pretrained / migrated weights (OSF, HuggingFace exports)
NEXTMOL_PRETRAINED = TORCH_PHARMA_HOME / "pretrained" / "nextmol"

# Experiment logs
TORCH_PHARMA_LOGS = TORCH_PHARMA_HOME / "logs"


def ensure_nextmol_dirs() -> None:
    """Create NExT-Mol data and checkpoint directories under TORCH_PHARMA_HOME."""
    for path in (
        NEXTMOL_DATA,
        NEXTMOL_QM9_LM,
        NEXTMOL_QM9_TORDF,
        NEXTMOL_GEOM_DRUGS,
        TORCH_PHARMA_CHECKPOINTS,
        NEXTMOL_CHECKPOINTS,
        NEXTMOL_MOLLAMA_CHECKPOINTS,
        NEXTMOL_DMT_CHECKPOINTS,
        NEXTMOL_PRETRAINED,
        TORCH_PHARMA_LOGS,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_log_file(log_file: str | Path | None, run_name: str) -> Path:
    """Resolve a log file under TORCH_PHARMA_LOGS, always with a .log suffix."""
    TORCH_PHARMA_LOGS.mkdir(parents=True, exist_ok=True)
    path = Path(log_file) if log_file is not None else Path(run_name)
    if path.suffix != ".log":
        path = path.with_suffix(".log")
    if len(path.parts) == 1:
        path = TORCH_PHARMA_LOGS / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_root(root: str | Path | None, dataset_key: str) -> Path:
    """Resolve a data root, defaulting to TORCH_PHARMA_HOME/nextmol paths."""
    if root is not None:
        return Path(root)
    key = dataset_key.lower()
    if "lm" in key or key == "qm9":
        return NEXTMOL_QM9_LM
    if "drug" in key or "geom" in key:
        return NEXTMOL_GEOM_DRUGS
    return NEXTMOL_QM9_TORDF


def resolve_checkpoint_dir(root: str | Path | None, task: str = "nextmol") -> Path:
    """Resolve checkpoint directory under TORCH_PHARMA_HOME."""
    if root is not None:
        return Path(root)
    ensure_nextmol_dirs()
    if task == "mollama":
        return NEXTMOL_MOLLAMA_CHECKPOINTS
    if task == "dmt":
        return NEXTMOL_DMT_CHECKPOINTS
    return NEXTMOL_CHECKPOINTS
