"""Tests for TORCH_PHARMA_HOME path layout."""

from pathlib import Path

from torch_pharma.paths import (
    TORCH_PHARMA_HOME,
    TORCH_PHARMA_LOGS,
    NEXTMOL_QM9_LM,
    NEXTMOL_QM9_TORDF,
    NEXTMOL_DMT_CHECKPOINTS,
    NEXTMOL_PRETRAINED,
    ensure_nextmol_dirs,
    resolve_data_root,
    resolve_checkpoint_dir,
    resolve_log_file,
)


def test_torch_pharma_home_under_user_home():
    assert TORCH_PHARMA_HOME.exists()
    assert "torch_pharma" in str(TORCH_PHARMA_HOME)


def test_nextmol_paths_under_home():
    ensure_nextmol_dirs()
    assert NEXTMOL_QM9_LM.is_relative_to(TORCH_PHARMA_HOME)
    assert NEXTMOL_QM9_TORDF.is_relative_to(TORCH_PHARMA_HOME)
    assert NEXTMOL_DMT_CHECKPOINTS.is_relative_to(TORCH_PHARMA_HOME)
    assert NEXTMOL_PRETRAINED.is_relative_to(TORCH_PHARMA_HOME)


def test_resolve_defaults():
    assert resolve_data_root(None, "qm9_lm") == NEXTMOL_QM9_LM
    assert resolve_data_root(None, "tordf_qm9") == NEXTMOL_QM9_TORDF
    assert resolve_checkpoint_dir(None, task="dmt") == NEXTMOL_DMT_CHECKPOINTS


def test_resolve_log_file_defaults():
    ensure_nextmol_dirs()
    assert resolve_log_file(None, "train_dmt_cond_gap") == TORCH_PHARMA_LOGS / "train_dmt_cond_gap.log"
    assert resolve_log_file("custom_run", "ignored") == TORCH_PHARMA_LOGS / "custom_run.log"
    custom_path = Path("C:/tmp/experiment") if Path("C:/").exists() else Path("/tmp/experiment")
    assert resolve_log_file(str(custom_path), "ignored") == custom_path.with_suffix(".log")
