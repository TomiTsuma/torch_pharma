"""Install NExT-Mol OSF datasets into TORCH_PHARMA_HOME.

Copies QM92014 (MoLlama), GEOM-QM9 (QM9 TorDF/DMT), and geom_drugs_jodo
from a local OSF archive into the canonical torch_pharma data layout.

Usage:
    py -3.11 scripts/install_nextmol_data.py
    py -3.11 scripts/install_nextmol_data.py --source "C:/path/to/osfstorage-archive"
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from torch_pharma.paths import (
    NEXTMOL_GEOM_DRUGS,
    NEXTMOL_QM9_LM,
    NEXTMOL_QM9_TORDF,
    TORCH_PHARMA_HOME,
    ensure_nextmol_dirs,
)
from torch_pharma.utils.logging import get_pylogger, setup_logging

log = get_pylogger(__name__)

DEFAULT_SOURCE = Path.home() / "Downloads" / "osfstorage-archive" / "datasets"


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        log.info("Copied %s -> %s", item, target)


def _copy_files(src_dir: Path, dst_dir: Path, names: list[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Expected file missing: {src}")
        shutil.copy2(src, dst_dir / name)
        log.info("Copied %s", name)


def install_qm9_lm(source: Path, dest: Path) -> None:
    """QM92014 preprocessed cache for MoLlama (stage 1)."""
    src = source / "QM92014" / "QM92014"
    _copy_tree(src, dest)
    cond_split = source / "split_dict_cond_qm9.pt"
    if cond_split.exists():
        shutil.copy2(cond_split, dest / "processed" / "split_dict_cond_qm9.pt")
        log.info("Copied split_dict_cond_qm9.pt for conditional LM training")


def install_tordf_qm9(source: Path, dest: Path) -> None:
    """GEOM-QM9 TorDF caches for QM9 DMT (stage 2/3)."""
    src = source / "GEOM-QM9" / "GEOM-QM9"
    names = [
        "processed_train.pt",
        "processed_val.pt",
        "processed_inference_test.pt",
        "tordf.train",
        "tordf.val",
        "tordf.test",
        "test_smiles.csv",
        "test_mols.pkl",
        "unseen_selfies_tokens.txt",
        "split.npy",
        "sta.txt",
    ]
    _copy_files(src, dest, names)


def install_geom_drugs(source: Path, dest: Path, force_extract: bool = False) -> None:
    """geom_drugs_jodo JODO-format cache for large-molecule workflows."""
    zip_path = source / "geom_drugs_jodo.zip"
    src = source / "geom_drugs_jodo"

    if force_extract or not (src / "processed_data.pt").exists():
        if not zip_path.exists():
            raise FileNotFoundError(f"geom_drugs_jodo.zip not found at {zip_path}")
        log.info("Extracting %s (this may take several minutes)...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(source)

    if not src.exists():
        raise FileNotFoundError(f"Expected extracted geom_drugs_jodo directory at {src}")
    names = [
        "data_geom_drug_1.pt",
        "split_dict_geom_drug_1.pt",
        "processed_data.pt",
        "unseen_sf_tokens.txt",
        "moses_stat.pkl",
        "target_geometry_stat.pk",
    ]
    _copy_files(src, dest, [n for n in names if (src / n).exists()])


def verify_installation() -> dict[str, bool]:
    checks = {
        "qm9_lm": (NEXTMOL_QM9_LM / "processed" / "data_qm9.pt").exists(),
        "qm9_lm_splits": (NEXTMOL_QM9_LM / "processed" / "split_dict_qm9.pt").exists(),
        "tordf_qm9_train": (NEXTMOL_QM9_TORDF / "processed_train.pt").exists(),
        "tordf_qm9_val": (NEXTMOL_QM9_TORDF / "processed_val.pt").exists(),
        "geom_drugs": (NEXTMOL_GEOM_DRUGS / "processed_data.pt").exists()
        or (NEXTMOL_GEOM_DRUGS / "data_geom_drug_1.pt").exists(),
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Install NExT-Mol OSF data into TORCH_PHARMA_HOME")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"OSF datasets folder (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument("--skip-geom", action="store_true", help="Skip geom_drugs (large ~7GB extract)")
    parser.add_argument("--force-extract-geom", action="store_true", help="Re-extract geom_drugs_jodo.zip")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level, run_name="install_nextmol_data")
    ensure_nextmol_dirs()

    source = args.source.resolve()
    log.info("Installing NExT-Mol data from %s into %s", source, TORCH_PHARMA_HOME)

    install_qm9_lm(source, NEXTMOL_QM9_LM)
    install_tordf_qm9(source, NEXTMOL_QM9_TORDF)
    if not args.skip_geom:
        install_geom_drugs(source, NEXTMOL_GEOM_DRUGS, force_extract=args.force_extract_geom)

    checks = verify_installation()
    for name, ok in checks.items():
        status = "OK" if ok else "MISSING"
        log.info("  [%s] %s", status, name)

    required = {k: v for k, v in checks.items() if not (args.skip_geom and k == "geom_drugs")}
    if not all(required.values()):
        missing = [k for k, v in required.items() if not v]
        raise SystemExit(f"Installation incomplete; missing: {missing}")

    log.info("NExT-Mol data installation complete.")


if __name__ == "__main__":
    main()
