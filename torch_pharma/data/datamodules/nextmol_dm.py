"""DataModule-style factory for NExT-Mol datasets."""

from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader

from torch_pharma.data.components.nextmol.collators import QM9Collater, LMCollater
from torch_pharma.data.components.nextmol.dataset_config import get_dataset_info
from torch_pharma.data.components.nextmol.datasets.geom_drugs import GeomDrugsTorDFDataset
from torch_pharma.data.components.nextmol.datasets.qm9_lm import QM9LMDataset
from torch_pharma.data.components.nextmol.datasets.qm9_tordf import QM9TorDFDataset
from torch_pharma.models.diffusion.vp_scheduler import NoiseScheduleVPV2
from torch_pharma.paths import ensure_nextmol_dirs, resolve_data_root
from torch_pharma.utils.logging import get_pylogger

log = get_pylogger(__name__)


def add_unseen_selfies_tokens(tokenizer, data_root: Path) -> None:
    """Extend tokenizer vocabulary with dataset-specific SELFIES tokens."""
    for name in ("unseen_selfies_tokens.txt", "unseen_sf_tokens.txt"):
        path = data_root / name
        if not path.exists():
            continue
        unseen = path.read_text(encoding="utf-8").splitlines()
        vocab = tokenizer.get_vocab()
        added = sum(1 for token in unseen if token and token not in vocab and tokenizer.add_tokens(token))
        if added:
            log.info("Added %d unseen SELFIES tokens from %s", added, path.name)
        return


def build_nextmol_dataloaders(
    root: Optional[Union[str, Path]] = None,
    dataset_name: str = "QM9-df",
    selfies_tokenizer = None,
    batch_size: int = 32,
    num_workers: int =2,
    mode: str = "dmt",
    noise_scheduler: str = "cosine",
):
    ensure_nextmol_dirs()
    data_root = Path(resolve_data_root(root, dataset_name if mode != "llm" else "qm9_lm"))
    info = get_dataset_info(dataset_name)
    add_unseen_selfies_tokens(selfies_tokenizer, data_root)
    log.info(
        "Building NExT-Mol dataloaders: mode=%s dataset=%s root=%s batch_size=%d",
        mode,
        dataset_name,
        data_root,
        batch_size,
    )
    if mode == "llm":
        train_ds = QM9LMDataset(str(data_root), split="train")
        val_ds = QM9LMDataset(str(data_root), split="val")
        collate = LMCollater(info["max_sf_tokens"], selfies_tokenizer)
        log.info("LLM dataloaders ready: train=%d val=%d samples", len(train_ds), len(val_ds))
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers),
        )

    scheduler = NoiseScheduleVPV2(schedule=noise_scheduler)
    collate = QM9Collater(
        max_atoms=info["max_atoms"],
        max_sf_tokens=info["max_sf_tokens"],
        selfies_tokenizer=selfies_tokenizer,
        noise_scheduler=scheduler,
    )
    dataset_cls = GeomDrugsTorDFDataset if "drug" in dataset_name.lower() or "geom" in dataset_name.lower() else QM9TorDFDataset
    train_ds = dataset_cls(str(data_root), split="train")
    val_ds = dataset_cls(str(data_root), split="val")
    log.info(
        "DMT dataloaders ready: train=%d val=%d samples noise_scheduler=%s",
        len(train_ds),
        len(val_ds),
        noise_scheduler,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers),
    )
