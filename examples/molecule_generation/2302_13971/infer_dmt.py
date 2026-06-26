"""Inference script for NExT-Mol DMT 3D conformer generation (arXiv 2302.13971, Stage 2/3).

Loads a trained DMT checkpoint and generates 3D molecular conformers for
molecules in the QM9 test split. Each conformer is saved as an SDF file.

The script uses ``QM9InferCollater`` which initialises atomic positions from
random noise, so no reference conformer is needed at inference time.

Usage:
    python infer_dmt.py \\
        --checkpoint ~/.torch_pharma/checkpoints/nextmol/dmt/best.pt \\
        --num_samples 100 \\
        --output_dir generated_conformers/

With property-conditioned checkpoint (e.g. trained via train_dmt_cond.py):
    python infer_dmt.py \\
        --checkpoint ~/.torch_pharma/checkpoints/nextmol/dmt/best.pt \\
        --property gap \\
        --num_samples 100 \\
        --output_dir generated_conformers/
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from torch_pharma.config.model.nextmol.config import DGTDiffusionConfig, NextMolTrainingConfig
from torch_pharma.data.components.nextmol.collators import QM9InferCollater
from torch_pharma.data.components.nextmol.dataset_config import get_dataset_info
from torch_pharma.data.components.nextmol.datasets.qm9_tordf import QM9TorDFDataset
from torch_pharma.data.datamodules.nextmol_dm import add_unseen_selfies_tokens
from torch_pharma.models.llm.mollama import init_mollama_tokenizer
from torch_pharma.paths import NEXTMOL_DMT_CHECKPOINTS, NEXTMOL_QM9_TORDF, resolve_data_root
from torch_pharma.tasks.molecule_generation.nextmol_dmt_task import NextMolDMTModule
from torch_pharma.utils.logging import get_pylogger, setup_logging

log = get_pylogger(__name__)


def save_sdf(rdmol, positions: torch.Tensor, out_path: Path) -> None:
    """Write a single 3D conformer to an SDF file."""
    from rdkit.Chem import Conformer, SDWriter

    mol_with_conf = rdmol.__class__(rdmol)
    mol_with_conf.RemoveAllConformers()
    conf = Conformer(mol_with_conf.GetNumAtoms())
    pos_np = positions.detach().cpu().float().numpy()
    for i, (x, y, z) in enumerate(pos_np):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol_with_conf.AddConformer(conf, assignId=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SDWriter(str(out_path))
    writer.write(mol_with_conf)
    writer.close()


def resolve_checkpoint(checkpoint_arg: Optional[str]) -> Path:
    if checkpoint_arg is not None:
        p = Path(checkpoint_arg)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    candidates = sorted(
        NEXTMOL_DMT_CHECKPOINTS.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .pt checkpoints found in {NEXTMOL_DMT_CHECKPOINTS}. "
            "Run train_dmt_uncond.py or train_dmt_cond.py first, "
            "or supply --checkpoint explicitly."
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NExT-Mol DMT: generate 3D conformers from QM9 test molecules"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a .pt DMT checkpoint. "
            f"Auto-detects latest in {NEXTMOL_DMT_CHECKPOINTS} when omitted."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=f"QM9 TorDF dataset root directory. Default: {NEXTMOL_QM9_TORDF}",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="acharkq/MoLlama",
        help="HuggingFace model ID for the MoLlama tokenizer (must match training).",
    )
    parser.add_argument(
        "--property",
        type=str,
        default=None,
        choices=["mu", "alpha", "homo", "lumo", "gap", "Cv"],
        help="Target property for conditional DMT checkpoint. Leave unset for unconditional.",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=100,
        help="Number of reverse-SDE denoising steps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_conformers",
        help="Directory where per-molecule SDF files will be written.",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level, run_name="infer_dmt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Paths ────────────────────────────────────────────────────────────────────
    ckpt_path = resolve_checkpoint(args.checkpoint)
    data_root = Path(resolve_data_root(args.data_root, "tordf_qm9"))
    log.info("Checkpoint : %s", ckpt_path)
    log.info("Data root  : %s", data_root)

    # ── Tokenizer + dataset ──────────────────────────────────────────────────────
    tokenizer = init_mollama_tokenizer(args.llm_model)
    add_unseen_selfies_tokens(tokenizer, data_root)

    ds_info = get_dataset_info("QM9-df")
    test_ds = QM9TorDFDataset(str(data_root), split="test")
    if len(test_ds) == 0:
        raise RuntimeError(
            f"QM9 test split is empty at {data_root}. "
            "Run scripts/install_nextmol_data.py to download the dataset."
        )
    log.info("Test dataset: %d molecules", len(test_ds))

    # QM9InferCollater initialises atom positions from random noise — no reference
    # conformer is required. The reverse SDE then refines these positions.
    collate = QM9InferCollater(
        max_atoms=ds_info["max_atoms"],
        max_sf_tokens=ds_info["max_sf_tokens"],
        selfies_tokenizer=tokenizer,
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
    )

    # ── Build & load model ───────────────────────────────────────────────────────
    train_cfg = NextMolTrainingConfig(
        use_llm=True,
        sampling_steps=args.sampling_steps,
        condition_property=args.property,
    )
    dmt_cfg = DGTDiffusionConfig.dmt_b()
    module = NextMolDMTModule(train_cfg, dmt_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        log.warning(
            "%d missing keys (may be normal for partial checkpoints): %s …",
            len(missing),
            missing[:3],
        )
    log.info(
        "Checkpoint loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected)
    )

    module = module.to(device)
    module.eval()

    # ── Inference loop ───────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    failed = 0

    for batch_idx, (data_batch, selfies_batch) in enumerate(loader):
        if generated >= args.num_samples:
            break

        data_batch = data_batch.to(device)
        selfies_batch = selfies_batch.to(device)

        # module.sample() runs the reverse VP-SDE and returns (data_batch, positions)
        context = None
        data_batch_out, positions = module.sample(data_batch, selfies_batch, context)

        # data_batch.ptr gives atom-level slice boundaries for each molecule
        ptr = data_batch_out.ptr.cpu()
        rdmols = getattr(data_batch_out, "rdmol", None)
        smiles_list = getattr(data_batch_out, "smiles", [])

        for i in range(len(smiles_list)):
            if generated >= args.num_samples:
                break

            start_atom = int(ptr[i])
            end_atom = int(ptr[i + 1])
            mol_pos = positions[start_atom:end_atom]  # (num_atoms, 3)

            smiles = smiles_list[i] if i < len(smiles_list) else f"mol_{generated}"
            rdmol = rdmols[i] if rdmols is not None and i < len(rdmols) else None

            out_file = out_dir / f"conf_{generated:04d}.sdf"
            if rdmol is not None:
                try:
                    save_sdf(rdmol, mol_pos, out_file)
                    log.debug("Saved %s  (%s)", out_file.name, smiles)
                except Exception as exc:
                    log.warning("Could not save SDF for molecule %d: %s", generated, exc)
                    failed += 1
            else:
                log.warning("No rdmol for molecule %d (%s) — skipping SDF write", generated, smiles)
                failed += 1

            generated += 1

        log.info(
            "Batch %d done  |  conformers saved: %d / %d",
            batch_idx,
            generated - failed,
            args.num_samples,
        )

    log.info(
        "Done. Saved %d conformers to %s (failed: %d)",
        generated - failed,
        out_dir,
        failed,
    )


if __name__ == "__main__":
    main()
