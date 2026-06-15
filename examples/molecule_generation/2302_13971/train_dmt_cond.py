"""Train property-conditioned DMT (NExT-Mol workflow B)."""

import argparse

from torch_pharma.data.datamodules.nextmol_dm import build_nextmol_dataloaders
from torch_pharma.config.model.nextmol.config import DGTDiffusionConfig, NextMolTrainingConfig
from torch_pharma.models.llm.mollama import init_mollama_tokenizer
from torch_pharma.paths import NEXTMOL_DMT_CHECKPOINTS, NEXTMOL_QM9_TORDF, resolve_checkpoint_dir, resolve_data_root
from torch_pharma.tasks.molecule_generation.nextmol_dmt_task import NextMolDMTTask
from torch_pharma.training import DeltaTrainingCallback, ModelCheckpoint, Trainer
from torch_pharma.utils.logging import get_pylogger, setup_logging

log = get_pylogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=f"Override data dir (default: {NEXTMOL_QM9_TORDF})",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=f"Override checkpoint dir (default: {NEXTMOL_DMT_CHECKPOINTS})",
    )
    parser.add_argument("--property", type=str, default="gap", choices=["mu", "alpha", "homo", "lumo", "gap", "Cv"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()

    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        run_name=f"train_dmt_cond_{args.property}",
    )
    log.info("Starting property-conditioned DMT training (property=%s)", args.property)

    data_root = resolve_data_root(args.data_root, "tordf_qm9")
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir, task="dmt")
    log.info("data_root=%s checkpoint_dir=%s", data_root, ckpt_dir)

    train_cfg = NextMolTrainingConfig(
        use_llm=True,
        condition_property=args.property,
        delta_train_epochs=10,
    )
    dmt_cfg = DGTDiffusionConfig.dmt_b()
    tokenizer = init_mollama_tokenizer(train_cfg.llm_model)
    train_loader, val_loader = build_nextmol_dataloaders(
        data_root, "QM9-df", tokenizer, batch_size=args.batch_size, mode="dmt"
    )

    task = NextMolDMTTask(train_cfg, dmt_cfg)
    trainer = Trainer(
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
        default_root_dir=str(ckpt_dir),
        callbacks=[
            DeltaTrainingCallback(warmup_epochs=train_cfg.delta_train_epochs),
            ModelCheckpoint(monitor="val_loss", dirpath=str(ckpt_dir)),
        ],
    )
    trainer.fit(task, train_loader, val_loader)
    log.info("Property-conditioned DMT training complete")


if __name__ == "__main__":
    main()
