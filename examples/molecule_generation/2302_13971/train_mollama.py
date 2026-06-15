"""Train MoLlama (NExT-Mol stage 1) via torch_pharma unified Trainer."""

import argparse

from torch_pharma.data.datamodules.nextmol_dm import build_nextmol_dataloaders
from torch_pharma.config.model.nextmol.config import NextMolTrainingConfig
from torch_pharma.models.llm.mollama import init_mollama_tokenizer
from torch_pharma.paths import NEXTMOL_MOLLAMA_CHECKPOINTS, NEXTMOL_QM9_LM, resolve_checkpoint_dir, resolve_data_root
from torch_pharma.tasks.molecule_generation.nextmol_llm_task import NextMolLLMTask
from torch_pharma.training import ModelCheckpoint, Trainer
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
        help=f"Override data dir (default: {NEXTMOL_QM9_LM})",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=f"Override checkpoint dir (default: {NEXTMOL_MOLLAMA_CHECKPOINTS})",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--llm_model", type=str, default="acharkq/MoLlama")
    args = parser.parse_args()

    setup_logging(level=args.log_level, log_file=args.log_file, run_name="train_mollama")
    log.info("Starting MoLlama stage-1 training (model=%s)", args.llm_model)

    data_root = resolve_data_root(args.data_root, "qm9_lm")
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir, task="mollama")
    log.info("data_root=%s checkpoint_dir=%s", data_root, ckpt_dir)

    cfg = NextMolTrainingConfig(llm_model=args.llm_model, llm_tune="lora", max_epochs=args.max_epochs)
    tokenizer = init_mollama_tokenizer(args.llm_model)
    train_loader, val_loader = build_nextmol_dataloaders(
        data_root, "QM9", tokenizer, batch_size=args.batch_size, mode="llm"
    )

    task = NextMolLLMTask(cfg)
    trainer = Trainer(
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
        default_root_dir=str(ckpt_dir),
        callbacks=[ModelCheckpoint(monitor="val_loss", dirpath=str(ckpt_dir))],
    )
    trainer.fit(task, train_loader, val_loader)
    log.info("MoLlama training complete")


if __name__ == "__main__":
    main()
