"""Inference script for MoLlama SELFIES generation (arXiv 2302.13971, Stage 1).

Loads a fine-tuned MoLlama checkpoint and generates novel molecules via
autoregressive SELFIES sampling. Each SELFIES sequence is decoded to a
canonical SMILES string.

Usage (base weights, no fine-tuning):
    python infer_mollama.py --num_samples 200 --output generated.smi

Usage (fine-tuned checkpoint directory):
    python infer_mollama.py \\
        --checkpoint ~/.torch_pharma/checkpoints/nextmol/mollama \\
        --num_samples 200 \\
        --output generated.smi

Usage (fine-tuned .pt file):
    python infer_mollama.py \\
        --checkpoint ~/.torch_pharma/checkpoints/nextmol/mollama/best.pt \\
        --num_samples 200 \\
        --output generated.smi
"""

import argparse
from pathlib import Path
from typing import Optional

import torch

from torch_pharma.models.llm.mollama import init_mollama_tokenizer, load_mollama
from torch_pharma.paths import NEXTMOL_MOLLAMA_CHECKPOINTS
from torch_pharma.utils.logging import get_pylogger, setup_logging

log = get_pylogger(__name__)


def selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """Convert a SELFIES string to canonical SMILES. Returns None on failure."""
    try:
        import selfies as sf
        from rdkit import Chem

        smiles = sf.decoder(selfies_str.strip())
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="MoLlama SELFIES molecule generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Fine-tuned MoLlama checkpoint: a HuggingFace model directory, a .pt file "
            f"with 'model_state_dict', or None to use base weights from --llm_model. "
            f"Default auto-detects latest checkpoint in {NEXTMOL_MOLLAMA_CHECKPOINTS}."
        ),
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="acharkq/MoLlama",
        help="Base HuggingFace model ID (used when --checkpoint is a .pt weight file).",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per SELFIES sequence.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (lower = more conservative).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability threshold.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of molecules to generate per forward pass.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_smiles.smi",
        help="Output file: one SMILES per line.",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level, run_name="infer_mollama")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Resolve checkpoint ───────────────────────────────────────────────────────
    ckpt_path: Optional[Path] = None
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        # Auto-detect latest .pt in the default checkpoint directory
        candidates = sorted(
            NEXTMOL_MOLLAMA_CHECKPOINTS.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            ckpt_path = candidates[-1]
            log.info("Auto-detected checkpoint: %s", ckpt_path)
        else:
            log.info("No fine-tuned checkpoint found; using base %s weights.", args.llm_model)

    # ── Load tokenizer ───────────────────────────────────────────────────────────
    tokenizer = init_mollama_tokenizer(args.llm_model)

    # ── Load model ───────────────────────────────────────────────────────────────
    if ckpt_path is not None and ckpt_path.is_dir():
        # HuggingFace saved-model directory (from trainer.save_pretrained or similar)
        from transformers import AutoModelForCausalLM

        log.info("Loading fine-tuned MoLlama from HF directory: %s", ckpt_path)
        model = AutoModelForCausalLM.from_pretrained(str(ckpt_path))
    else:
        log.info("Loading base MoLlama: %s", args.llm_model)
        model = load_mollama(args.llm_model, llm_tune="freeze")

        if ckpt_path is not None and ckpt_path.is_file():
            log.info("Loading fine-tuned weights from: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
            missing, unexpected = model.load_state_dict(state, strict=False)
            log.info(
                "Weights loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected)
            )

    model = model.to(device)
    model.eval()

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    if bos_id is None:
        raise ValueError("Tokenizer has no bos_token_id — check the MoLlama tokenizer config.")

    # ── Generation loop ──────────────────────────────────────────────────────────
    all_smiles = []
    total_generated = 0

    log.info("Generating %d molecules (batch_size=%d) …", args.num_samples, args.batch_size)

    while total_generated < args.num_samples:
        n = min(args.batch_size, args.num_samples - total_generated)
        input_ids = torch.full((n, 1), bos_id, dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )

        for seq in output_ids:
            selfies_str = tokenizer.decode(seq, skip_special_tokens=True)
            smiles = selfies_to_smiles(selfies_str)
            if smiles:
                all_smiles.append(smiles)

        total_generated += n
        log.info(
            "Progress: %d/%d sampled  |  valid SMILES: %d",
            total_generated,
            args.num_samples,
            len(all_smiles),
        )

    # ── Write output ─────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(all_smiles))
    log.info(
        "Wrote %d valid SMILES (%.1f%% valid from %d samples) → %s",
        len(all_smiles),
        100.0 * len(all_smiles) / max(args.num_samples, 1),
        args.num_samples,
        out_path,
    )


if __name__ == "__main__":
    main()
