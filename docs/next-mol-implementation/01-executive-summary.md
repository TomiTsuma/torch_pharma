# Executive Summary

**NExT-Mol** combines a SELFIES autoregressive language model (MoLlama) with a graph-transformer diffusion model (DMT) to generate chemically valid 3D molecules. The LLM provides 1D structure; DMT lifts 2D graphs to 3D coordinates via VP-SDE diffusion conditioned on LLM hidden states.

## Integration Decisions

1. **Full native port** — no runtime dependency on the NExT-Mol repository
2. **Unified Trainer** — replace PyTorch Lightning with `torch_pharma.training.Trainer`
3. **Parallel model family** — NExT-Mol DMT coexists with existing EVD/EDM path
4. **Isolation boundary** — all SELFIES-specific code under `data/components/nextmol/`

## Key Innovation Preserved

Cross-modal conditioning via `rdmol2selfies` alignment: LLM token embeddings are projected to per-atom features before DGTDiffusion denoising.
