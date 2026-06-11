# Current torch_pharma State

## Completed Ports

| Component | Location |
|-----------|----------|
| `DGTDiffusion` | `torch_pharma/models/diffusion/dmt.py` |
| `EquivariantBlock` | `torch_pharma/models/dynamics/equivariance.py` |
| `TransformLayer` | `torch_pharma/models/message_passing/transformation_layer.py` |
| `ExtendedProjector` | `torch_pharma/models/dynamics/projector.py` |
| `GaussianLayer` | `torch_pharma/models/diffusion/noise.py` |
| `NoiseScheduleVPV2` | `torch_pharma/models/diffusion/vp_scheduler.py` |

## Fixed Issues (Phase 0)

- Circular import between `dmt.py` and `equivariance.py` resolved via re-exports
- `DGTDiffusionConfig` dataclass replaces argparse `args` namespace
- Device-agnostic `disable_compile` via `torch_pharma/utils/device.py`

## Remaining Gaps

- Dataset preprocessing still uses NExT-Mol OSF caches (`.pt` files)
- Full JODO evaluation stack not ported (simplified metrics in `evaluation/nextmol/`)
- `Trainer` DDP multi-GPU not yet implemented
