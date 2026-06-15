"""Smoke tests for NExT-Mol native integration modules."""

import pytest


def test_dmt_import():
    from torch_pharma.models.diffusion.dmt import DGTDiffusion, DGTDiffusionConfig, TransLayer

    assert DGTDiffusion is not None
    assert TransLayer is not None


def test_equivariance_import():
    from torch_pharma.models.dynamics.equivariance import EquivariantBlock

    assert EquivariantBlock is not None


def test_vp_scheduler():
    import torch
    from torch_pharma.models.diffusion.vp_scheduler import NoiseScheduleVPV2

    sched = NoiseScheduleVPV2(schedule="cosine")
    t = torch.tensor([0.5])
    alpha, sigma = sched.marginal_prob(t)
    assert alpha.shape == t.shape
    assert sigma.shape == t.shape


def test_trainer_import():
    from torch_pharma.training import Trainer, ModelCheckpoint, EarlyStopping

    assert Trainer is not None


def test_nextmol_tasks_import():
    from torch_pharma.tasks.molecule_generation import NextMolDMTTask, NextMolLLMTask

    assert NextMolDMTTask is not None


def test_dgt_diffusion_config_instantiation():
    from types import SimpleNamespace

    from torch_pharma.config.model.nextmol.config import DGTDiffusionConfig
    from torch_pharma.models.diffusion.dmt import DGTDiffusion

    cfg = DGTDiffusionConfig.dmt_b()
    cfg.use_llm = False
    model = DGTDiffusion(cfg)
    assert model.n_blocks == 6
