"""Data collators for NExT-Mol (ported from diffusion_data_module.py)."""

import torch
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch

from torch_pharma.models.diffusion.vp_scheduler import NoiseScheduleVPV2, sample_com_rand_pos


class QM9Collater:
    """Batches 3D graphs with VP noise injection and SELFIES tokenization."""

    def __init__(
        self,
        max_atoms: int,
        max_sf_tokens: int,
        selfies_tokenizer,
        noise_scheduler: NoiseScheduleVPV2,
        aug_rotation: bool = False,
        t_cond: str = "t",
        disable_com: bool = False,
        aug_translation: bool = False,
        load_mapping: bool = True,
    ):
        self.max_atoms = max_atoms
        self.max_sf_tokens = max_sf_tokens
        self.selfies_tokenizer = selfies_tokenizer
        self.noise_scheduler = noise_scheduler
        self.aug_rotation = aug_rotation
        self.t_cond = t_cond
        self.disable_com = disable_com
        self.aug_translation = aug_translation
        self.load_mapping = load_mapping

    def add_noise(self, data):
        t_eps = 1e-5
        bs = len(data.ptr) - 1
        t = (torch.rand(1) + torch.linspace(0, 1, bs)) % 1
        data.t = t * (1.0 - t_eps) + t_eps

        alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
        data.alpha_t_batch = alpha_t
        data.sigma_t_batch = sigma_t
        data.loss_norm = torch.sqrt(alpha_t / sigma_t)
        noise_level = torch.log(alpha_t ** 2 / sigma_t ** 2)
        noise_level, alpha_t, sigma_t = (
            noise_level[data.batch],
            alpha_t[data.batch],
            sigma_t[data.batch],
        )

        dtype = torch.float
        bs = len(data.smiles)
        if self.aug_rotation:
            rot_aug = Rotation.random(bs)
            rot_aug = rot_aug[data.batch.numpy()]
            data.pos = torch.from_numpy(rot_aug.apply(data.pos.numpy())).to(dtype)

        if self.aug_translation:
            trans_aug = 0.01 * torch.randn(bs, 3, dtype=dtype)
            data.pos = data.pos + trans_aug[data.batch]

        data.gt_pos = data.pos.clone()

        if self.disable_com:
            noise = torch.randn(data.pos.shape)
        else:
            noise = sample_com_rand_pos(data.pos.shape, data.batch, bs=bs)

        data.pos = alpha_t.view(-1, 1) * data.pos + sigma_t.view(-1, 1) * noise
        data.alpha_t = alpha_t.view(-1, 1)
        data.sigma_t = sigma_t.view(-1, 1)
        data.noise = noise

        if self.t_cond == "t":
            data.t_cond = t[data.batch]
        elif self.t_cond == "noise_level":
            data.t_cond = noise_level
        else:
            raise ValueError(f"Unknown t_cond {self.t_cond}")
        return data

    def __call__(self, data_list):
        selfies = [d["selfies"] for d in data_list]
        self.selfies_tokenizer.padding_side = "right"
        selfie_batch = self.selfies_tokenizer(
            selfies,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_sf_tokens,
            truncation=True,
            add_special_tokens=True,
        )

        batch_size = len(data_list)
        rdmol2selfies = [d.pop("rdmol2selfies") for d in data_list]
        rdmol2selfies_mask = [d.pop("rdmol2selfies_mask") for d in data_list]
        for d in data_list:
            d.pop("passed_conf_matching", None)

        data_batch = Batch.from_data_list(data_list)
        data_batch.max_seqlen = int((data_batch.ptr[1:] - data_batch.ptr[:-1]).max())
        data_batch = self.add_noise(data_batch)

        sf_max_len = selfie_batch.input_ids.shape[1]
        atom_max_len = int((data_batch.ptr[1:] - data_batch.ptr[:-1]).max())

        if self.load_mapping:
            padded_mask = rdmol2selfies_mask[0].new_zeros((batch_size, atom_max_len))
            padded_map = rdmol2selfies[0].new_zeros((batch_size, atom_max_len, sf_max_len))
            for i in range(batch_size):
                mask = rdmol2selfies_mask[i]
                padded_mask[i, : mask.shape[0]].copy_(mask)
                mapping = rdmol2selfies[i]
                padded_map[i, : mapping.shape[0], 1 : 1 + mapping.shape[1]].copy_(mapping)
            data_batch.rdmol2selfies = padded_map
            data_batch.rdmol2selfies_mask = padded_mask

        data_batch.x = data_batch.x.to(torch.float)
        return data_batch, selfie_batch


class QM9InferCollater:
    """Inference collater: random initial positions, no noise schedule step."""

    def __init__(
        self,
        max_atoms: int,
        max_sf_tokens: int,
        selfies_tokenizer,
        disable_com: bool = False,
        load_mapping: bool = True,
    ):
        self.max_sf_tokens = max_sf_tokens
        self.selfies_tokenizer = selfies_tokenizer
        self.disable_com = disable_com
        self.load_mapping = load_mapping

    def __call__(self, data_list):
        selfies = [d["selfies"] for d in data_list]
        self.selfies_tokenizer.padding_side = "right"
        selfie_batch = self.selfies_tokenizer(
            selfies,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_sf_tokens,
            truncation=True,
            add_special_tokens=True,
        )

        batch_size = len(data_list)
        rdmol2selfies = [d.pop("rdmol2selfies") for d in data_list]
        rdmol2selfies_mask = [d.pop("rdmol2selfies_mask") for d in data_list]
        for d in data_list:
            d.pop("passed_conf_matching", None)

        data_batch = Batch.from_data_list(data_list)
        data_batch.max_seqlen = int((data_batch.ptr[1:] - data_batch.ptr[:-1]).max())

        shape = (data_batch.x.shape[0], 3)
        if self.disable_com:
            data_batch.pos = torch.randn(shape)
        else:
            data_batch.pos = sample_com_rand_pos(shape, data_batch.batch)

        if self.load_mapping:
            sf_max_len = selfie_batch.input_ids.shape[1]
            atom_max_len = int((data_batch.ptr[1:] - data_batch.ptr[:-1]).max())
            padded_mask = rdmol2selfies_mask[0].new_zeros((batch_size, atom_max_len))
            padded_map = rdmol2selfies[0].new_zeros((batch_size, atom_max_len, sf_max_len))
            for i in range(batch_size):
                mask = rdmol2selfies_mask[i]
                padded_mask[i, : mask.shape[0]].copy_(mask)
                mapping = rdmol2selfies[i]
                padded_map[i, : mapping.shape[0], 1 : 1 + mapping.shape[1]].copy_(mapping)
            data_batch.rdmol2selfies = padded_map
            data_batch.rdmol2selfies_mask = padded_mask

        return data_batch, selfie_batch


class LMCollater:
    """SELFIES-only collater for MoLlama training."""

    def __init__(self, max_sf_tokens: int, selfies_tokenizer):
        self.max_sf_tokens = max_sf_tokens
        self.selfies_tokenizer = selfies_tokenizer

    def __call__(self, data_list):
        selfies = [d["selfies"] for d in data_list]
        self.selfies_tokenizer.padding_side = "right"
        return self.selfies_tokenizer(
            selfies,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_sf_tokens,
            truncation=True,
            add_special_tokens=True,
        )
