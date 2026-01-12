from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad = requires_grad


def enable_requires_grad_by_prefix(model: nn.Module, prefixes: Sequence[str]) -> List[str]:
    enabled = []
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in prefixes):
            p.requires_grad = True
            enabled.append(name)
    return enabled


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def load_checkpoint_forgiving(
    model: nn.Module,
    checkpoint_path: str,
    *,
    map_location: str,
    strict: bool = False,
    drop_prefixes: Sequence[str] = (),
) -> torch.nn.modules.module._IncompatibleKeys:
    ckpt_path = Path(checkpoint_path)

    # The provided Google Drive "checkpoint.pt" is sometimes a zip bundle containing the actual .pt.
    if zipfile.is_zipfile(ckpt_path):
        with zipfile.ZipFile(ckpt_path, "r") as zf:
            members = [
                m
                for m in zf.namelist()
                if (not m.endswith("/"))
                and (not m.startswith("__MACOSX/"))
                and (m.lower().endswith(".pt") or m.lower().endswith(".pth") or m.lower().endswith(".bin"))
            ]
            if len(members) == 0:
                raise ValueError(f"No .pt/.pth/.bin files found inside zip checkpoint: {checkpoint_path}")

            # Prefer a single inner checkpoint if present.
            inner = members[0] if len(members) == 1 else members[0]
            extracted = ckpt_path.parent / inner
            extracted.parent.mkdir(parents=True, exist_ok=True)
            if not extracted.exists() or extracted.stat().st_size == 0:
                zf.extract(inner, path=ckpt_path.parent)
            ckpt_path = extracted

    ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # Might already be a raw state_dict
        state_dict = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")

    # Handle DDP 'module.' prefix if present.
    has_module_prefix = all(k.startswith("module.") for k in state_dict.keys())
    if has_module_prefix:
        state_dict = {k.partition("module.")[2]: v for k, v in state_dict.items()}

    if drop_prefixes:
        state_dict = {k: v for k, v in state_dict.items() if not any(k.startswith(p) for p in drop_prefixes)}

    return model.load_state_dict(state_dict, strict=strict)


def maybe_subset_dataset(dataset, subset_size: int, seed: int):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=subset_size, replace=False)
    return torch.utils.data.Subset(dataset, idx.tolist())




