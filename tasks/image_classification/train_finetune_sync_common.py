from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

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
    report_mismatches: int = 0,
) -> torch.nn.modules.module._IncompatibleKeys:
    ckpt_path = resolve_checkpoint_path(checkpoint_path)

    # PyTorch 2.6+ defaults `weights_only=True`, which can fail on checkpoints that contain
    # non-tensor objects (e.g. argparse.Namespace in ckpt["args"]). We trust our own checkpoints
    # here and explicitly request `weights_only=False` when supported.
    try:
        ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location=map_location)
    state_dict = extract_state_dict_from_checkpoint(ckpt, checkpoint_path=checkpoint_path)

    # Handle DDP 'module.' prefix if present.
    has_module_prefix = all(k.startswith("module.") for k in state_dict.keys())
    if has_module_prefix:
        state_dict = {k.partition("module.")[2]: v for k, v in state_dict.items()}

    if drop_prefixes:
        state_dict = {k: v for k, v in state_dict.items() if not any(k.startswith(p) for p in drop_prefixes)}

    # IMPORTANT: torch will still error on size mismatches even with strict=False.
    # Filter out any tensors whose shapes don't match the target model.
    target_sd = model.state_dict()
    filtered = {}
    dropped = 0
    mismatches: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for k, v in state_dict.items():
        tv = target_sd.get(k, None)
        if tv is None:
            continue
        if hasattr(tv, "shape") and hasattr(v, "shape") and tuple(tv.shape) != tuple(v.shape):
            dropped += 1
            if report_mismatches and len(mismatches) < report_mismatches:
                try:
                    mismatches.append((k, tuple(v.shape), tuple(tv.shape)))
                except Exception:
                    pass
            continue
        filtered[k] = v

    if dropped:
        print(f"[load_checkpoint_forgiving] Dropped {dropped} keys due to shape mismatch.")
        if mismatches:
            print("[load_checkpoint_forgiving] Example mismatches (checkpoint_shape -> model_shape):")
            for k, cs, ms in mismatches:
                print(f"  - {k}: {cs} -> {ms}")

    return model.load_state_dict(filtered, strict=strict)


def resolve_checkpoint_path(checkpoint_path: str) -> Path:
    """
    If `checkpoint_path` is a zip bundle (e.g. the Drive download), extract the inner .pt and return its path.
    Otherwise return the original path.

    IMPORTANT:
    - PyTorch checkpoints saved with `torch.save(...)` may themselves be stored in a zip-based container format.
      Those files are valid checkpoints and should be passed directly to `torch.load` (no extraction).
    - We only extract when the zip contains a nested checkpoint file (*.pt/*.pth/*.bin), like the Google Drive bundle.
    """
    ckpt_path = Path(checkpoint_path)
    if zipfile.is_zipfile(ckpt_path):
        with zipfile.ZipFile(ckpt_path, "r") as zf:
            members = [
                m
                for m in zf.namelist()
                if (not m.endswith("/"))
                and (not m.startswith("__MACOSX/"))
                and (m.lower().endswith(".pt") or m.lower().endswith(".pth") or m.lower().endswith(".bin"))
            ]
            # If no nested checkpoint is present, this is likely a normal torch checkpoint in zip container format.
            # In that case, do NOT extract; just let torch.load handle it.
            if len(members) == 0:
                return ckpt_path
            inner = members[0] if len(members) == 1 else members[0]
            extracted = ckpt_path.parent / inner
            extracted.parent.mkdir(parents=True, exist_ok=True)
            if not extracted.exists() or extracted.stat().st_size == 0:
                zf.extract(inner, path=ckpt_path.parent)
            return extracted
    return ckpt_path


def extract_state_dict_from_checkpoint(ckpt: Any, *, checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt  # raw state_dict
    raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")


def load_checkpoint_raw(checkpoint_path: str, *, map_location: str) -> Dict[str, Any]:
    """
    Load a checkpoint dict (handling Drive zip bundles).
    """
    ckpt_path = resolve_checkpoint_path(checkpoint_path)
    # PyTorch 2.6+ defaults `weights_only=True`. Explicitly request `weights_only=False` when supported
    # to allow loading common training checkpoints that include metadata like argparse.Namespace.
    try:
        ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected checkpoint dict at {checkpoint_path}, got {type(ckpt)}")
    return ckpt


def maybe_subset_dataset(dataset, subset_size: int, seed: int):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=subset_size, replace=False)
    return torch.utils.data.Subset(dataset, idx.tolist())




