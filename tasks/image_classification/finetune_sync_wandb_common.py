from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

from tasks.image_classification.train_finetune_sync_common import (
    extract_state_dict_from_checkpoint,
    load_checkpoint_raw,
)


def ensure_wandb() -> Any:
    try:
        import wandb  # type: ignore

        return wandb
    except Exception as e:
        raise RuntimeError(
            "wandb is not installed. Install with: pip install wandb\n"
            "Or remove/disable wandb logging."
        ) from e


def maybe_set_hf_token(hf_token: Optional[str]) -> None:
    if hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token


def _coerce_args_dict(maybe_args: Any) -> Dict[str, Any]:
    if maybe_args is None:
        return {}
    if isinstance(maybe_args, dict):
        return maybe_args
    # argparse.Namespace
    if hasattr(maybe_args, "__dict__"):
        return dict(vars(maybe_args))
    return {}


def infer_arch_from_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Returns:
      inferred_cfg: best-effort CTM config dict (prefers checkpoint['args'] if present)
      sd: state dict (model weights)
    """
    ckpt = load_checkpoint_raw(checkpoint_path, map_location="cpu")
    ckpt_args = _coerce_args_dict(ckpt.get("args", None))
    sd = extract_state_dict_from_checkpoint(ckpt, checkpoint_path=checkpoint_path)

    d_model = int(sd["start_activated_state"].numel()) if "start_activated_state" in sd else None
    d_input = None
    w = sd.get("synapses.first_projection.0.weight", None)
    if w is not None and d_model is not None:
        d_input = int(w.shape[1]) - d_model

    down_idxs = []
    for k in sd.keys():
        if k.startswith("synapses.down_projections."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                down_idxs.append(int(parts[2]))
    synapse_depth = (max(down_idxs) + 2) if down_idxs else None

    p_out = int(sd.get("decay_params_out").numel()) if "decay_params_out" in sd else None
    p_action = int(sd.get("decay_params_action").numel()) if "decay_params_action" in sd else None
    left_out = int(sd.get("out_neuron_indices_left").numel()) if "out_neuron_indices_left" in sd else None
    left_action = int(sd.get("action_neuron_indices_left").numel()) if "action_neuron_indices_left" in sd else None

    neuron_select_type = None
    n_synch_out = None
    n_synch_action = None

    if p_out is not None and left_out is not None:
        if p_out == left_out:
            neuron_select_type = "random-pairing"
            n_synch_out = left_out
        elif p_out == (left_out * (left_out + 1)) // 2:
            n_synch_out = left_out
    if p_action is not None and left_action is not None:
        if p_action == left_action:
            neuron_select_type = neuron_select_type or "random-pairing"
            n_synch_action = left_action
        elif p_action == (left_action * (left_action + 1)) // 2:
            n_synch_action = left_action

    try:
        out_left = sd.get("out_neuron_indices_left", None)
        act_left = sd.get("action_neuron_indices_left", None)
        if d_model is not None and out_left is not None and act_left is not None:
            n_out = int(out_left.numel())
            n_act = int(act_left.numel())
            if (
                torch.equal(out_left.cpu(), torch.arange(0, n_out))
                and torch.equal(act_left.cpu(), torch.arange(d_model - n_act, d_model))
            ):
                neuron_select_type = "first-last"
    except Exception:
        pass

    inferred = {
        # Prefer explicit checkpoint args when available, else fall back to inference.
        "d_model": ckpt_args.get("d_model", d_model),
        "d_input": ckpt_args.get("d_input", d_input),
        "synapse_depth": ckpt_args.get("synapse_depth", synapse_depth),
        "neuron_select_type": ckpt_args.get("neuron_select_type", neuron_select_type),
        "n_synch_out": ckpt_args.get("n_synch_out", n_synch_out),
        "n_synch_action": ckpt_args.get("n_synch_action", n_synch_action),
        "iterations": ckpt_args.get("iterations", None),
        "heads": ckpt_args.get("heads", None),
        "memory_length": ckpt_args.get("memory_length", None),
        "memory_hidden_dims": ckpt_args.get("memory_hidden_dims", None),
        "backbone_type": ckpt_args.get("backbone_type", None),
        "positional_embedding_type": ckpt_args.get("positional_embedding_type", None),
        "n_random_pairing_self": ckpt_args.get("n_random_pairing_self", None),
        # ckpt uses `deep_memory`; CTM uses `deep_nlms`
        "deep_nlms": ckpt_args.get("deep_memory", None),
        "out_dims": ckpt_args.get("out_dims", None),
    }
    return inferred, sd


def get_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


