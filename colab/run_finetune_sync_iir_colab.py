import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is importable even when running as: python colab/...
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from tqdm.auto import tqdm

from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES
from utils.housekeeping import set_seed
from utils.losses import image_classification_loss

from models.ctm_sync_filters import ContinuousThoughtMachineIIR
from tasks.image_classification.train_finetune_sync_common import (
    enable_requires_grad_by_prefix,
    get_trainable_params,
    load_checkpoint_raw,
    extract_state_dict_from_checkpoint,
    load_checkpoint_forgiving,
    set_requires_grad,
)
from colab.streaming_imagenet_min import get_min_imagenet_loaders


def _maybe_load_hf_token_from_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path), override=False)
    except Exception:
        pass


def _infer_arch_from_state_dict(sd: dict) -> dict:
    d_model = int(sd["start_activated_state"].numel()) if "start_activated_state" in sd else None
    d_input = None
    w = sd.get("synapses.first_projection.0.weight", None)
    if w is not None and d_model is not None:
        in_features = int(w.shape[1])
        d_input = in_features - d_model

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

    return {
        "d_model": d_model,
        "d_input": d_input,
        "synapse_depth": synapse_depth,
        "neuron_select_type": neuron_select_type,
        "n_synch_out": n_synch_out,
        "n_synch_action": n_synch_action,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint (.pt OR the Drive zip .pt).")
    p.add_argument("--log_dir", type=str, default="logs/colab_finetune_sync_iir")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=412)

    # Minimal streaming ImageNet
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    # NOTE: keep this 0 by default so HuggingFace auth/token works reliably in Colab.
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--shuffle_buffer_size", type=int, default=2000)
    p.add_argument("--hf_token", type=str, default=None, help="Optional HF token (preferred: set HF_TOKEN env / colab/.env).")

    # Training
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)

    # CTM (auto-inferred from checkpoint unless explicitly overridden)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--d_input", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--synapse_depth", type=int, default=None)
    p.add_argument("--memory_length", type=int, default=None)
    p.add_argument("--deep_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory_hidden_dims", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--dropout_nlm", type=float, default=None)
    p.add_argument("--do_normalisation", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--backbone_type", type=str, default=None)
    p.add_argument("--positional_embedding_type", type=str, default=None)
    p.add_argument("--n_synch_out", type=int, default=None)
    p.add_argument("--n_synch_action", type=int, default=None)
    p.add_argument("--neuron_select_type", type=str, default=None)
    p.add_argument("--n_random_pairing_self", type=int, default=0)

    # IIR specifics
    p.add_argument("--iir_alpha_init", type=float, default=0.9)
    p.add_argument("--iir_eps", type=float, default=1e-4)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, False)
    os.makedirs(args.log_dir, exist_ok=True)

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    _maybe_load_hf_token_from_dotenv()
    if args.hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = args.hf_token

    ckpt = load_checkpoint_raw(args.checkpoint_path, map_location="cpu")
    sd = extract_state_dict_from_checkpoint(ckpt, checkpoint_path=args.checkpoint_path)
    inferred = _infer_arch_from_state_dict(sd)
    ckpt_args = ckpt.get("args", None)

    def _get(name, default=None):
        if getattr(args, name) is not None:
            return getattr(args, name)
        if ckpt_args is not None and hasattr(ckpt_args, name):
            return getattr(ckpt_args, name)
        if ckpt_args is not None and isinstance(ckpt_args, dict) and name in ckpt_args:
            return ckpt_args[name]
        return inferred.get(name, default)

    loaders = get_min_imagenet_loaders(
        n_train=args.n_train,
        n_val=args.n_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    out_dims = len(IMAGENET2012_CLASSES)
    prediction_reshaper = [-1]

    model = ContinuousThoughtMachineIIR(
        iterations=_get("iterations", 50),
        d_model=_get("d_model", 4096),
        d_input=_get("d_input", 1024),
        heads=_get("heads", 4),
        n_synch_out=_get("n_synch_out", 512),
        n_synch_action=_get("n_synch_action", 512),
        synapse_depth=_get("synapse_depth", 8),
        memory_length=_get("memory_length", 25),
        deep_nlms=args.deep_memory,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=args.do_normalisation,
        backbone_type=_get("backbone_type", "resnet18-4"),
        positional_embedding_type=_get("positional_embedding_type", "none"),
        out_dims=out_dims,
        prediction_reshaper=prediction_reshaper,
        dropout=args.dropout,
        dropout_nlm=args.dropout_nlm,
        neuron_select_type=_get("neuron_select_type", "random-pairing"),
        n_random_pairing_self=args.n_random_pairing_self,
        iir_alpha_init=args.iir_alpha_init,
        iir_eps=args.iir_eps,
    ).to(device)

    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    model(dummy)

    load_res = load_checkpoint_forgiving(model, args.checkpoint_path, map_location=device, strict=False)
    print(f"Loaded checkpoint. Missing={len(load_res.missing_keys)} Unexpected={len(load_res.unexpected_keys)}")

    set_requires_grad(model, False)
    enabled = enable_requires_grad_by_prefix(model, prefixes=("sync_filter_action", "sync_filter_out"))
    print(f"Trainable params enabled: {len(enabled)}")
    opt = torch.optim.Adam(get_trainable_params(model), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        with tqdm(total=args.n_train // max(args.batch_size, 1), leave=False, dynamic_ncols=True) as pbar:
            for xb, yb in loaders.trainloader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds, certs, _ = model(xb)
                loss, _where = image_classification_loss(preds, certs, yb, use_most_certain=True)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                pbar.set_description(f"epoch={epoch+1}/{args.epochs} loss={loss.item():.4f}")
                pbar.update(1)

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for xb, yb in loaders.valloader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds, certs, _ = model(xb)
                _loss, where = image_classification_loss(preds, certs, yb, use_most_certain=True)
                yhat = preds.argmax(1)[torch.arange(preds.size(0), device=preds.device), where]
                correct += (yhat == yb).sum().item()
                total += yb.numel()
        acc = correct / max(total, 1)
        mean_loss = float(sum(losses) / max(len(losses), 1))
        print(f"Epoch {epoch+1}: train_loss={mean_loss:.4f} val_acc={acc:.4f}")
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, f"{args.log_dir}/checkpoint_epoch{epoch+1}.pt")


if __name__ == "__main__":
    main()


