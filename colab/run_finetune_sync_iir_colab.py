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
    load_checkpoint_forgiving,
    set_requires_grad,
)
from colab.streaming_imagenet_min import get_min_imagenet_loaders


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

    # Training
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)

    # CTM (match ImageNet checkpoint expectations by default)
    p.add_argument("--d_model", type=int, default=4096)
    p.add_argument("--d_input", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--synapse_depth", type=int, default=4)
    p.add_argument("--memory_length", type=int, default=25)
    p.add_argument("--deep_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory_hidden_dims", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--dropout_nlm", type=float, default=None)
    p.add_argument("--do_normalisation", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--backbone_type", type=str, default="resnet18-4")
    p.add_argument("--positional_embedding_type", type=str, default="none")
    p.add_argument("--n_synch_out", type=int, default=512)
    p.add_argument("--n_synch_action", type=int, default=512)
    p.add_argument("--neuron_select_type", type=str, default="random-pairing")
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
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=args.deep_memory,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=args.do_normalisation,
        backbone_type=args.backbone_type,
        positional_embedding_type=args.positional_embedding_type,
        out_dims=out_dims,
        prediction_reshaper=prediction_reshaper,
        dropout=args.dropout,
        dropout_nlm=args.dropout_nlm,
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
        iir_alpha_init=args.iir_alpha_init,
        iir_eps=args.iir_eps,
    ).to(device)

    xb, yb = next(iter(loaders.trainloader))
    model(xb.to(device))

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


