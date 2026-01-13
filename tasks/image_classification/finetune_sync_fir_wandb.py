import argparse
import os

import torch
from tqdm.auto import tqdm

from models.ctm_sync_filters import ContinuousThoughtMachineFIR
from tasks.image_classification.finetune_sync_wandb_common import (
    count_trainable_params,
    ensure_wandb,
    get_device,
    infer_arch_from_checkpoint,
    maybe_set_hf_token,
)
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES
from tasks.image_classification.streaming_imagenet import get_streaming_imagenet_loaders
from tasks.image_classification.train_finetune_sync_common import (
    enable_requires_grad_by_prefix,
    get_trainable_params,
    load_checkpoint_forgiving,
    set_requires_grad,
)
from utils.housekeeping import set_seed
from utils.losses import image_classification_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--hf_token", type=str, default=None)

    # data
    p.add_argument("--n_train", type=int, default=50000)
    p.add_argument("--n_val", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--shuffle_buffer_size", type=int, default=10000)

    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=412)
    p.add_argument("--log_dir", type=str, default="logs/finetune_sync_fir_wandb")

    # FIR
    p.add_argument("--fir_k", type=int, default=16)
    p.add_argument("--fir_init", type=str, default="exp", choices=["exp", "zeros", "randn"])
    p.add_argument("--fir_exp_alpha", type=float, default=0.5)

    # wandb
    p.add_argument("--wandb_project", type=str, default="ctm-sync")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_tags", type=str, nargs="*", default=["fir"])

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, False)
    os.makedirs(args.log_dir, exist_ok=True)

    maybe_set_hf_token(args.hf_token)
    device = get_device(args.device)

    inferred, _sd = infer_arch_from_checkpoint(args.checkpoint_path)

    loaders = get_streaming_imagenet_loaders(
        n_train=args.n_train,
        n_val=args.n_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )

    out_dims = len(IMAGENET2012_CLASSES)
    model = ContinuousThoughtMachineFIR(
        iterations=50,  # checkpoint default; keep stable
        d_model=inferred.get("d_model", 4096),
        d_input=inferred.get("d_input", 1024),
        heads=4,
        n_synch_out=inferred.get("n_synch_out", 512),
        n_synch_action=inferred.get("n_synch_action", 512),
        synapse_depth=inferred.get("synapse_depth", 8),
        memory_length=25,
        deep_nlms=True,
        memory_hidden_dims=4,
        do_layernorm_nlm=False,
        backbone_type="resnet18-4",
        positional_embedding_type="none",
        out_dims=out_dims,
        prediction_reshaper=[-1],
        dropout=0.0,
        dropout_nlm=None,
        neuron_select_type=inferred.get("neuron_select_type", "random-pairing") or "random-pairing",
        n_random_pairing_self=0,
        fir_k=args.fir_k,
        fir_init=args.fir_init,
        fir_exp_alpha=args.fir_exp_alpha,
    ).to(device)

    # init lazies
    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    model(dummy)

    load_res = load_checkpoint_forgiving(model, args.checkpoint_path, map_location=device, strict=False)

    set_requires_grad(model, False)
    enable_requires_grad_by_prefix(model, prefixes=("sync_filter_action", "sync_filter_out"))
    trainable = count_trainable_params(model)

    wandb = None
    run = None
    if args.wandb_mode != "disabled":
        wandb = ensure_wandb()
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            tags=args.wandb_tags,
            config={**vars(args), "inferred": inferred, "trainable_params": trainable},
        )

    opt = torch.optim.Adam(get_trainable_params(model), lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loaders.trainloader, total=args.n_train // max(args.batch_size, 1), dynamic_ncols=True)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            preds, certs, _ = model(xb)
            loss, _where = image_classification_loss(preds, certs, yb, use_most_certain=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            pbar.set_description(f"epoch={epoch} step={global_step} loss={loss.item():.4f}")
            if wandb:
                wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)

        # val
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
        if wandb:
            # log filter stats
            wandb.log(
                {
                    "val/acc_most_certain": acc,
                    "filters/fir_raw_w_mean": float(model.sync_filter_out.raw_w.mean().detach().cpu()),
                },
                step=global_step,
            )

        torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, f"{args.log_dir}/checkpoint_epoch{epoch}.pt")
        print(f"Epoch {epoch}: val_acc={acc:.4f} trainable_params={trainable} missing={len(load_res.missing_keys)}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()


