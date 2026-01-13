import argparse
import os

import torch
from tqdm.auto import tqdm

from models.ctm_sync_filters import ContinuousThoughtMachineIIR
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
    p.add_argument("--ckpt_report_mismatches", type=int, default=0)

    # data
    p.add_argument("--n_train", type=int, default=50000)
    p.add_argument("--n_val", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--shuffle_buffer_size", type=int, default=10000)

    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=412)
    p.add_argument("--log_dir", type=str, default="logs/finetune_sync_iir_wandb")

    # IIR
    p.add_argument("--iir_alpha_init", type=float, default=0.9)
    p.add_argument("--iir_eps", type=float, default=1e-4)

    # wandb
    p.add_argument("--wandb_project", type=str, default="ctm-sync")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_tags", type=str, nargs="*", default=["iir"])

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

    out_dims = int(inferred.get("out_dims") or len(IMAGENET2012_CLASSES))
    model = ContinuousThoughtMachineIIR(
        iterations=int(inferred.get("iterations") or 50),
        d_model=int(inferred.get("d_model") or 4096),
        d_input=int(inferred.get("d_input") or 1024),
        heads=int(inferred.get("heads") or 16),
        n_synch_out=int(inferred.get("n_synch_out") or 512),
        n_synch_action=int(inferred.get("n_synch_action") or 512),
        synapse_depth=int(inferred.get("synapse_depth") or 8),
        memory_length=int(inferred.get("memory_length") or 25),
        deep_nlms=bool(True if inferred.get("deep_nlms") is None else inferred.get("deep_nlms")),
        memory_hidden_dims=int(inferred.get("memory_hidden_dims") or 64),
        do_layernorm_nlm=False,
        backbone_type=str(inferred.get("backbone_type") or "resnet152-4"),
        positional_embedding_type=str(inferred.get("positional_embedding_type") or "none"),
        out_dims=out_dims,
        prediction_reshaper=[-1],
        dropout=0.0,
        dropout_nlm=None,
        neuron_select_type=inferred.get("neuron_select_type", "random-pairing") or "random-pairing",
        n_random_pairing_self=int(inferred.get("n_random_pairing_self") or 0),
        iir_alpha_init=args.iir_alpha_init,
        iir_eps=args.iir_eps,
    ).to(device)

    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    model(dummy)
    load_res = load_checkpoint_forgiving(
        model,
        args.checkpoint_path,
        map_location=device,
        strict=False,
        report_mismatches=args.ckpt_report_mismatches,
    )

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
            alpha = torch.sigmoid(model.sync_filter_out.raw_alpha).detach().cpu()
            wandb.log(
                {
                    "val/acc_most_certain": acc,
                    "filters/iir_alpha_mean": float(alpha.mean()),
                    "filters/iir_alpha_std": float(alpha.std()),
                },
                step=global_step,
            )

        torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, f"{args.log_dir}/checkpoint_epoch{epoch}.pt")
        print(f"Epoch {epoch}: val_acc={acc:.4f} trainable_params={trainable} missing={len(load_res.missing_keys)}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()


