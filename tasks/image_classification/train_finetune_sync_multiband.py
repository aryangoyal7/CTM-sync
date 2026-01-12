import argparse
import os

import torch
from tqdm.auto import tqdm

from tasks.image_classification.train import get_dataset
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import image_classification_loss

from models.ctm_sync_filters import ContinuousThoughtMachineMultiBand
from tasks.image_classification.train_finetune_sync_common import (
    enable_requires_grad_by_prefix,
    get_trainable_params,
    load_checkpoint_forgiving,
    maybe_subset_dataset,
    set_requires_grad,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True, help="Path to old CTM checkpoint (.pt).")
    p.add_argument("--log_dir", type=str, default="logs/finetune_sync_multiband")
    p.add_argument("--dataset", type=str, default="imagenet", choices=["cifar10", "cifar100", "imagenet"])
    p.add_argument("--data_root", type=str, default="data/")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--batch_size_test", type=int, default=32)
    p.add_argument("--num_workers_train", type=int, default=2)
    p.add_argument("--device", type=int, default=-1, help="-1=cpu/mps, else cuda index")
    p.add_argument("--seed", type=int, default=412)
    p.add_argument("--subset_size", type=int, default=-1, help="If >0, fine-tune on a random subset of train.")
    p.add_argument("--subset_seed", type=int, default=0)

    # Model args
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--backbone_type", type=str, default="resnet18-4")
    p.add_argument("--d_input", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--iterations", type=int, default=75)
    p.add_argument("--positional_embedding_type", type=str, default="none")
    p.add_argument("--synapse_depth", type=int, default=4)
    p.add_argument("--n_synch_out", type=int, default=512)
    p.add_argument("--n_synch_action", type=int, default=512)
    p.add_argument("--neuron_select_type", type=str, default="random-pairing")
    p.add_argument("--n_random_pairing_self", type=int, default=0)
    p.add_argument("--memory_length", type=int, default=25)
    p.add_argument("--deep_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memory_hidden_dims", type=int, default=4)
    p.add_argument("--dropout_nlm", type=float, default=None)
    p.add_argument("--do_normalisation", action=argparse.BooleanOptionalAction, default=False)

    # Multiband specifics
    p.add_argument("--band_ks", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--fir_init", type=str, default="exp", choices=["exp", "zeros", "randn"])
    p.add_argument("--fir_exp_alpha", type=float, default=0.5)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, False)

    os.makedirs(args.log_dir, exist_ok=True)
    zip_python_code(f"{args.log_dir}/repo_state.zip")
    with open(f"{args.log_dir}/args.txt", "w") as f:
        print(args, file=f)

    if args.device != -1 and torch.cuda.is_available():
        device = f"cuda:{args.device}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_data, test_data, class_labels, _, _ = get_dataset(args.dataset, args.data_root)
    train_data = maybe_subset_dataset(train_data, args.subset_size, args.subset_seed)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers_train
    )
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=1, drop_last=False
    )

    prediction_reshaper = [-1]
    out_dims = len(class_labels)

    model = ContinuousThoughtMachineMultiBand(
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
        band_ks=args.band_ks,
        fir_init=args.fir_init,
        fir_exp_alpha=args.fir_exp_alpha,
    ).to(device)

    # Initialize lazy modules with the *new* sync dimensionality (P * n_bands)
    pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
    model(pseudo_inputs)

    # Old checkpoint has incompatible shapes for q_proj and output_projector due to increased sync dims.
    # Drop those keys so we can still reuse backbone/synapses/NLMs/attention weights.
    load_res = load_checkpoint_forgiving(
        model,
        args.checkpoint_path,
        map_location=device,
        strict=False,
        drop_prefixes=("q_proj.", "output_projector."),
    )
    print(f"Loaded checkpoint. Missing={len(load_res.missing_keys)} Unexpected={len(load_res.unexpected_keys)}")

    # Freeze everything; unfreeze multiband filters + the two projection heads that must adapt to new dims.
    set_requires_grad(model, False)
    enabled = enable_requires_grad_by_prefix(
        model,
        prefixes=("sync_filter_action", "sync_filter_out", "q_proj.", "output_projector."),
    )
    print(f"Trainable params enabled: {len(enabled)}")

    opt = torch.optim.Adam(get_trainable_params(model), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        with tqdm(total=len(trainloader), leave=False, dynamic_ncols=True) as pbar:
            for inputs, targets in trainloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions, certainties, _ = model(inputs)
                loss, _where = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
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
            for inputs, targets in testloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions, certainties, _ = model(inputs)
                _loss, where = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                preds = predictions.argmax(1)[torch.arange(predictions.size(0), device=predictions.device), where]
                correct += (preds == targets).sum().item()
                total += targets.numel()
        acc = correct / max(total, 1)

        ckpt_out = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_acc_most_certain": acc,
            "train_loss_mean": float(sum(losses) / max(len(losses), 1)),
        }
        torch.save(ckpt_out, f"{args.log_dir}/checkpoint_epoch{epoch+1}.pt")
        print(f"Epoch {epoch+1}: train_loss={ckpt_out['train_loss_mean']:.4f} val_acc={acc:.4f}")


if __name__ == "__main__":
    main()




