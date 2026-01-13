import argparse
import os
import subprocess
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="ctm-sync")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--n_train", type=int, default=50000)
    p.add_argument("--n_val", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--base_lr_fir", type=float, default=1e-3)
    p.add_argument("--base_lr_iir", type=float, default=1e-4)
    p.add_argument("--base_lr_mb", type=float, default=1e-3)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2], help="GPU ids to use for FIR/IIR/MB respectively.")
    return p.parse_args()


def _env_for_gpu(gpu: int, hf_token: str | None) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if hf_token:
        env.setdefault("HF_TOKEN", hf_token)
    return env


def main():
    args = parse_args()
    scripts = [
        ("fir", "tasks.image_classification.finetune_sync_fir_wandb", args.base_lr_fir),
        ("iir", "tasks.image_classification.finetune_sync_iir_wandb", args.base_lr_iir),
        ("multiband-reduced", "tasks.image_classification.finetune_sync_multiband_reduced_wandb", args.base_lr_mb),
    ]
    if len(args.gpus) < 3:
        raise ValueError("Need 3 GPU ids in --gpus for FIR/IIR/MB.")

    procs: List[subprocess.Popen] = []
    for (tag, module, lr), gpu in zip(scripts, args.gpus):
        cmd = [
            "python",
            "-m",
            module,
            "--checkpoint_path",
            args.checkpoint_path,
            "--n_train",
            str(args.n_train),
            "--n_val",
            str(args.n_val),
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(lr),
            "--wandb_project",
            args.wandb_project,
            "--wandb_mode",
            args.wandb_mode,
            "--wandb_tags",
            tag,
        ]
        if args.wandb_entity:
            cmd += ["--wandb_entity", args.wandb_entity]
        if args.hf_token:
            cmd += ["--hf_token", args.hf_token]

        print("Launching:", " ".join(cmd), "on GPU", gpu)
        procs.append(subprocess.Popen(cmd, env=_env_for_gpu(gpu, args.hf_token)))

    exit_codes = [p.wait() for p in procs]
    print("Exit codes:", exit_codes)
    if any(c != 0 for c in exit_codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()


