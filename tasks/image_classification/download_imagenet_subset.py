import argparse
import os

from datasets import load_dataset


def build_split(base_split, max_samples, subset_percent):
    if max_samples is not None and max_samples > 0:
        return f"{base_split}[:{int(max_samples)}]"
    if subset_percent is not None and subset_percent > 0:
        pct = "%g" % subset_percent
        return f"{base_split}[:{pct}%]"
    return base_split


def parse_args():
    parser = argparse.ArgumentParser(description="Download and store ImageNet subset locally via HuggingFace datasets.")
    parser.add_argument("--output_root", type=str, required=True, help="Output directory to store train/validation subsets.")
    parser.add_argument("--hf_cache_dir", type=str, default="", help="Optional HF cache directory.")
    parser.add_argument("--train_split", type=str, default="train", help="Base train split.")
    parser.add_argument("--val_split", type=str, default="validation", help="Base validation split.")
    parser.add_argument("--train_max_samples", type=int, default=-1, help="Use first N train samples.")
    parser.add_argument("--val_max_samples", type=int, default=-1, help="Use first N validation samples.")
    parser.add_argument("--train_subset_percent", type=float, default=-1.0, help="Use first X%% train samples when max_samples is unset.")
    parser.add_argument("--val_subset_percent", type=float, default=-1.0, help="Use first X%% validation samples when max_samples is unset.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    train_split = build_split(args.train_split, args.train_max_samples, args.train_subset_percent)
    val_split = build_split(args.val_split, args.val_max_samples, args.val_subset_percent)
    print(f"Resolved train split: {train_split}")
    print(f"Resolved val split: {val_split}")

    load_kwargs = {"path": "imagenet-1k", "trust_remote_code": True}
    if args.hf_cache_dir:
        load_kwargs["cache_dir"] = args.hf_cache_dir

    train_ds = load_dataset(split=train_split, **load_kwargs)
    val_ds = load_dataset(split=val_split, **load_kwargs)

    train_out = os.path.join(args.output_root, "train")
    val_out = os.path.join(args.output_root, "validation")
    train_ds.save_to_disk(train_out)
    val_ds.save_to_disk(val_out)
    print(f"Saved train subset to: {train_out}")
    print(f"Saved validation subset to: {val_out}")


if __name__ == "__main__":
    main()
