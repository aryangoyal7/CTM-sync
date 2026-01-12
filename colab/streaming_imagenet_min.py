from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from datasets import load_dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _StreamingImageNetIterable(IterableDataset):
    """
    PyTorch IterableDataset wrapper around HuggingFace streaming ImageNet-1k.
    """

    def __init__(
        self,
        split: str,
        n: int,
        *,
        seed: int = 0,
        shuffle: bool = True,
        shuffle_buffer_size: int = 2000,
        transform=None,
    ):
        super().__init__()
        self.split = split
        self.n = int(n)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.transform = transform

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        # Note: imagenet-1k on HF is gated; you must authenticate first.
        # In Colab: `from huggingface_hub import notebook_login; notebook_login()`
        #
        # Also: datasets no longer supports `trust_remote_code` here.
        # Load token from colab/.env if available (recommended for Colab non-interactive workers)
        # Expected keys: HF_TOKEN or hf_token.
        try:
            from dotenv import load_dotenv  # type: ignore

            env_path = Path(__file__).resolve().parent / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=False)
        except Exception:
            # If python-dotenv isn't installed, we'll fall back to the environment or interactive login.
            pass

        token = os.environ.get("HF_TOKEN") or os.environ.get("hf_token")
        ds = load_dataset("imagenet-1k", split=self.split, streaming=True, token=(token or True))
        if self.shuffle:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)

        it = ds.take(self.n)
        for ex in it:
            img = ex["image"].convert("RGB")
            y = int(ex["label"])
            x = self.transform(img) if self.transform is not None else img
            yield x, y


@dataclass
class MiniImageNetLoaders:
    trainloader: DataLoader
    valloader: DataLoader


def make_imagenet_transforms(image_size: int = 224):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def get_min_imagenet_loaders(
    *,
    n_train: int,
    n_val: int,
    batch_size: int,
    num_workers: int = 2,
    seed: int = 0,
    image_size: int = 224,
    shuffle_buffer_size: int = 2000,
) -> MiniImageNetLoaders:
    train_t, val_t = make_imagenet_transforms(image_size=image_size)

    train_ds = _StreamingImageNetIterable(
        split="train",
        n=n_train,
        seed=seed,
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
        transform=train_t,
    )
    val_ds = _StreamingImageNetIterable(
        split="validation",
        n=n_val,
        seed=seed + 1,
        shuffle=False,
        shuffle_buffer_size=shuffle_buffer_size,
        transform=val_t,
    )

    # IterableDataset: DataLoader must have shuffle=False.
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return MiniImageNetLoaders(trainloader=trainloader, valloader=valloader)


