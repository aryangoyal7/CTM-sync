# Colab: Minimal ImageNet Fine-Tuning (Sync Filters)

This folder contains a Colab-friendly workflow to fine-tune the **new synchronization filter variants** (FIR / IIR / MultiBand) on a **tiny** ImageNet subset using the provided ImageNet checkpoint zip from Google Drive.

## What you get

- A notebook: `finetune_sync_imagenet_minimal.ipynb`
- Script runners (work in Colab + locally):
  - `run_finetune_sync_fir_colab.py`
  - `run_finetune_sync_iir_colab.py`
  - `run_finetune_sync_multiband_colab.py`
- A streaming ImageNet loader: `streaming_imagenet_min.py`

## Important notes

- **This uses streaming** (`datasets.load_dataset(..., streaming=True)`) and takes only the first `N` examples.
- **Results on tiny subsets are sanity-check / debug signal**, not a meaningful ImageNet benchmark.
- The Google Drive file you shared is a **zip** that contains the actual `.pt` checkpoint; our loader handles that automatically.

## Typical Colab commands (inside the notebook)

- Install deps
- Download checkpoint zip (gdown)
- Run one of:
  - `python colab/run_finetune_sync_fir_colab.py --checkpoint_path checkpoints/ctm_checkpoint.pt --n_train 2000 --n_val 500`
  - `python colab/run_finetune_sync_iir_colab.py --checkpoint_path checkpoints/ctm_checkpoint.pt --n_train 2000 --n_val 500`
  - `python colab/run_finetune_sync_multiband_colab.py --checkpoint_path checkpoints/ctm_checkpoint.pt --n_train 2000 --n_val 500`


