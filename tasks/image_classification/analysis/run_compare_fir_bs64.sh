#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tasks/image_classification/analysis/run_compare_fir_bs64.sh \
#     --base checkpoints/ctm_checkpoint.pt \
#     --finetuned logs/finetune_sync_fir_wandb/checkpoint_epoch7.pt \
#     --out logs/analysis_fir_bs64_epoch7 \
#     --n_val 5000
#
# Notes:
# - This runs BOTH baseline (untrained filters) and trained (finetuned filters) in one invocation.
# - Requires HF_TOKEN to access gated ImageNet-1k on HuggingFace.
# - Batch size 64 may OOM depending on GPU; if it does, try 32 or 16.

BASE=""
FINETUNED=""
OUT="logs/analysis_fir_bs64"
NVAL="5000"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --finetuned) FINETUNED="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --n_val) NVAL="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    *) echo "Unknown arg: $1" ; exit 1;;
  esac
done

if [[ -z "$BASE" || -z "$FINETUNED" ]]; then
  echo "Missing --base or --finetuned"
  exit 1
fi

mkdir -p "$OUT"

python -u -m tasks.image_classification.analysis.compare_sync_filters_analysis \
  --filter_type fir \
  --base_checkpoint "$BASE" \
  --finetuned_checkpoint "$FINETUNED" \
  --n_val "$NVAL" \
  --batch_size 64 \
  --certainty_threshold 0.8 \
  --topk 5 \
  --output_dir "$OUT" \
  --device "$DEVICE"

echo "Done. See:"
echo "  $OUT/baseline/steps_versus_correct_0.8.png"
echo "  $OUT/trained/steps_versus_correct_0.8.png"
echo "  $OUT/baseline/imagenet_calibration.png"
echo "  $OUT/trained/imagenet_calibration.png"


