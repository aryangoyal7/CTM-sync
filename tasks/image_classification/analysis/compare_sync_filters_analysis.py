"""
Compare trained vs untrained learnable sync filters on ImageNet-1k validation and reproduce:
  (a) Adaptive compute potential via certainty threshold (e.g., 0.8): %correct/%incorrect above threshold per tick
  (b) Calibration curves per internal tick (mean predicted probability vs ratio of positives)

This mirrors the logic in `tasks/image_classification/analysis/run_imagenet_analysis.py`,
but runs for two models:
  - baseline: base checkpoint + freshly initialized sync filters
  - trained:  base checkpoint + sync filter weights loaded from a finetune checkpoint
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from models.ctm_sync_filters import (
    ContinuousThoughtMachineFIR,
    ContinuousThoughtMachineIIR,
    ContinuousThoughtMachineMultiBandReduced,
)
from tasks.image_classification.finetune_sync_wandb_common import get_device, infer_arch_from_checkpoint
from tasks.image_classification.streaming_imagenet import get_streaming_imagenet_loaders
from tasks.image_classification.train_finetune_sync_common import (
    extract_state_dict_from_checkpoint,
    load_checkpoint_forgiving,
    load_checkpoint_raw,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_checkpoint", type=str, required=True, help="Base ImageNet CTM checkpoint (zip or .pt).")
    p.add_argument(
        "--finetuned_checkpoint",
        type=str,
        required=True,
        help="Checkpoint from finetune run (e.g., logs/.../checkpoint_epochX.pt).",
    )
    p.add_argument("--filter_type", type=str, required=True, choices=["fir", "iir", "multiband-reduced"])

    # dataset / eval
    p.add_argument("--n_val", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=412)

    # analysis settings
    p.add_argument("--certainty_threshold", type=float, default=0.8)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--calib_bins", type=int, default=10)

    # filter hyperparams (must match training run)
    p.add_argument("--fir_k", type=int, default=16)
    p.add_argument("--fir_init", type=str, default="exp", choices=["exp", "zeros", "randn"])
    p.add_argument("--fir_exp_alpha", type=float, default=0.5)
    p.add_argument("--iir_alpha_init", type=float, default=0.9)
    p.add_argument("--iir_eps", type=float, default=1e-4)
    p.add_argument("--band_ks", type=int, nargs="+", default=[8, 16, 32])

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--output_dir", type=str, default="logs/compare_sync_filters")
    return p.parse_args()


def _build_model(args, inferred: Dict) -> torch.nn.Module:
    common = dict(
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
        out_dims=int(inferred.get("out_dims") or 1000),
        prediction_reshaper=[-1],
        dropout=0.0,
        dropout_nlm=None,
        neuron_select_type=inferred.get("neuron_select_type", "random-pairing") or "random-pairing",
        n_random_pairing_self=int(inferred.get("n_random_pairing_self") or 0),
    )

    if args.filter_type == "fir":
        return ContinuousThoughtMachineFIR(
            **common,
            fir_k=args.fir_k,
            fir_init=args.fir_init,
            fir_exp_alpha=args.fir_exp_alpha,
        )
    if args.filter_type == "iir":
        return ContinuousThoughtMachineIIR(
            **common,
            iir_alpha_init=args.iir_alpha_init,
            iir_eps=args.iir_eps,
        )
    if args.filter_type == "multiband-reduced":
        return ContinuousThoughtMachineMultiBandReduced(
            **common,
            band_ks=args.band_ks,
            fir_init=args.fir_init,
            fir_exp_alpha=args.fir_exp_alpha,
        )
    raise ValueError(args.filter_type)


def _load_only_sync_filters(model: torch.nn.Module, finetuned_checkpoint: str, *, map_location: str) -> None:
    ckpt = load_checkpoint_raw(finetuned_checkpoint, map_location=map_location)
    sd = extract_state_dict_from_checkpoint(ckpt, checkpoint_path=finetuned_checkpoint)
    filt = {k: v for k, v in sd.items() if k.startswith("sync_filter_action") or k.startswith("sync_filter_out")}
    res = model.load_state_dict(filt, strict=False)
    print(
        f"[load_only_sync_filters] loaded={len(filt)} missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
    )


@dataclass
class AdaptiveStats:
    correct_frac_per_tick: np.ndarray  # (T,)
    incorrect_frac_per_tick: np.ndarray  # (T,)


@dataclass
class CalibrationStats:
    bin_centers_per_tick: List[np.ndarray]
    accuracies_per_tick: List[np.ndarray]


def _compute_batch_stats(
    logits: torch.Tensor, certainties: torch.Tensor, targets: torch.Tensor, *, threshold: float, topk: int, calib_bins: int
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    logits: (B, C, T)
    certainties: (B, 2, T) where certainties[:,1,:] is 1-normalised_entropy (higher = more certain)
    targets: (B,)
    Returns:
      correct_counts_per_tick, incorrect_counts_per_tick: (T,) counts (not normalised)
      calib_bin_sum_probs_per_tick: list[T] of (bins,) float sums
      calib_bin_sum_correct_per_tick: list[T] of (bins,) float sums (counts of correct)
      calib_bin_counts_per_tick: list[T] of (bins,) int counts
    """
    B, C, T = logits.shape
    device = logits.device
    # top-k predictions per tick
    topk = int(topk)
    topk_idx = logits.topk(k=topk, dim=1).indices  # (B, topk, T)
    targets_exp = targets.view(B, 1, 1).expand(B, topk, T)
    is_correct_topk = (topk_idx == targets_exp).any(dim=1)  # (B, T)

    cert = certainties[:, 1, :]  # (B, T)
    mask = cert >= float(threshold)  # (B, T)

    correct_counts = (is_correct_topk & mask).sum(dim=0).to(torch.float32).cpu().numpy()
    incorrect_counts = ((~is_correct_topk) & mask).sum(dim=0).to(torch.float32).cpu().numpy()

    # calibration, mirroring run_imagenet_analysis:
    # probabilities = softmax(logits up to t)[pred_t].mean_over_time
    probs_all = torch.softmax(logits, dim=1)  # (B, C, T)
    pred_t = logits.argmax(dim=1)  # (B, T)

    bin_edges = torch.linspace(0, 1, calib_bins + 1, device=device)
    sum_probs_per_tick: List[np.ndarray] = []
    sum_correct_per_tick: List[np.ndarray] = []
    counts_per_tick: List[np.ndarray] = []

    for t in range(T):
        # (B,) mean predicted probability of the predicted class, averaged from 0..t
        cls_idx = pred_t[:, t]  # (B,)
        p_cls_hist = probs_all[torch.arange(B, device=device), cls_idx, : t + 1].mean(dim=-1)  # (B,)
        is_correct_top1 = (cls_idx == targets)  # (B,)

        sums = torch.zeros(calib_bins, device=device, dtype=torch.float32)
        sums_correct = torch.zeros(calib_bins, device=device, dtype=torch.float32)
        counts = torch.zeros(calib_bins, device=device, dtype=torch.float32)

        for bi in range(calib_bins):
            lo = bin_edges[bi]
            hi = bin_edges[bi + 1]
            if bi == calib_bins - 1:
                m = (p_cls_hist >= lo) & (p_cls_hist <= hi)
            else:
                m = (p_cls_hist >= lo) & (p_cls_hist < hi)
            if m.any():
                counts[bi] = m.sum()
                sums[bi] = p_cls_hist[m].sum()
                sums_correct[bi] = is_correct_top1[m].to(torch.float32).sum()

        sum_probs_per_tick.append(sums.cpu().numpy())
        sum_correct_per_tick.append(sums_correct.cpu().numpy())
        counts_per_tick.append(counts.cpu().numpy())

    return correct_counts, incorrect_counts, sum_probs_per_tick, sum_correct_per_tick, counts_per_tick


def evaluate_model(
    model: torch.nn.Module,
    loaders,
    *,
    device: str,
    threshold: float,
    topk: int,
    calib_bins: int,
) -> Tuple[AdaptiveStats, CalibrationStats]:
    model.eval()
    total = 0
    correct_counts_total = None
    incorrect_counts_total = None

    sum_probs_total: List[np.ndarray] = []
    sum_correct_total: List[np.ndarray] = []
    counts_total: List[np.ndarray] = []

    with torch.inference_mode():
        for xb, yb in tqdm(loaders.valloader, desc="eval", dynamic_ncols=True):
            xb = xb.to(device)
            yb = yb.to(device)
            logits, certs, _ = model(xb)  # logits: (B,C,T), certs: (B,2,T)
            B = xb.size(0)
            total += B

            cc, ic, sp, sc, cnt = _compute_batch_stats(
                logits, certs, yb, threshold=threshold, topk=topk, calib_bins=calib_bins
            )
            if correct_counts_total is None:
                correct_counts_total = cc
                incorrect_counts_total = ic
                sum_probs_total = sp
                sum_correct_total = sc
                counts_total = cnt
            else:
                correct_counts_total += cc
                incorrect_counts_total += ic
                for t in range(len(sum_probs_total)):
                    sum_probs_total[t] += sp[t]
                    sum_correct_total[t] += sc[t]
                    counts_total[t] += cnt[t]

    assert correct_counts_total is not None and incorrect_counts_total is not None
    correct_frac = correct_counts_total / max(total, 1)
    incorrect_frac = incorrect_counts_total / max(total, 1)

    bin_centers_per_tick: List[np.ndarray] = []
    accuracies_per_tick: List[np.ndarray] = []
    for t in range(len(sum_probs_total)):
        cnt = counts_total[t]
        centers = np.where(cnt > 0, sum_probs_total[t] / np.maximum(cnt, 1e-12), np.nan)
        accs = np.where(cnt > 0, sum_correct_total[t] / np.maximum(cnt, 1e-12), np.nan)
        bin_centers_per_tick.append(centers)
        accuracies_per_tick.append(accs)

    return AdaptiveStats(correct_frac, incorrect_frac), CalibrationStats(bin_centers_per_tick, accuracies_per_tick)


def plot_adaptive(stats: AdaptiveStats, *, T: int, threshold: float, out_path: str, title: str) -> None:
    fig = plt.figure(figsize=(6.5, 4))
    ax = fig.add_subplot(111)
    xs = np.arange(T) + 1
    ax.bar(xs, stats.correct_frac_per_tick, color="forestgreen", hatch="OO", width=0.9, label="Positive", alpha=0.9)
    ax.bar(
        xs,
        stats.incorrect_frac_per_tick,
        bottom=stats.correct_frac_per_tick,
        color="crimson",
        hatch="xx",
        width=0.9,
        label="Negative",
        alpha=0.9,
    )
    ax.set_xlim(-1, T + 1)
    ax.set_xlabel("Internal tick")
    ax.set_ylabel("% of data")
    ax.set_title(f"{title} (certainty >= {threshold})")
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_calibration(stats: CalibrationStats, *, out_path: str, title: str) -> None:
    T = len(stats.bin_centers_per_tick)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    for t in range(T):
        color = cmap(t / max(T - 1, 1))
        xs = stats.bin_centers_per_tick[t]
        ys = stats.accuracies_per_tick[t]
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() == 0:
            continue
        if t == T - 1:
            ax.plot(xs[mask], ys[mask], linestyle="-", marker=".", color="#4050f7", alpha=1, label="After all ticks")
        else:
            ax.plot(xs[mask], ys[mask], linestyle="-", marker=".", color=color, alpha=0.65)
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), "k--")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("Mean predicted probabilities")
    ax.set_ylabel("Ratio of positives")
    ax.set_title(title)
    ax.legend(loc="upper left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Internal ticks")
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device(args.device)
    inferred, _ = infer_arch_from_checkpoint(args.base_checkpoint)

    loaders = get_streaming_imagenet_loaders(
        n_train=1,
        n_val=args.n_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
        shuffle_buffer_size=2000,
    )

    def prep_model(tag: str, *, load_filters: bool) -> torch.nn.Module:
        m = _build_model(args, inferred).to(device)
        # init lazies
        m(torch.randn(1, 3, args.image_size, args.image_size, device=device))
        load_checkpoint_forgiving(m, args.base_checkpoint, map_location=device, strict=False)
        if load_filters:
            _load_only_sync_filters(m, args.finetuned_checkpoint, map_location=device)
        print(f"[{tag}] ready")
        return m

    baseline = prep_model("baseline_untrained_filters", load_filters=False)
    trained = prep_model("trained_filters", load_filters=True)

    # evaluate
    print("Evaluating baseline (untrained filters)...")
    base_ad, base_cal = evaluate_model(
        baseline,
        loaders,
        device=device,
        threshold=args.certainty_threshold,
        topk=args.topk,
        calib_bins=args.calib_bins,
    )
    print("Evaluating trained filters...")
    tr_ad, tr_cal = evaluate_model(
        trained,
        loaders,
        device=device,
        threshold=args.certainty_threshold,
        topk=args.topk,
        calib_bins=args.calib_bins,
    )

    T = len(base_ad.correct_frac_per_tick)
    out_base = Path(args.output_dir) / "baseline"
    out_tr = Path(args.output_dir) / "trained"
    out_base.mkdir(parents=True, exist_ok=True)
    out_tr.mkdir(parents=True, exist_ok=True)

    plot_adaptive(
        base_ad,
        T=T,
        threshold=args.certainty_threshold,
        out_path=str(out_base / f"steps_versus_correct_{args.certainty_threshold}.png"),
        title="Baseline (untrained filters)",
    )
    plot_adaptive(
        tr_ad,
        T=T,
        threshold=args.certainty_threshold,
        out_path=str(out_tr / f"steps_versus_correct_{args.certainty_threshold}.png"),
        title="Trained filters",
    )
    plot_calibration(base_cal, out_path=str(out_base / "imagenet_calibration.png"), title="Baseline calibration")
    plot_calibration(tr_cal, out_path=str(out_tr / "imagenet_calibration.png"), title="Trained calibration")

    np.savez(
        str(Path(args.output_dir) / "metrics.npz"),
        baseline_correct=base_ad.correct_frac_per_tick,
        baseline_incorrect=base_ad.incorrect_frac_per_tick,
        trained_correct=tr_ad.correct_frac_per_tick,
        trained_incorrect=tr_ad.incorrect_frac_per_tick,
    )
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()


