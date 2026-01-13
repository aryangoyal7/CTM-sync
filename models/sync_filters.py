import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FIRState:
    # Shape: (B, P, Kmax)
    history: torch.Tensor


@dataclass
class IIRState:
    # Shape: (B, P)
    value: torch.Tensor


class FIRSyncFilter(nn.Module):
    """
    Learnable FIR filter over the recent history of pairwise products.

    - State: rolling window of length K of pairwise_product values.
    - Output: weighted sum of the window, weights constrained by softmax.
    """

    def __init__(
        self,
        n_pairs: int,
        k: int,
        *,
        init: str = "exp",
        exp_alpha: float = 0.5,
    ):
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.n_pairs = int(n_pairs)
        self.k = int(k)

        raw = torch.zeros(self.n_pairs, self.k)
        if init == "exp":
            # Decreasing weights: exp(-alpha * t), then log to land roughly in softmax space.
            t = torch.arange(self.k, dtype=torch.float32)
            w = torch.exp(-float(exp_alpha) * t)
            w = w / (w.sum() + 1e-12)
            raw = (w + 1e-12).log().unsqueeze(0).repeat(self.n_pairs, 1)
        elif init == "zeros":
            raw = torch.zeros(self.n_pairs, self.k)
        elif init == "randn":
            raw = torch.randn(self.n_pairs, self.k) / math.sqrt(self.k)
        else:
            raise ValueError(f"Unknown init={init}")

        self.raw_w = nn.Parameter(raw)

    def forward(
        self, pairwise_product: torch.Tensor, state: Optional[FIRState]
    ) -> Tuple[torch.Tensor, FIRState]:
        # pairwise_product: (B, P)
        if pairwise_product.ndim != 2:
            raise ValueError(f"pairwise_product must be (B,P), got {pairwise_product.shape}")
        B, P = pairwise_product.shape
        if P != self.n_pairs:
            raise ValueError(f"Expected P={self.n_pairs}, got {P}")

        if state is None:
            history = torch.zeros(B, P, self.k, device=pairwise_product.device, dtype=pairwise_product.dtype)
        else:
            history = state.history
            if history.shape != (B, P, self.k):
                # Batch size can vary across calls; reset safely.
                history = torch.zeros(B, P, self.k, device=pairwise_product.device, dtype=pairwise_product.dtype)

        history = torch.roll(history, shifts=1, dims=-1)
        history[..., 0] = pairwise_product

        w = F.softmax(self.raw_w, dim=-1)  # (P, K)
        y = (history * w.unsqueeze(0)).sum(dim=-1)  # (B, P)
        return y, FIRState(history=history)


class MultiBandFIRSyncFilter(nn.Module):
    """
    Multi-band FIR filter bank.

    Combine modes:
    - concat (default): returns concatenated band outputs (B, P * n_bands)
    - sum: returns per-pair weighted sum across bands (B, P) with learnable band mixing per pair
    """

    def __init__(
        self,
        n_pairs: int,
        band_ks: List[int],
        *,
        combine: str = "concat",
        band_mix_per_pair: bool = True,
        init: str = "exp",
        exp_alpha: float = 0.5,
    ):
        super().__init__()
        if len(band_ks) == 0:
            raise ValueError("band_ks must be non-empty")
        if any(k <= 0 for k in band_ks):
            raise ValueError(f"All band ks must be > 0, got {band_ks}")

        self.n_pairs = int(n_pairs)
        self.band_ks = [int(k) for k in band_ks]
        self.k_max = max(self.band_ks)
        self.combine = str(combine)
        if self.combine not in ("concat", "sum"):
            raise ValueError(f"Unknown combine={combine}. Expected one of: concat, sum")

        self.n_bands = len(self.band_ks)
        self.band_mix_per_pair = bool(band_mix_per_pair)
        if self.combine == "sum":
            # Learnable mixing weights across bands. Softmaxed along band dim.
            mix_shape = (self.n_pairs, self.n_bands) if self.band_mix_per_pair else (self.n_bands,)
            self.raw_band_mix = nn.Parameter(torch.zeros(mix_shape))

        raws = []
        for k in self.band_ks:
            raw = torch.zeros(self.n_pairs, k)
            if init == "exp":
                t = torch.arange(k, dtype=torch.float32)
                w = torch.exp(-float(exp_alpha) * t)
                w = w / (w.sum() + 1e-12)
                raw = (w + 1e-12).log().unsqueeze(0).repeat(self.n_pairs, 1)
            elif init == "zeros":
                raw = torch.zeros(self.n_pairs, k)
            elif init == "randn":
                raw = torch.randn(self.n_pairs, k) / math.sqrt(k)
            else:
                raise ValueError(f"Unknown init={init}")
            raws.append(nn.Parameter(raw))

        self.raw_ws = nn.ParameterList(raws)

    def forward(
        self, pairwise_product: torch.Tensor, state: Optional[FIRState]
    ) -> Tuple[torch.Tensor, FIRState]:
        if pairwise_product.ndim != 2:
            raise ValueError(f"pairwise_product must be (B,P), got {pairwise_product.shape}")
        B, P = pairwise_product.shape
        if P != self.n_pairs:
            raise ValueError(f"Expected P={self.n_pairs}, got {P}")

        if state is None:
            history = torch.zeros(B, P, self.k_max, device=pairwise_product.device, dtype=pairwise_product.dtype)
        else:
            history = state.history
            if history.shape != (B, P, self.k_max):
                history = torch.zeros(B, P, self.k_max, device=pairwise_product.device, dtype=pairwise_product.dtype)

        history = torch.roll(history, shifts=1, dims=-1)
        history[..., 0] = pairwise_product

        outs = []
        for raw_w in self.raw_ws:
            k = raw_w.shape[-1]
            w = F.softmax(raw_w, dim=-1)  # (P, k)
            yb = (history[..., :k] * w.unsqueeze(0)).sum(dim=-1)  # (B, P)
            outs.append(yb)

        if self.combine == "concat":
            y = torch.cat(outs, dim=-1)  # (B, P * n_bands)
        else:
            # (n_bands, B, P) -> (B, P, n_bands)
            stacked = torch.stack(outs, dim=0).permute(1, 2, 0)
            mix = F.softmax(self.raw_band_mix, dim=-1)
            if mix.ndim == 1:
                # (n_bands,) -> (1, 1, n_bands)
                mix = mix.view(1, 1, -1)
            else:
                # (P, n_bands) -> (1, P, n_bands)
                mix = mix.unsqueeze(0)
            y = (stacked * mix).sum(dim=-1)  # (B, P)
        return y, FIRState(history=history)


class IIRSyncFilter(nn.Module):
    """
    Learnable IIR (leaky integrator) filter.

    s_t = alpha * s_{t-1} + (1 - alpha) * x_t

    alpha constrained to (eps, 1-eps) via sigmoid.
    """

    def __init__(self, n_pairs: int, *, alpha_init: float = 0.9, eps: float = 1e-4):
        super().__init__()
        if not (0.0 < alpha_init < 1.0):
            raise ValueError("alpha_init must be in (0,1)")
        self.n_pairs = int(n_pairs)
        self.eps = float(eps)

        # Inverse-sigmoid for initialization.
        alpha_init_t = torch.full((self.n_pairs,), float(alpha_init))
        raw = torch.log(alpha_init_t) - torch.log1p(-alpha_init_t)
        self.raw_alpha = nn.Parameter(raw)

    def forward(
        self, pairwise_product: torch.Tensor, state: Optional[IIRState]
    ) -> Tuple[torch.Tensor, IIRState]:
        if pairwise_product.ndim != 2:
            raise ValueError(f"pairwise_product must be (B,P), got {pairwise_product.shape}")
        B, P = pairwise_product.shape
        if P != self.n_pairs:
            raise ValueError(f"Expected P={self.n_pairs}, got {P}")

        alpha = torch.sigmoid(self.raw_alpha)  # (P,)
        alpha = alpha * (1.0 - 2.0 * self.eps) + self.eps  # avoid exact 0/1

        if state is None or state.value.shape != (B, P):
            s = pairwise_product
        else:
            s = alpha.unsqueeze(0) * state.value + (1.0 - alpha).unsqueeze(0) * pairwise_product

        return s, IIRState(value=s)




