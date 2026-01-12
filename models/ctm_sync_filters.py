from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from models.ctm import ContinuousThoughtMachine
from models.sync_filters import FIRState, IIRState, FIRSyncFilter, IIRSyncFilter, MultiBandFIRSyncFilter


FilterState = Union[None, FIRState, IIRState]


class _CTMSyncFilterMixin:
    """
    Mixin holding shared sync utilities, used by the sync-filter CTM variants.
    """

    def _pairwise_product(self, activated_state: torch.Tensor, synch_type: str) -> torch.Tensor:
        if synch_type == "action":
            n_synch = self.n_synch_action
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == "out":
            n_synch = self.n_synch_out
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        else:
            raise ValueError(f"Invalid synch_type: {synch_type}")

        if self.neuron_select_type in ("first-last", "random"):
            if self.neuron_select_type == "first-last":
                if synch_type == "action":
                    selected_left = selected_right = activated_state[:, -n_synch:]
                else:
                    selected_left = selected_right = activated_state[:, :n_synch]
            else:
                selected_left = activated_state[:, neuron_indices_left]
                selected_right = activated_state[:, neuron_indices_right]

            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            i, j = torch.triu_indices(n_synch, n_synch, device=activated_state.device)
            pairwise_product = outer[:, i, j]
        elif self.neuron_select_type == "random-pairing":
            left = activated_state[:, neuron_indices_left]
            right = activated_state[:, neuron_indices_right]
            pairwise_product = left * right
        else:
            raise ValueError("Invalid neuron selection type")

        return pairwise_product


class ContinuousThoughtMachineFIR(ContinuousThoughtMachine, _CTMSyncFilterMixin):
    """
    CTM variant with learnable FIR synchronization filter for both action and output sync.
    """

    def __init__(
        self,
        *args,
        fir_k: int = 16,
        fir_init: str = "exp",
        fir_exp_alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        p_action = self.synch_representation_size_action
        p_out = self.synch_representation_size_out
        if p_action:
            self.sync_filter_action = FIRSyncFilter(p_action, fir_k, init=fir_init, exp_alpha=fir_exp_alpha)
        else:
            self.sync_filter_action = None
        self.sync_filter_out = FIRSyncFilter(p_out, fir_k, init=fir_init, exp_alpha=fir_exp_alpha)

    def forward(self, x, track: bool = False):
        B = x.size(0)
        device = x.device

        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        kv = self.compute_features(x)

        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        state_action: FilterState = None
        state_out: FilterState = None

        for stepi in range(self.iterations):
            if self.sync_filter_action is not None:
                pp_action = self._pairwise_product(activated_state, "action")
                synchronisation_action, state_action = self.sync_filter_action(pp_action, state_action)  # (B, P)
            else:
                synchronisation_action = torch.zeros(B, 0, device=device, dtype=activated_state.dtype)

            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            activated_state = self.trace_processor(state_trace)

            pp_out = self._pairwise_product(activated_state, "out")
            synchronisation_out, state_out = self.sync_filter_out(pp_out, state_out)

            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        if track:
            return (
                predictions,
                certainties,
                (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                np.array(pre_activations_tracking),
                np.array(post_activations_tracking),
                np.array(attention_tracking),
            )
        return predictions, certainties, synchronisation_out


class ContinuousThoughtMachineIIR(ContinuousThoughtMachine, _CTMSyncFilterMixin):
    """
    CTM variant with learnable IIR (leaky integrator) synchronization filter for both action and output sync.
    """

    def __init__(
        self,
        *args,
        iir_alpha_init: float = 0.9,
        iir_eps: float = 1e-4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        p_action = self.synch_representation_size_action
        p_out = self.synch_representation_size_out
        if p_action:
            self.sync_filter_action = IIRSyncFilter(p_action, alpha_init=iir_alpha_init, eps=iir_eps)
        else:
            self.sync_filter_action = None
        self.sync_filter_out = IIRSyncFilter(p_out, alpha_init=iir_alpha_init, eps=iir_eps)

    def forward(self, x, track: bool = False):
        B = x.size(0)
        device = x.device

        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        kv = self.compute_features(x)

        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        state_action: FilterState = None
        state_out: FilterState = None

        for stepi in range(self.iterations):
            if self.sync_filter_action is not None:
                pp_action = self._pairwise_product(activated_state, "action")
                synchronisation_action, state_action = self.sync_filter_action(pp_action, state_action)  # (B, P)
            else:
                synchronisation_action = torch.zeros(B, 0, device=device, dtype=activated_state.dtype)

            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            activated_state = self.trace_processor(state_trace)

            pp_out = self._pairwise_product(activated_state, "out")
            synchronisation_out, state_out = self.sync_filter_out(pp_out, state_out)

            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        if track:
            return (
                predictions,
                certainties,
                (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                np.array(pre_activations_tracking),
                np.array(post_activations_tracking),
                np.array(attention_tracking),
            )
        return predictions, certainties, synchronisation_out


class ContinuousThoughtMachineMultiBand(ContinuousThoughtMachine, _CTMSyncFilterMixin):
    """
    CTM variant with multi-band FIR synchronization filter bank.

    Note: This increases the sync representation size to P * n_bands. This means:
    - `q_proj` and `output_projector` shapes will differ from base checkpoints and must be re-initialized.
    """

    def __init__(
        self,
        *args,
        band_ks: List[int] = (8, 16, 32),
        fir_init: str = "exp",
        fir_exp_alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        p_action = self.synch_representation_size_action
        p_out = self.synch_representation_size_out
        if p_action:
            self.sync_filter_action = MultiBandFIRSyncFilter(p_action, list(band_ks), init=fir_init, exp_alpha=fir_exp_alpha)
        else:
            self.sync_filter_action = None
        self.sync_filter_out = MultiBandFIRSyncFilter(p_out, list(band_ks), init=fir_init, exp_alpha=fir_exp_alpha)

    def forward(self, x, track: bool = False):
        B = x.size(0)
        device = x.device

        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        kv = self.compute_features(x)

        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)

        state_action: FilterState = None
        state_out: FilterState = None

        for stepi in range(self.iterations):
            if self.sync_filter_action is not None:
                pp_action = self._pairwise_product(activated_state, "action")
                synchronisation_action, state_action = self.sync_filter_action(pp_action, state_action)  # (B, P*n_bands)
            else:
                synchronisation_action = torch.zeros(B, 0, device=device, dtype=activated_state.dtype)

            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            activated_state = self.trace_processor(state_trace)

            pp_out = self._pairwise_product(activated_state, "out")
            synchronisation_out, state_out = self.sync_filter_out(pp_out, state_out)  # (B, P*n_bands)

            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        if track:
            return (
                predictions,
                certainties,
                (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                np.array(pre_activations_tracking),
                np.array(post_activations_tracking),
                np.array(attention_tracking),
            )
        return predictions, certainties, synchronisation_out




