from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass
class _ExpertStatistics:
    covariance: torch.Tensor
    unweighted_square_sum: torch.Tensor
    max_abs: torch.Tensor
    gate_square_sum: torch.Tensor
    gate_fourth_sum: torch.Tensor
    route_count: int
    down_proj: torch.Tensor


class RoutedExpertCovarianceAccumulator:
    """Collect sufficient statistics for selected physical MoE experts."""

    def __init__(
        self,
        target_experts: Mapping[int, Sequence[int]],
        *,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> None:
        if accumulation_dtype not in (torch.float32, torch.float64):
            raise ValueError("accumulation_dtype must be float32 or float64.")
        self.target_experts = {
            int(layer_idx): tuple(sorted({int(expert_idx) for expert_idx in expert_ids}))
            for layer_idx, expert_ids in target_experts.items()
        }
        self.accumulation_dtype = accumulation_dtype
        self._statistics: dict[int, dict[int, _ExpertStatistics]] = {}

    @staticmethod
    def _is_fused(experts) -> bool:
        return (
            hasattr(experts, "gate_up_proj")
            and hasattr(experts, "down_proj")
            and not isinstance(experts, torch.nn.ModuleList)
        )

    def _weights_and_activation(self, experts, expert_idx: int, hidden_states: torch.Tensor):
        if self._is_fused(experts):
            gate_weight, up_weight = experts.gate_up_proj[expert_idx].chunk(2, dim=0)
            gate_hidden = F.linear(hidden_states, gate_weight)
            up_hidden = F.linear(hidden_states, up_weight)
            middle = experts.act_fn(gate_hidden) * up_hidden
            down_proj = experts.down_proj[expert_idx]
            return middle, down_proj
        expert = experts[expert_idx]
        gate_hidden = F.linear(hidden_states, expert.gate_proj.weight)
        up_hidden = F.linear(hidden_states, expert.up_proj.weight)
        middle = getattr(expert, "act_fn", F.silu)(gate_hidden) * up_hidden
        return middle, expert.down_proj.weight

    def _get_or_create(
        self,
        layer_idx: int,
        expert_idx: int,
        intermediate_size: int,
        down_proj: torch.Tensor,
    ) -> _ExpertStatistics:
        layer = self._statistics.setdefault(int(layer_idx), {})
        if int(expert_idx) not in layer:
            device = down_proj.device
            layer[int(expert_idx)] = _ExpertStatistics(
                covariance=torch.zeros(
                    (intermediate_size, intermediate_size),
                    dtype=self.accumulation_dtype,
                    device=device,
                ),
                unweighted_square_sum=torch.zeros(
                    intermediate_size,
                    dtype=self.accumulation_dtype,
                    device=device,
                ),
                max_abs=torch.zeros(
                    intermediate_size,
                    dtype=self.accumulation_dtype,
                    device=device,
                ),
                gate_square_sum=torch.zeros((), dtype=self.accumulation_dtype, device=device),
                gate_fourth_sum=torch.zeros((), dtype=self.accumulation_dtype, device=device),
                route_count=0,
                down_proj=down_proj.detach().float().cpu().contiguous(),
            )
        return layer[int(expert_idx)]

    @torch.no_grad()
    def initialize_layer(self, layer_idx: int, experts) -> None:
        """Initialize zero statistics for every target expert in one layer."""

        for expert_idx in self.target_experts.get(int(layer_idx), ()):
            if self._is_fused(experts):
                intermediate_size = int(experts.gate_up_proj.shape[1] // 2)
                down_proj = experts.down_proj[int(expert_idx)]
            else:
                expert = experts[int(expert_idx)]
                intermediate_size = int(expert.gate_proj.weight.shape[0])
                down_proj = expert.down_proj.weight
            self._get_or_create(int(layer_idx), int(expert_idx), intermediate_size, down_proj)

    @torch.no_grad()
    def update(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        experts,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        """Accumulate statistics for target experts routed in one MoE layer call."""

        targets = self.target_experts.get(int(layer_idx), ())
        if not targets:
            return
        if selected_experts.shape != routing_weights.shape or selected_experts.ndim != 2:
            raise ValueError("selected_experts and routing_weights must share shape [tokens, slots].")
        if hidden_states.ndim != 2 or hidden_states.shape[0] != selected_experts.shape[0]:
            raise ValueError("hidden_states must have shape [tokens, hidden_size].")

        for expert_idx in targets:
            positions = torch.nonzero(selected_experts == int(expert_idx), as_tuple=False)
            if positions.numel() == 0:
                continue
            token_indices = positions[:, 0]
            slot_indices = positions[:, 1]
            current_states = hidden_states.index_select(0, token_indices)
            middle, down_proj = self._weights_and_activation(experts, int(expert_idx), current_states)
            middle_stats = middle.detach().to(dtype=self.accumulation_dtype)
            gate = routing_weights[token_indices, slot_indices].detach().to(dtype=self.accumulation_dtype)
            weighted_middle = middle_stats * gate[:, None]
            stats = self._get_or_create(
                int(layer_idx),
                int(expert_idx),
                int(middle_stats.shape[-1]),
                down_proj,
            )
            stats.covariance.add_(weighted_middle.transpose(0, 1) @ weighted_middle)
            stats.unweighted_square_sum.add_(middle_stats.square().sum(dim=0))
            stats.max_abs.copy_(torch.maximum(stats.max_abs, middle_stats.abs().amax(dim=0)))
            stats.gate_square_sum.add_(gate.square().sum())
            stats.gate_fourth_sum.add_(gate.pow(4).sum())
            stats.route_count += int(middle_stats.shape[0])

    def to_payload(self) -> dict[int, dict[int, dict[str, object]]]:
        """Move collected statistics to CPU tensors for serialization."""

        payload: dict[int, dict[int, dict[str, object]]] = {}
        for layer_idx, layer in self._statistics.items():
            payload[layer_idx] = {}
            for expert_idx, stats in layer.items():
                payload[layer_idx][expert_idx] = {
                    "covariance": stats.covariance.detach().cpu(),
                    "unweighted_square_sum": stats.unweighted_square_sum.detach().cpu(),
                    "max_abs": stats.max_abs.detach().cpu(),
                    "gate_square_sum": stats.gate_square_sum.detach().cpu(),
                    "gate_fourth_sum": stats.gate_fourth_sum.detach().cpu(),
                    "route_count": int(stats.route_count),
                    "down_proj": stats.down_proj,
                }
        return payload