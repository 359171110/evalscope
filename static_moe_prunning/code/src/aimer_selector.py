from __future__ import annotations

from typing import Dict, List

import torch

from .amp_proxy import count_routed_experts, split_gate_up_proj
from .model_structure import iter_moe_layer_bindings


def compute_aimer_removal_score(
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    tensors = (
        gate_proj_weight.detach().float(),
        up_proj_weight.detach().float(),
        down_proj_weight.detach().float(),
    )
    absolute_sum = sum(weight.abs().sum() for weight in tensors)
    squared_sum = sum(weight.square().sum() for weight in tensors)
    numel = sum(weight.numel() for weight in tensors)
    mean_abs = absolute_sum / float(numel)
    rms = torch.sqrt(squared_sum / float(numel) + eps)
    return mean_abs / (rms + eps)


def _expert_weights(experts, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        gate, up = split_gate_up_proj(experts.gate_up_proj[expert_idx])
        return gate, up, experts.down_proj[expert_idx]
    expert = experts[expert_idx]
    return expert.gate_proj.weight, expert.up_proj.weight, expert.down_proj.weight


def build_aimer_keep_table_for_model(model, eps: float = 1e-8) -> Dict[int, torch.Tensor]:
    table: Dict[int, torch.Tensor] = {}
    for binding in iter_moe_layer_bindings(model):
        removal_scores: List[torch.Tensor] = []
        for expert_idx in range(count_routed_experts(binding.experts)):
            gate, up, down = _expert_weights(binding.experts, expert_idx)
            removal_scores.append(compute_aimer_removal_score(gate, up, down, eps=eps))
        removal = torch.stack(removal_scores)
        normalized = removal / (removal.mean() + eps)
        keep = 1.0 / (normalized + eps)
        table[int(binding.layer_idx)] = (keep / (keep.mean() + eps)).cpu()
    return table