from __future__ import annotations

from typing import Dict, List

import torch

from .model_structure import get_layer_gamma_weight, iter_moe_layer_bindings


def split_gate_up_proj(gate_up_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused SwiGLU projection into gate and up branches."""

    half = gate_up_proj.shape[0] // 2
    return gate_up_proj[:half], gate_up_proj[half:]


def compute_expert_slanc_exact(
    gamma: torch.Tensor,
    expert_up: torch.Tensor,
    expert_gate: torch.Tensor,
    expert_down: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    gamma = gamma.float()
    expert_up = expert_up.float()
    expert_gate = expert_gate.float()
    expert_down = expert_down.float()
    gamma_up = gamma[:, None] * expert_up
    gamma_gate = gamma[:, None] * expert_gate
    norm_up = torch.norm(gamma_up, p="fro")
    norm_gate = torch.norm(gamma_gate, p="fro")
    down_gate = expert_gate @ expert_down
    up_down = expert_up @ expert_down
    first = torch.norm(gamma[:, None] * (norm_up * down_gate), p="fro")
    second = torch.norm(gamma[:, None] * (norm_gate * up_down), p="fro")
    return torch.sqrt(first * second + eps)


def _expert_weights(experts, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        gate, up = split_gate_up_proj(experts.gate_up_proj[expert_idx])
        return gate, up, experts.down_proj[expert_idx]
    expert = experts[expert_idx]
    return expert.gate_proj.weight, expert.up_proj.weight, expert.down_proj.weight


def count_routed_experts(experts) -> int:
    if hasattr(experts, "gate_up_proj"):
        return int(experts.gate_up_proj.shape[0])
    return len(experts)


def build_amp_table_for_model(model, eps: float = 1e-8) -> Dict[int, torch.Tensor]:
    table: Dict[int, torch.Tensor] = {}
    for binding in iter_moe_layer_bindings(model):
        gamma = get_layer_gamma_weight(binding.layer).detach()
        scores: List[torch.Tensor] = []
        for expert_idx in range(count_routed_experts(binding.experts)):
            gate, up, down = _expert_weights(binding.experts, expert_idx)
            scores.append(
                compute_expert_slanc_exact(
                    gamma,
                    up.transpose(0, 1).contiguous(),
                    gate.transpose(0, 1).contiguous(),
                    down.transpose(0, 1).contiguous(),
                    eps=eps,
                )
            )
        raw = torch.stack(scores)
        table[int(binding.layer_idx)] = (raw / (raw.mean() + eps)).cpu()
    return table