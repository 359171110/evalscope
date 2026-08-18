from __future__ import annotations

from dataclasses import dataclass

import torch

from .budgeted_micro_expert import allocate_prefix_blocks


def _top_p_residual_score(gate: torch.Tensor) -> torch.Tensor:
    sorted_gate, sorted_idx = torch.sort(gate.float(), dim=-1, descending=True)
    cumulative_before = torch.cumsum(sorted_gate, dim=-1) - sorted_gate
    residual_sorted = 1.0 - cumulative_before
    residual = torch.empty_like(residual_sorted)
    residual.scatter_(dim=-1, index=sorted_idx, src=residual_sorted)
    return residual


def build_apa_teacher_parent_score(
    gate: torch.Tensor,
    amp_selected: torch.Tensor,
    aimer_selected: torch.Tensor,
    *,
    mode: str = "combined",
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Reconstruct the frozen APA parent utility from router and dual priors."""

    if gate.ndim != 2 or amp_selected.shape != gate.shape or aimer_selected.shape != gate.shape:
        raise ValueError("gate, amp_selected, and aimer_selected must share [tokens, slots].")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    gate_f = gate.float().clamp_min(0.0)
    selected_mode = str(mode)
    if selected_mode == "gate":
        return gate_f
    top_p = _top_p_residual_score(gate_f)
    top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(eps)
    amp_score = gate_f * amp_selected.to(gate.device, torch.float32).clamp_min(0.0)
    amp_score = amp_score / amp_score.sum(dim=-1, keepdim=True).clamp_min(eps)
    aimer_score = gate_f * aimer_selected.to(gate.device, torch.float32).clamp_min(0.0)
    aimer_score = aimer_score / aimer_score.sum(dim=-1, keepdim=True).clamp_min(eps)
    dual = torch.sqrt((amp_score * aimer_score).clamp_min(0.0) + eps)
    if selected_mode == "top_p":
        parent = top_p
    elif selected_mode == "dual":
        parent = dual
    elif selected_mode == "combined":
        parent = torch.maximum(top_p, dual)
    else:
        raise ValueError(f"Unsupported APA parent score mode: {selected_mode}")
    parent.scatter_(1, gate_f.argmax(dim=-1, keepdim=True), 1.0)
    return parent


def scatter_physical_expert_blocks(
    selected_experts: torch.Tensor,
    slot_block_values: torch.Tensor,
    *,
    num_experts: int,
) -> torch.Tensor:
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must have shape [tokens, slots].")
    if slot_block_values.ndim != 3 or slot_block_values.shape[:2] != selected_experts.shape:
        raise ValueError("slot_block_values must have shape [tokens, slots, blocks].")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    selected = selected_experts.to(device=slot_block_values.device, dtype=torch.long)
    if bool(((selected < 0) | (selected >= num_experts)).any()):
        raise ValueError("selected_experts contains an out-of-range physical expert ID.")
    result = torch.zeros(
        (num_experts, slot_block_values.shape[-1]),
        device=slot_block_values.device,
        dtype=slot_block_values.dtype,
    )
    result.index_add_(
        0,
        selected.reshape(-1),
        slot_block_values.reshape(-1, slot_block_values.shape[-1]),
    )
    return result


@dataclass(frozen=True)
class DynamicRegretBatch:
    widths: torch.Tensor
    block_values: torch.Tensor
    unconditional_block_values: torch.Tensor
    block_demands: torch.Tensor
    route_counts: torch.Tensor
    parent_scores: torch.Tensor


def compute_dynamic_regret_batch(
    *,
    gate: torch.Tensor,
    selected_experts: torch.Tensor,
    amp_layer: torch.Tensor,
    aimer_layer: torch.Tensor,
    block_coverage_layer: torch.Tensor,
    total_blocks: int,
    num_experts: int,
    parent_mode: str = "combined",
    eps: float = 1.0e-8,
) -> DynamicRegretBatch:
    """Collect expected truncation regret from one frozen APA teacher batch."""

    if gate.shape != selected_experts.shape:
        raise ValueError("gate and selected_experts must have the same shape.")
    if amp_layer.shape != (num_experts,) or aimer_layer.shape != (num_experts,):
        raise ValueError("AMP/AIMER layers must have shape [num_experts].")
    if block_coverage_layer.ndim != 2 or block_coverage_layer.shape[0] != num_experts:
        raise ValueError("block_coverage_layer must have shape [num_experts, blocks].")
    selected = selected_experts.to(dtype=torch.long)
    amp_selected = amp_layer.to(gate.device, torch.float32)[selected]
    aimer_selected = aimer_layer.to(gate.device, torch.float32)[selected]
    parent = build_apa_teacher_parent_score(
        gate, amp_selected, aimer_selected, mode=parent_mode, eps=eps
    )
    marginal = block_coverage_layer.to(gate.device, torch.float32)[selected]
    allocation = allocate_prefix_blocks(
        parent,
        marginal,
        total_blocks=total_blocks,
        min_blocks_per_slot=0,
        eps=eps,
    )
    values = scatter_physical_expert_blocks(
        selected,
        allocation.selected_marginal_scores,
        num_experts=num_experts,
    )
    unconditional_values = scatter_physical_expert_blocks(
        selected,
        parent.unsqueeze(-1) * (marginal + eps),
        num_experts=num_experts,
    )
    demands = scatter_physical_expert_blocks(
        selected,
        allocation.block_mask.to(torch.float32),
        num_experts=num_experts,
    )
    route_counts = torch.zeros(num_experts, device=gate.device, dtype=torch.float32)
    route_counts.index_add_(
        0,
        selected.reshape(-1),
        torch.ones(selected.numel(), device=gate.device, dtype=torch.float32),
    )
    return DynamicRegretBatch(
        widths=allocation.widths,
        block_values=values,
        unconditional_block_values=unconditional_values,
        block_demands=demands,
        route_counts=route_counts,
        parent_scores=parent,
    )
