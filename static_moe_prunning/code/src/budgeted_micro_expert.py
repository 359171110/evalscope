from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockAllocation:
    """Prefix-constrained block allocation for routed expert slots."""

    widths: torch.Tensor
    block_mask: torch.Tensor
    selected_marginal_scores: torch.Tensor


def prefix_mask_from_widths(widths: torch.Tensor, num_blocks: int) -> torch.Tensor:
    if widths.ndim != 2:
        raise ValueError("widths must have shape [tokens, routed_slots].")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if bool(((widths < 0) | (widths > num_blocks)).any()):
        raise ValueError("widths must be in [0, num_blocks].")
    block_index = torch.arange(num_blocks, device=widths.device).view(1, 1, -1)
    return block_index < widths.to(torch.long).unsqueeze(-1)


def allocate_prefix_blocks(
    parent_scores: torch.Tensor,
    marginal_block_scores: torch.Tensor,
    total_blocks: int | torch.Tensor,
    *,
    min_blocks_per_slot: int = 1,
    eps: float = 1.0e-12,
) -> BlockAllocation:
    """Allocate a strict per-token block budget by greedy marginal utility.

    Each routed slot first receives ``min_blocks_per_slot`` prefix blocks. The
    remaining budget is allocated one block at a time. Only the next block in a
    slot's prefix is eligible, so every returned selection is hardware-friendly
    and nested by construction.
    """

    if parent_scores.ndim != 2:
        raise ValueError("parent_scores must have shape [tokens, routed_slots].")
    if marginal_block_scores.ndim != 3:
        raise ValueError(
            "marginal_block_scores must have shape [tokens, routed_slots, blocks]."
        )
    if marginal_block_scores.shape[:2] != parent_scores.shape:
        raise ValueError("parent and marginal score shapes do not agree.")
    if not bool(torch.isfinite(parent_scores).all()) or not bool(
        torch.isfinite(marginal_block_scores).all()
    ):
        raise ValueError("allocation scores must be finite.")

    tokens, routed_slots = parent_scores.shape
    num_blocks = int(marginal_block_scores.shape[-1])
    minimum = int(min_blocks_per_slot)
    if minimum < 0 or minimum > num_blocks:
        raise ValueError("min_blocks_per_slot must be in [0, num_blocks].")

    if isinstance(total_blocks, torch.Tensor):
        budgets = total_blocks.to(device=parent_scores.device, dtype=torch.long)
        if budgets.ndim == 0:
            budgets = budgets.expand(tokens)
        if budgets.shape != (tokens,):
            raise ValueError("total_blocks tensor must be scalar or shape [tokens].")
    else:
        budgets = torch.full(
            (tokens,), int(total_blocks), device=parent_scores.device, dtype=torch.long
        )

    min_budget = routed_slots * minimum
    max_budget = routed_slots * num_blocks
    if bool((budgets < min_budget).any()):
        raise ValueError("total_blocks must provide at least one block per routed slot.")
    if bool((budgets > max_budget).any()):
        raise ValueError("total_blocks cannot exceed all available prefix blocks.")

    utility = parent_scores.float().clamp_min(0.0).unsqueeze(-1) * (
        marginal_block_scores.float().clamp_min(0.0) + eps
    )
    widths = torch.full(
        (tokens, routed_slots),
        minimum,
        device=parent_scores.device,
        dtype=torch.long,
    )
    selected_scores = torch.zeros_like(utility)
    initial_mask = prefix_mask_from_widths(widths, num_blocks)
    selected_scores[initial_mask] = utility[initial_mask]

    remaining = budgets - min_budget
    max_rounds = int(remaining.max().item())
    token_ids = torch.arange(tokens, device=parent_scores.device)
    slot_ids = torch.arange(routed_slots, device=parent_scores.device).view(1, -1)
    for round_idx in range(max_rounds):
        active_tokens = remaining > round_idx
        eligible = widths < num_blocks
        next_ids = widths.clamp_max(num_blocks - 1)
        candidates = utility[token_ids.view(-1, 1), slot_ids, next_ids]
        candidates = candidates.masked_fill(~eligible, -torch.inf)
        candidates = candidates.masked_fill(~active_tokens.unsqueeze(-1), -torch.inf)
        winners = torch.argmax(candidates, dim=-1)
        active_ids = token_ids[active_tokens]
        active_winners = winners[active_tokens]
        active_blocks = widths[active_ids, active_winners]
        if bool((active_blocks >= num_blocks).any()):
            raise RuntimeError("allocator exhausted candidates before budget.")
        selected_scores[active_ids, active_winners, active_blocks] = utility[
            active_ids, active_winners, active_blocks
        ]
        widths[active_ids, active_winners] += 1

    return BlockAllocation(
        widths=widths,
        block_mask=prefix_mask_from_widths(widths, num_blocks),
        selected_marginal_scores=selected_scores,
    )


def apply_hierarchical_completion(
    partial_outputs: torch.Tensor,
    gate: torch.Tensor,
    coverage: torch.Tensor,
    *,
    local_weight: torch.Tensor | None = None,
    observed_override: torch.Tensor | None = None,
    eps: float = 1.0e-8,
    max_correction_ratio: float | None = None,
    reliability_mode: str = "none",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Complete missing expert mass with local/global observed directions."""

    if partial_outputs.ndim != 3:
        raise ValueError(
            "partial_outputs must have shape [tokens, routed_slots, hidden_dim]."
        )
    if gate.shape != partial_outputs.shape[:2] or coverage.shape != gate.shape:
        raise ValueError("gate and coverage must match partial output slot dimensions.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if max_correction_ratio is not None and max_correction_ratio < 0:
        raise ValueError("max_correction_ratio must be non-negative.")

    work_dtype = torch.float32
    partial = partial_outputs.to(work_dtype)
    gate_f = gate.to(device=partial.device, dtype=work_dtype).clamp_min(0.0)
    coverage_f = coverage.to(device=partial.device, dtype=work_dtype).clamp(0.0, 1.0)
    if local_weight is None:
        local_weight_f = coverage_f
    else:
        if local_weight.shape != gate.shape:
            raise ValueError("local_weight must match gate shape.")
        local_weight_f = local_weight.to(partial.device, work_dtype).clamp(0.0, 1.0)

    if observed_override is None:
        observed = (gate_f.unsqueeze(-1) * partial).sum(dim=1)
    else:
        if observed_override.shape != (partial.shape[0], partial.shape[2]):
            raise ValueError("observed_override must have shape [tokens, hidden_dim].")
        observed = observed_override.to(device=partial.device, dtype=work_dtype)
    observed_mass = (gate_f * coverage_f).sum(dim=1, keepdim=True)
    total_gate_mass = gate_f.sum(dim=1, keepdim=True)
    global_direction = observed / observed_mass.clamp_min(eps)

    local_direction = partial / coverage_f.clamp_min(eps).unsqueeze(-1)
    hierarchical_direction = (
        local_weight_f.unsqueeze(-1) * local_direction
        + (1.0 - local_weight_f).unsqueeze(-1) * global_direction.unsqueeze(1)
    )
    per_slot_missing = gate_f * (1.0 - coverage_f)
    missing_mass = per_slot_missing.sum(dim=1, keepdim=True)
    correction = (
        per_slot_missing.unsqueeze(-1) * hierarchical_direction
    ).sum(dim=1)

    reliability = torch.ones_like(missing_mass)
    if reliability_mode == "directional_agreement":
        direction_norm = torch.linalg.vector_norm(local_direction, dim=-1).clamp_min(eps)
        unit_direction = local_direction / direction_norm.unsqueeze(-1)
        weights = per_slot_missing
        weight_sum = weights.sum(dim=1, keepdim=True)
        resultant = torch.linalg.vector_norm(
            (weights.unsqueeze(-1) * unit_direction).sum(dim=1),
            dim=-1,
            keepdim=True,
        ) / weight_sum.clamp_min(eps)
        random_baseline = torch.sqrt((weights.square()).sum(dim=1, keepdim=True)) / (
            weight_sum.clamp_min(eps)
        )
        reliability = ((resultant - random_baseline) / (1.0 - random_baseline).clamp_min(eps)).clamp(0.0, 1.0)
        reliability = torch.where(weight_sum > eps, reliability, torch.ones_like(reliability))
        correction = correction * reliability
    elif reliability_mode != "none":
        raise ValueError(f"Unsupported reliability_mode: {reliability_mode}")

    if max_correction_ratio is not None:
        observed_norm = torch.linalg.vector_norm(observed, dim=-1, keepdim=True)
        correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
        allowed = float(max_correction_ratio) * observed_norm
        scale = torch.minimum(
            torch.ones_like(correction_norm), allowed / correction_norm.clamp_min(eps)
        )
        correction = correction * scale

    output = observed + correction
    output_dtype = (
        observed_override.dtype
        if observed_override is not None
        else partial_outputs.dtype
    )
    output = output.to(output_dtype)
    return output, {
        "observed": observed,
        "observed_mass": observed_mass,
        "total_gate_mass": total_gate_mass,
        "missing_mass": missing_mass,
        "global_direction": global_direction,
        "local_direction": local_direction,
        "local_weight": local_weight_f,
        "correction": correction,
        "completion_reliability": reliability,
    }
