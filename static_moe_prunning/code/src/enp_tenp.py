from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from .channel_runtime import LayerChannelTable


def _validate_retention(value: float, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return resolved


def _round_nearest(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


@torch.no_grad()
def signed_projection_scores(
    middle: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Accumulate TENP's signed channel projection without a [K, T, d] tensor."""

    if middle.ndim != 2:
        raise ValueError("middle must have shape [tokens, channels].")
    if down_weight.ndim != 2:
        raise ValueError("down_weight must have shape [hidden_dim, channels].")
    if int(middle.shape[1]) != int(down_weight.shape[1]):
        raise ValueError("middle and down_weight channel dimensions must match.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    middle_float = middle.float()
    down_float = down_weight.float()
    output = middle_float @ down_float.T
    projection = output @ down_float
    denominator = output.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    return (middle_float * projection / denominator).sum(dim=0)


@torch.no_grad()
def gate_norm_direction_score_sum(
    hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Sum gate-weighted output magnitude times input-output direction change."""

    if hidden_states.ndim != 2 or expert_outputs.shape != hidden_states.shape:
        raise ValueError("hidden_states and expert_outputs must have shape [tokens, hidden_dim].")
    if routing_weights.ndim != 1 or int(routing_weights.numel()) != int(hidden_states.shape[0]):
        raise ValueError("routing_weights must have shape [tokens].")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    hidden_float = hidden_states.float()
    output_float = expert_outputs.float()
    gate_float = routing_weights.float()
    output_norm = output_float.norm(dim=-1)
    cosine = (hidden_float * output_float).sum(dim=-1) / (
        hidden_float.norm(dim=-1) * output_norm + float(eps)
    )
    direction_change = 1.0 - cosine.clamp(-1.0, 1.0)
    return (gate_float * output_norm * direction_change).sum()


def build_signed_projection_channel_table(
    raw_scores_by_layer: Mapping[int, torch.Tensor],
    *,
    block_size: int,
    eps: float = 1.0e-8,
) -> dict[int, LayerChannelTable]:
    """Rank channels by signed projection and derive non-negative runtime metadata."""

    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    table: dict[int, LayerChannelTable] = {}
    for layer_idx, raw_scores in raw_scores_by_layer.items():
        if raw_scores.ndim != 2 or int(raw_scores.shape[1]) <= 0:
            raise ValueError("raw scores must have shape [experts, channels].")
        scores = raw_scores.detach().float().cpu()
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("raw signed projection scores must be finite.")
        intermediate_size = int(scores.shape[1])
        block_sizes = torch.tensor(
            [
                min(block, intermediate_size - start)
                for start in range(0, intermediate_size, block)
            ],
            dtype=torch.long,
        )
        ranked_rows = []
        relative_rows = []
        coverage_rows = []
        for expert_scores in scores:
            order = torch.argsort(expert_scores, descending=True, stable=True)
            ranked = expert_scores.index_select(0, order)
            nonnegative = ranked - ranked.min() + float(eps)
            block_scores = torch.stack(
                [
                    nonnegative[start : start + block].sum()
                    for start in range(0, intermediate_size, block)
                ]
            )
            ranked_rows.append(order)
            relative_rows.append(block_scores / block_scores.max().clamp_min(float(eps)))
            coverage_rows.append(block_scores / block_scores.sum().clamp_min(float(eps)))
        table[int(layer_idx)] = LayerChannelTable(
            ranked_indices=torch.stack(ranked_rows),
            block_relative_scores=torch.stack(relative_rows),
            block_coverage_scores=torch.stack(coverage_rows),
            block_sizes=block_sizes,
            intermediate_size=intermediate_size,
        )
    if not table:
        raise ValueError("raw_scores_by_layer must not be empty.")
    return table


def build_enp_widths(
    *,
    num_layers: int,
    num_experts: int,
    num_blocks: int,
    routed_param_retention: float,
) -> torch.Tensor:
    """Build ENP's uniform routed-expert width plan."""

    layers = int(num_layers)
    experts = int(num_experts)
    blocks = int(num_blocks)
    if layers <= 0 or experts <= 0 or blocks <= 0:
        raise ValueError("ENP dimensions must be positive.")
    retention = _validate_retention(routed_param_retention, "routed_param_retention")
    retained_blocks = min(blocks, _round_nearest(retention * blocks))
    return torch.full((layers, experts), retained_blocks, dtype=torch.long)


def trapezoid_counts(
    num_experts_by_layer: Sequence[int],
    *,
    important_expert_ratio: float,
    shallow_weight: float = 1.0,
    deep_weight: float = 2.0,
) -> list[int]:
    """Allocate TENP full experts with a deterministic linear trapezoid."""

    counts = [int(value) for value in num_experts_by_layer]
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("num_experts_by_layer must contain positive values.")
    ratio = _validate_retention(important_expert_ratio, "important_expert_ratio")
    shallow = float(shallow_weight)
    deep = float(deep_weight)
    if not math.isfinite(shallow) or not math.isfinite(deep) or shallow <= 0.0 or deep <= 0.0:
        raise ValueError("trapezoid weights must be finite and positive.")
    layers = len(counts)
    full_experts = _round_nearest(ratio * sum(counts))
    depth = [0.0] if layers == 1 else [layer_idx / (layers - 1) for layer_idx in range(layers)]
    weights = [
        counts[layer_idx] * (shallow + (deep - shallow) * depth[layer_idx])
        for layer_idx in range(layers)
    ]
    total_weight = sum(weights)
    targets = [full_experts * weight / total_weight for weight in weights]
    allocated = [min(counts[layer_idx], int(math.floor(targets[layer_idx]))) for layer_idx in range(layers)]
    remaining = full_experts - sum(allocated)
    while remaining > 0:
        candidates = [layer_idx for layer_idx in range(layers) if allocated[layer_idx] < counts[layer_idx]]
        if not candidates:
            raise RuntimeError("unable to allocate the requested TENP full experts.")
        selected = max(
            candidates,
            key=lambda layer_idx: (targets[layer_idx] - allocated[layer_idx], layer_idx),
        )
        allocated[selected] += 1
        remaining -= 1
    return allocated


def build_tenp_widths(
    expert_scores: torch.Tensor,
    *,
    num_blocks: int,
    routed_param_retention: float,
    important_expert_ratio: float,
    shallow_weight: float = 1.0,
    deep_weight: float = 2.0,
    forced_full_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build an exact-budget TENP profile at the runtime's channel-block granularity."""

    if expert_scores.ndim != 2:
        raise ValueError("expert_scores must have shape [layers, experts].")
    if not bool(torch.isfinite(expert_scores).all()):
        raise ValueError("expert_scores must be finite.")
    layers, experts = (int(size) for size in expert_scores.shape)
    blocks = int(num_blocks)
    if layers <= 0 or experts <= 0 or blocks <= 0:
        raise ValueError("TENP dimensions must be positive.")
    retention = _validate_retention(routed_param_retention, "routed_param_retention")
    important_ratio = _validate_retention(important_expert_ratio, "important_expert_ratio")
    if important_ratio > retention:
        raise ValueError("important_expert_ratio must not exceed routed_param_retention.")

    full_counts = trapezoid_counts(
        [experts] * layers,
        important_expert_ratio=important_ratio,
        shallow_weight=shallow_weight,
        deep_weight=deep_weight,
    )
    scores = expert_scores.detach().float().cpu()
    if forced_full_mask is None:
        forced_mask = torch.zeros((layers, experts), dtype=torch.bool)
    else:
        if forced_full_mask.shape != expert_scores.shape:
            raise ValueError("forced_full_mask must match expert_scores shape.")
        forced_mask = forced_full_mask.detach().bool().cpu()
    target_full_experts = sum(full_counts)
    if int(forced_mask.sum().item()) > target_full_experts:
        raise ValueError("forced full experts exceed TENP's important-expert budget.")
    forced_counts = forced_mask.sum(dim=1).tolist()
    for layer_idx, forced_count in enumerate(forced_counts):
        full_counts[layer_idx] = max(full_counts[layer_idx], int(forced_count))
    excess = sum(full_counts) - target_full_experts
    while excess > 0:
        candidates = [
            layer_idx
            for layer_idx in range(layers)
            if full_counts[layer_idx] > int(forced_counts[layer_idx])
        ]
        if not candidates:
            raise RuntimeError("unable to rebalance TENP full experts around forced experts.")
        selected = min(candidates, key=lambda layer_idx: (layer_idx, -full_counts[layer_idx]))
        full_counts[selected] -= 1
        excess -= 1

    important_mask = forced_mask.clone()
    for layer_idx, count in enumerate(full_counts):
        order = torch.argsort(scores[layer_idx], descending=True, stable=True)
        candidates = order[~forced_mask[layer_idx].index_select(0, order)]
        remaining = count - int(forced_counts[layer_idx])
        important_mask[layer_idx, candidates[:remaining]] = True

    maximum_blocks = layers * experts * blocks
    target_blocks = _round_nearest(retention * maximum_blocks)
    full_blocks = int(important_mask.sum().item()) * blocks
    if full_blocks > target_blocks:
        raise ValueError("TENP full experts exceed the routed-expert parameter budget.")
    narrow_mask = ~important_mask
    narrow_experts = int(narrow_mask.sum().item())
    remaining_blocks = target_blocks - full_blocks
    if narrow_experts == 0:
        if remaining_blocks != 0:
            raise ValueError("TENP has no narrow experts available for the remaining budget.")
        widths = torch.full((layers, experts), blocks, dtype=torch.long)
        narrow_retention = 1.0
    else:
        base_width, extra_experts = divmod(remaining_blocks, narrow_experts)
        if base_width > blocks or (base_width == blocks and extra_experts > 0):
            raise ValueError("TENP narrow-expert budget exceeds full width.")
        widths = torch.zeros((layers, experts), dtype=torch.long)
        widths[important_mask] = blocks
        widths[narrow_mask] = base_width
        if extra_experts > 0:
            flattened_scores = scores.reshape(-1)
            candidate_ids = torch.nonzero(narrow_mask.reshape(-1), as_tuple=False).flatten()
            candidate_scores = flattened_scores.index_select(0, candidate_ids)
            order = torch.argsort(candidate_scores, descending=True, stable=True)
            selected = candidate_ids.index_select(0, order[:extra_experts])
            widths.reshape(-1)[selected] += 1
        narrow_retention = remaining_blocks / (narrow_experts * blocks)

    if int(widths.sum().item()) != target_blocks:
        raise RuntimeError("TENP width construction failed to meet the exact block budget.")
    unique_widths, width_counts = torch.unique(widths[narrow_mask], return_counts=True)
    return widths, important_mask, {
        "total_blocks": target_blocks,
        "maximum_blocks": maximum_blocks,
        "full_experts_by_layer": full_counts,
        "full_expert_count": int(important_mask.sum().item()),
        "forced_full_expert_count": int(forced_mask.sum().item()),
        "narrow_expert_count": narrow_experts,
        "narrow_retention": float(narrow_retention),
        "narrow_width_histogram": {
            str(int(width)): int(count)
            for width, count in zip(unique_widths.tolist(), width_counts.tolist())
        },
        "integer_budget_rule": "equal_base_width_then_stable_expert_score_largest_remainder",
    }