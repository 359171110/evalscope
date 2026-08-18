from __future__ import annotations

import math
from typing import Sequence

import torch


def _retention(value: float, name: str) -> float:
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
    """Compute ENP neuron scores using the paper's signed projection."""

    middle_float = middle.float()
    down_float = down_weight.float()
    output = middle_float @ down_float.T
    projection = output @ down_float
    return (
        middle_float
        * projection
        / output.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    ).sum(dim=0)


@torch.no_grad()
def expert_importance_score(
    hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Compute TENP's gate, output-magnitude, and direction-change score."""

    hidden_float = hidden_states.float()
    output_float = expert_outputs.float()
    output_norm = output_float.norm(dim=-1)
    cosine = (hidden_float * output_float).sum(dim=-1) / (
        hidden_float.norm(dim=-1) * output_norm + float(eps)
    )
    return (
        routing_weights.float()
        * output_norm
        * (1.0 - cosine.clamp(-1.0, 1.0))
    ).sum()


def enp_width(original_width: int, routed_param_retention: float) -> int:
    """Return ENP's common retained neuron count for every routed expert."""

    width = int(original_width)
    if width <= 0:
        raise ValueError("original_width must be positive.")
    return min(width, _round_nearest(_retention(routed_param_retention, "routed_param_retention") * width))


def trapezoid_counts(
    num_experts_by_layer: Sequence[int],
    *,
    important_expert_ratio: float,
    shallow_weight: float = 1.0,
    deep_weight: float = 2.0,
) -> list[int]:
    """Allocate full TENP experts from shallow to deep layers."""

    counts = [int(value) for value in num_experts_by_layer]
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("num_experts_by_layer must contain positive values.")
    ratio = _retention(important_expert_ratio, "important_expert_ratio")
    layers = len(counts)
    depth = [0.0] if layers == 1 else [layer_idx / (layers - 1) for layer_idx in range(layers)]
    weights = [
        counts[layer_idx]
        * (float(shallow_weight) + (float(deep_weight) - float(shallow_weight)) * depth[layer_idx])
        for layer_idx in range(layers)
    ]
    full_experts = _round_nearest(ratio * sum(counts))
    targets = [full_experts * weight / sum(weights) for weight in weights]
    allocated = [min(counts[layer_idx], int(math.floor(targets[layer_idx]))) for layer_idx in range(layers)]
    while sum(allocated) < full_experts:
        candidates = [layer_idx for layer_idx in range(layers) if allocated[layer_idx] < counts[layer_idx]]
        selected = max(candidates, key=lambda layer_idx: (targets[layer_idx] - allocated[layer_idx], layer_idx))
        allocated[selected] += 1
    return allocated


def tenp_narrow_retention(
    routed_param_retention: float,
    important_expert_ratio: float,
) -> float:
    """Return the paper's narrow-expert neuron retention for equal-size experts."""

    retention = _retention(routed_param_retention, "routed_param_retention")
    important = _retention(important_expert_ratio, "important_expert_ratio")
    if important > retention:
        raise ValueError("important_expert_ratio must not exceed routed_param_retention.")
    if important == 1.0:
        return 1.0
    return (retention - important) / (1.0 - important)