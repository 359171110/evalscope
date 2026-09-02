"""Core allocation primitives for HARP structured MoE pruning."""

from __future__ import annotations

import torch


def structural_score(tensor: torch.Tensor) -> float:
    """Compute log structural participation for one flattened parameter tensor."""

    value = tensor.detach().to(dtype=torch.float32)
    l1 = value.abs().sum(dtype=torch.float32)
    l2_squared = value.square().sum(dtype=torch.float32)
    if float(l1) <= 0.0 or float(l2_squared) <= 0.0:
        return float("-inf")
    return float(torch.log((value.numel() * l2_squared / l1.square()).to(dtype=torch.float64)).item())


def detect_anchor_layer(layer_scores: torch.Tensor) -> int | None:
    """Detect a layer-0 structural outlier using the documented 2-sigma rule."""

    if layer_scores.ndim != 1 or layer_scores.numel() == 0:
        raise ValueError("layer_scores must be a non-empty vector.")
    mean = layer_scores.mean()
    std = layer_scores.std(unbiased=False)
    if float(layer_scores[0]) > float(mean + 2.0 * std):
        return 0
    return None


def allocate_layer_upgrade_units(
    layer_scores: torch.Tensor,
    *,
    total_units: int,
    max_units_per_layer: int = 2,
    anchor_layer: int | None = None,
    anchor_min_units: int = 1,
) -> torch.Tensor:
    """Allocate layer-level low/mid/high upgrade units under an exact budget.

    One unit raises the average layer width by one hardware block. The result
    contains 0, 1, or 2 units per layer and therefore maps to the three HARP
    width tiers. Higher Layer-SP receives units first; an anchor receives its
    protected floor before ordinary layers are considered.
    """

    if layer_scores.ndim != 1 or not bool(torch.isfinite(layer_scores).all()):
        raise ValueError("layer_scores must be a finite vector.")
    layers = int(layer_scores.numel())
    total = int(total_units)
    maximum = int(max_units_per_layer)
    if not 0 <= total <= layers * maximum:
        raise ValueError("total_units is outside the layer allocation range.")
    units = torch.zeros(layers, dtype=torch.long)
    if anchor_layer is not None:
        anchor = int(anchor_layer)
        if not 0 <= anchor < layers or not 0 <= int(anchor_min_units) <= maximum:
            raise ValueError("Invalid anchor layer or anchor floor.")
        protected = min(int(anchor_min_units), total)
        units[anchor] = protected
        total -= protected
    order = torch.argsort(layer_scores, descending=True, stable=True).tolist()
    while total:
        candidates = [layer for layer in order if int(units[layer]) < maximum]
        if not candidates:
            raise RuntimeError("Unable to satisfy exact Layer-SP budget.")
        for layer in candidates:
            if total == 0:
                break
            units[layer] += 1
            total -= 1
    return units


def allocate_expert_widths(
    expert_scores: torch.Tensor,
    *,
    low_blocks: int,
    target_units: int,
) -> torch.Tensor:
    """Assign three-tier expert widths by low-first, Expert-SP ordered upgrades."""

    if expert_scores.ndim != 1 or not bool(torch.isfinite(expert_scores).all()):
        raise ValueError("expert_scores must be a finite vector.")
    experts = int(expert_scores.numel())
    units = int(target_units)
    if not 0 <= units <= 2 * experts:
        raise ValueError("target_units must be between zero and two upgrades per expert.")
    widths = torch.full((experts,), int(low_blocks), dtype=torch.long)
    order = torch.argsort(expert_scores, descending=True, stable=True)
    remaining = units
    for expert in order.tolist():
        upgrade = min(2, remaining)
        widths[expert] += upgrade
        remaining -= upgrade
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError("Unable to satisfy exact expert budget.")
    return widths
