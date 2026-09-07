"""Core allocation primitives for HARP-RankAdaptive."""

from __future__ import annotations

import math
from typing import Any

import torch

from HARP.rank_analysis_core import one_change_point, perturb_scores


def allocate_layer_rank_units(
    layer_scores: torch.Tensor,
    *,
    n_experts: int,
    step_fraction: float = 0.25,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Allocate an exact one-unit-per-expert global budget by layer rank."""

    if layer_scores.ndim != 1 or layer_scores.numel() == 0:
        raise ValueError("layer_scores must be a non-empty vector.")
    if not bool(torch.isfinite(layer_scores).all()):
        raise ValueError("layer_scores must be finite.")
    experts = int(n_experts)
    step = float(step_fraction)
    if experts <= 0:
        raise ValueError("n_experts must be positive.")
    if not 0.0 <= step <= 1.0:
        raise ValueError("step_fraction must be in [0, 1].")

    layers = int(layer_scores.numel())
    order = torch.argsort(layer_scores, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(1, layers + 1, dtype=torch.long))
    edge = max(1, int(math.ceil(0.25 * layers)))
    signals = torch.zeros(layers, dtype=torch.float64)
    signals[ranks <= edge] = 1.0
    signals[ranks > layers - edge] = -1.0
    centered = signals - signals.mean()
    raw = float(experts) + step * float(experts) * centered
    raw = raw.clamp(min=0.0, max=float(2 * experts))

    units = torch.floor(raw).to(dtype=torch.long)
    target = layers * experts
    remaining = target - int(units.sum().item())
    fractions = raw - units.to(dtype=torch.float64)
    if remaining > 0:
        correction_order = sorted(
            range(layers),
            key=lambda row: (-float(fractions[row]), int(ranks[row]), row),
        )
        for row in correction_order:
            if remaining == 0:
                break
            if int(units[row]) < 2 * experts:
                units[row] += 1
                remaining -= 1
    elif remaining < 0:
        correction_order = sorted(
            range(layers),
            key=lambda row: (float(fractions[row]), -int(ranks[row]), row),
        )
        for row in correction_order:
            if remaining == 0:
                break
            if int(units[row]) > 0:
                units[row] -= 1
                remaining += 1
    if remaining != 0 or int(units.sum().item()) != target:
        raise RuntimeError("Unable to close the exact Layer-SP rank budget.")

    groups = ["top" if int(rank) <= edge else "bottom" if int(rank) > layers - edge else "middle"
              for rank in ranks.tolist()]
    diagnostics = {
        "layer_ranks": [int(value) for value in ranks.tolist()],
        "layer_groups": groups,
        "layer_rank_signals": [float(value) for value in signals.tolist()],
        "layer_centered_signals": [float(value) for value in centered.tolist()],
        "layer_target_units_real": [float(value) for value in raw.tolist()],
        "layer_target_units": [int(value) for value in units.tolist()],
        "layer_group_size": edge,
        "layer_step_fraction": step,
        "total_target_units": target,
    }
    return units, diagnostics


def detect_stable_expert_head(
    expert_scores: torch.Tensor,
    *,
    relative_noise: float = 0.005,
    repeats: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Detect a non-endpoint Expert-SP head and test its rank stability."""

    if expert_scores.ndim != 1 or expert_scores.numel() == 0:
        raise ValueError("expert_scores must be a non-empty vector.")
    if not bool(torch.isfinite(expert_scores).all()):
        raise ValueError("expert_scores must be finite.")
    if relative_noise < 0.0 or repeats <= 0:
        raise ValueError("relative_noise and repeats must be positive.")

    scores = expert_scores.to(dtype=torch.float64)
    ranked = torch.sort(scores, descending=True, stable=True).values
    base = one_change_point(ranked)
    base_rank = base["rank"]
    shifts: list[int] = []
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if base_rank is not None:
        for _ in range(int(repeats)):
            noisy = perturb_scores(scores, float(relative_noise), generator)
            found = one_change_point(torch.sort(noisy, descending=True, stable=True).values)["rank"]
            if found is not None:
                shifts.append(abs(int(found) - int(base_rank)))
    within_two = float(sum(shift <= 2 for shift in shifts) / len(shifts)) if shifts else 0.0
    strong = bool(base["natural_boundary"] and within_two >= 0.75)
    return {
        "head_rank": int(base_rank) if base_rank is not None else None,
        "change_stat": float(base["stat"]),
        "gap": float(base.get("gap", 0.0)),
        "gap_q90": float(base.get("gap_q90", 0.0)),
        "endpoint_only": bool(base.get("endpoint_only", False)),
        "natural_boundary": bool(base["natural_boundary"]),
        "stability_within_two": within_two,
        "strong_head": strong,
        "relative_noise": float(relative_noise),
        "repeats": int(repeats),
    }


def allocate_expert_tiers(
    expert_scores: torch.Tensor,
    *,
    target_units: int,
    head: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Assign low/mid/high tier indices under an exact layer unit budget."""

    if expert_scores.ndim != 1 or not bool(torch.isfinite(expert_scores).all()):
        raise ValueError("expert_scores must be a finite vector.")
    experts = int(expert_scores.numel())
    units = int(target_units)
    if not 0 <= units <= 2 * experts:
        raise ValueError("target_units must be in [0, 2 * n_experts].")

    if bool(head.get("strong_head")):
        minimum_high = max(0, units - experts)
        maximum_high = units // 2
        requested_high = int(head.get("head_rank") or 0)
        n_high = min(max(requested_high, minimum_high), maximum_high)
        mode = "strong_three_tier"
    elif units <= experts:
        n_high = 0
        mode = "weak_low_mid"
    else:
        n_high = units - experts
        mode = "weak_mid_high"
    n_mid = units - 2 * n_high
    n_low = experts - n_high - n_mid
    if min(n_high, n_mid, n_low) < 0 or 2 * n_high + n_mid != units:
        raise RuntimeError("Invalid RankAdaptive tier count solution.")

    tiers = torch.zeros(experts, dtype=torch.long)
    order = torch.argsort(expert_scores, descending=True, stable=True)
    tiers[order[:n_high]] = 2
    tiers[order[n_high:n_high + n_mid]] = 1
    diagnostics = {
        "heterogeneity_mode": mode,
        "n_high": int(n_high),
        "n_mid": int(n_mid),
        "n_low": int(n_low),
        "target_units": units,
        **head,
    }
    return tiers, diagnostics


def allocate_rank_adaptive_widths(
    layer_scores: torch.Tensor,
    expert_scores_by_layer: list[torch.Tensor],
    *,
    low_width: int,
    mid_width: int,
    high_width: int,
    layer_step_fraction: float = 0.25,
    relative_noise: float = 0.005,
    repeats: int = 32,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Allocate exact RankAdaptive expert widths in channels."""

    if not int(low_width) < int(mid_width) < int(high_width):
        raise ValueError("RankAdaptive requires low_width < mid_width < high_width.")
    gap = int(mid_width) - int(low_width)
    if int(high_width) - int(mid_width) != gap:
        raise ValueError("RankAdaptive width tiers must be symmetric.")
    layers = int(layer_scores.numel())
    if len(expert_scores_by_layer) != layers or layers == 0:
        raise ValueError("expert_scores_by_layer must match layer_scores.")
    experts = int(expert_scores_by_layer[0].numel())
    if any(int(scores.numel()) != experts for scores in expert_scores_by_layer):
        raise ValueError("All layers must contain the same number of experts.")

    layer_units, layer_diag = allocate_layer_rank_units(
        layer_scores,
        n_experts=experts,
        step_fraction=layer_step_fraction,
    )
    widths = torch.zeros((layers, experts), dtype=torch.long)
    layer_details: list[dict[str, Any]] = []
    for row, scores in enumerate(expert_scores_by_layer):
        head = detect_stable_expert_head(
            scores,
            relative_noise=relative_noise,
            repeats=repeats,
            seed=row,
        )
        tiers, detail = allocate_expert_tiers(scores, target_units=int(layer_units[row]), head=head)
        widths[row] = int(low_width) + tiers * gap
        layer_details.append({
            "row": row,
            "layer_rank": layer_diag["layer_ranks"][row],
            "layer_group": layer_diag["layer_groups"][row],
            "target_units": int(layer_units[row]),
            "actual_average_width": float(widths[row].to(dtype=torch.float64).mean().item()),
            **detail,
        })

    target_total = layers * experts * int(mid_width)
    actual_total = int(widths.sum().item())
    if actual_total != target_total:
        raise RuntimeError("RankAdaptive failed to preserve the exact global width budget.")
    diagnostics = {
        **layer_diag,
        "low_width": int(low_width),
        "mid_width": int(mid_width),
        "high_width": int(high_width),
        "relative_noise": float(relative_noise),
        "head_stability_repeats": int(repeats),
        "layers": layer_details,
        "target_total_width": target_total,
        "actual_total_width": actual_total,
        "budget_error": actual_total - target_total,
    }
    return widths, diagnostics