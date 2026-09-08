"""Core allocation primitives for HARP-RankAdaptive."""

from __future__ import annotations

import math
from typing import Any

import torch

from HARP.harp_core import (
    assign_expert_tier_widths,
    layer_sp_weighted_targets,
    search_expert_tier_counts,
)
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


def allocate_v2_rank_adaptive_widths(
    layer_scores: torch.Tensor,
    expert_scores_by_layer: list[torch.Tensor],
    *,
    low_width: int,
    mid_width: int,
    high_width: int,
    gamma: float = 2.0,
    min_fraction: float = 0.15,
    relative_noise: float = 0.005,
    repeats: int = 32,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Allocate widths with HARP-v2 combo search plus rank diagnostics.

    Layer-SP water-fill and Expert-SP tier search are intentionally delegated to
    the validated HARP-v2 implementation. RankAdaptive contributes diagnostics
    and interpretation, but never converts a change-point rank into n_high.
    """

    if layer_scores.ndim != 1 or layer_scores.numel() == 0:
        raise ValueError("layer_scores must be a non-empty vector.")
    if not bool(torch.isfinite(layer_scores).all()):
        raise ValueError("layer_scores must be finite.")
    layers = int(layer_scores.numel())
    if len(expert_scores_by_layer) != layers:
        raise ValueError("expert_scores_by_layer must match layer_scores.")
    experts = int(expert_scores_by_layer[0].numel())
    if any(int(scores.numel()) != experts for scores in expert_scores_by_layer):
        raise ValueError("All layers must contain the same number of experts.")
    initial_targets = layer_sp_weighted_targets(
        layer_scores,
        low_width=int(low_width),
        global_avg_width=float(mid_width),
        high_width=int(high_width),
        gamma=float(gamma),
    )
    order = torch.argsort(layer_scores, descending=True, stable=True)
    targets = _project_bounded_targets(
        initial_targets,
        target_sum=float(layers * int(mid_width)),
        lower=float(low_width),
        upper=float(high_width),
        priority=order,
    )
    budgets = (targets * experts).tolist()
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(1, layers + 1, dtype=torch.long))
    widths = torch.zeros((layers, experts), dtype=torch.long)
    actual = [0.0] * layers
    counts: list[list[int]] = [[0, 0, 0] for _ in range(layers)]
    remainder = 0.0
    for row in order.tolist():
        available = float(budgets[row]) + remainder
        n_high, n_mid, n_low = search_expert_tier_counts(
            available,
            experts,
            int(low_width),
            int(mid_width),
            int(high_width),
            min_fraction=float(min_fraction),
            allow_mid=True,
        )
        layer_widths = assign_expert_tier_widths(
            expert_scores_by_layer[row],
            n_high,
            n_mid,
            int(low_width),
            int(mid_width),
            int(high_width),
        )
        used = float(layer_widths.sum().item())
        widths[row] = layer_widths
        actual[row] = used
        counts[row] = [n_high, n_mid, n_low]
        remainder = available - used
    widths, closure = _close_discrete_budget(
        widths,
        layer_scores,
        expert_scores_by_layer,
        target_total=layers * experts * int(mid_width),
        tier_gap=int(mid_width) - int(low_width),
        high_width=int(high_width),
    )
    actual = [float(widths[row].sum().item()) for row in range(layers)]
    counts = [[
        int((widths[row] == int(high_width)).sum().item()),
        int((widths[row] == int(mid_width)).sum().item()),
        int((widths[row] == int(low_width)).sum().item()),
    ] for row in range(layers)]
    diagnostics_layers: list[dict[str, Any]] = []
    for row, expert_scores in enumerate(expert_scores_by_layer):
        head = detect_stable_expert_head(
            expert_scores,
            relative_noise=relative_noise,
            repeats=repeats,
            seed=row,
        )
        tier_counts = counts[row]
        diagnostics_layers.append({
            "row": row,
            "layer_rank": int(ranks[row].item()),
            "layer_score": float(layer_scores[row].item()),
            "layer_target_width_initial": float(initial_targets[row].item()),
            "layer_target_width": float(targets[row].item()),
            "layer_budget_init": float(budgets[row]),
            "layer_budget_actual": float(actual[row]),
            "n_high": int(tier_counts[0]),
            "n_mid": int(tier_counts[1]),
            "n_low": int(tier_counts[2]),
            "head": head,
            "head_used_as_high_count": False,
        })
    diagnostics = {
        "allocator": "v2_rank_adaptive",
        "base_allocator": "combo",
        "gamma": float(gamma),
        "min_fraction": float(min_fraction),
        "relative_noise": float(relative_noise),
        "head_stability_repeats": int(repeats),
        "layer_order_descending": [int(value) for value in order.tolist()],
        "layers": diagnostics_layers,
        "layer_target_widths_initial": [float(value) for value in initial_targets.tolist()],
        "layer_target_widths": [float(value) for value in targets.tolist()],
        "layer_budget_init": [float(value) for value in budgets],
        "layer_budget_actual": actual,
        "expert_tier_counts_by_layer": counts,
        "leftover_before_closure": float(remainder),
        "leftover_final": 0.0,
        "budget_closure": closure,
    }
    return widths, diagnostics


def _project_bounded_targets(
    targets: torch.Tensor,
    *,
    target_sum: float,
    lower: float,
    upper: float,
    priority: torch.Tensor,
) -> torch.Tensor:
    """Project clipped targets with residual budget assigned by Layer-SP rank."""

    projected = targets.to(dtype=torch.float64).clamp(min=lower, max=upper).clone()
    if priority.ndim != 1 or priority.numel() != projected.numel():
        raise ValueError("priority must be a permutation matching targets.")
    if sorted(priority.tolist()) != list(range(int(projected.numel()))):
        raise ValueError("priority must be a permutation of target indices.")
    residual = float(target_sum - projected.sum().item())
    if residual > 0.0:
        for row in priority.tolist():
            room = float(upper - projected[row])
            if room <= 1.0e-12:
                continue
            amount = min(residual, room)
            projected[row] += amount
            residual -= amount
            if residual <= 1.0e-9:
                break
    elif residual < 0.0:
        for row in reversed(priority.tolist()):
            room = float(projected[row] - lower)
            if room <= 1.0e-12:
                continue
            amount = min(-residual, room)
            projected[row] -= amount
            residual += amount
            if residual >= -1.0e-9:
                break
    if abs(residual) > 1.0e-6 or abs(float(projected.sum().item()) - target_sum) > 1.0e-6:
        raise RuntimeError("Unable to project Layer-SP targets onto the exact bounded budget.")
    return projected


def _close_discrete_budget(
    widths: torch.Tensor,
    layer_scores: torch.Tensor,
    expert_scores_by_layer: list[torch.Tensor],
    *,
    target_total: int,
    tier_gap: int,
    high_width: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Close the remaining aligned budget with rank-ordered tier upgrades."""

    result = widths.clone()
    missing = int(target_total) - int(result.sum().item())
    if missing < 0 or tier_gap <= 0 or missing % tier_gap:
        raise RuntimeError("Discrete RankAdaptive budget error is not an aligned nonnegative gap.")
    upgrades = missing // tier_gap
    candidates: list[tuple[int, int, int, int]] = []
    layer_order = torch.argsort(layer_scores, descending=True, stable=True).tolist()
    layer_rank = {layer: rank for rank, layer in enumerate(layer_order)}
    for layer, scores in enumerate(expert_scores_by_layer):
        expert_order = torch.argsort(scores, descending=True, stable=True).tolist()
        expert_rank = {expert: rank for rank, expert in enumerate(expert_order)}
        for expert in range(int(result.shape[1])):
            if int(result[layer, expert]) < int(high_width):
                candidates.append((layer_rank[layer], expert_rank[expert], layer, expert))
    candidates.sort()
    if upgrades > len(candidates):
        raise RuntimeError("Not enough expert tiers to close the global budget.")
    touched: list[list[int]] = []
    for _, _, layer, expert in candidates[:upgrades]:
        result[layer, expert] += int(tier_gap)
        touched.append([int(layer), int(expert)])
    if int(result.sum().item()) != int(target_total):
        raise RuntimeError("RankAdaptive discrete budget closure failed.")
    return result, {
        "missing_width_before": missing,
        "tier_upgrades": upgrades,
        "touched_layer_experts": touched,
    }