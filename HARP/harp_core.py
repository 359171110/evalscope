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


def _tier_count_variance(n_high: int, n_mid: int, n_low: int, n_experts: int, *, allow_mid: bool) -> float:
    """Squared deviation from an even split over the enabled tiers."""

    if allow_mid:
        center = n_experts / 3.0
        return (n_high - center) ** 2 + (n_mid - center) ** 2 + (n_low - center) ** 2
    center = n_experts / 2.0
    return (n_high - center) ** 2 + (n_low - center) ** 2


def search_expert_tier_counts(
    budget: float,
    n_experts: int,
    low_width: int,
    mid_width: int,
    high_width: int,
    *,
    min_fraction: float = 0.15,
    allow_mid: bool = True,
) -> tuple[int, int, int]:
    """Enumerate (n_high, n_mid, n_low) under a hard budget.

    Primary objective is consumed extra width vs low. Ties break toward a more
    even count split. The 15% floor is relaxed to zero when it makes the budget
    infeasible. When ``allow_mid`` is false, n_mid is always zero.
    """

    if n_experts <= 0:
        raise ValueError("n_experts must be positive.")
    if allow_mid:
        if not low_width < mid_width < high_width:
            raise ValueError("HARP combo search requires K_low < K_mid < K_high.")
    elif not low_width < high_width:
        raise ValueError("HARP two-tier search requires K_low < K_high.")
    experts = int(n_experts)
    min_enabled = 3 if allow_mid else 2

    def search(min_count: int) -> tuple[int, int, int] | None:
        best: tuple[int, int, int] | None = None
        best_score = -1.0
        best_variance = float("inf")
        if allow_mid:
            low_hi = experts - 2 * min_count
            if low_hi < min_count:
                return None
            high_range = range(min_count, low_hi + 1)
        else:
            if experts - min_count < min_count:
                return None
            high_range = range(min_count, experts - min_count + 1)
        for n_high in high_range:
            if allow_mid:
                mid_range = range(min_count, experts - n_high - min_count + 1)
            else:
                mid_range = (0,)
            for n_mid in mid_range:
                n_low = experts - n_high - n_mid
                if n_low < min_count:
                    continue
                cost = n_high * high_width + n_mid * mid_width + n_low * low_width
                if cost > budget:
                    continue
                score = float(n_high * (high_width - low_width) + n_mid * (mid_width - low_width))
                variance = _tier_count_variance(n_high, n_mid, n_low, experts, allow_mid=allow_mid)
                if score > best_score or (score == best_score and variance < best_variance):
                    best_score = score
                    best_variance = variance
                    best = (n_high, n_mid, n_low)
        return best

    min_count = int(min_fraction * experts)
    if min_count < 0 or min_enabled * min_count > experts:
        min_count = 0
    chosen = search(min_count)
    if chosen is None and min_count > 0:
        chosen = search(0)
    if chosen is None:
        return 0, 0, experts
    return chosen


def assign_expert_tier_widths(
    expert_scores: torch.Tensor,
    n_high: int,
    n_mid: int,
    low_width: int,
    mid_width: int,
    high_width: int,
) -> torch.Tensor:
    """Map Expert-SP order onto a (high, mid, low) count triple."""

    if expert_scores.ndim != 1 or not bool(torch.isfinite(expert_scores).all()):
        raise ValueError("expert_scores must be a finite vector.")
    experts = int(expert_scores.numel())
    if n_high < 0 or n_mid < 0 or n_high + n_mid > experts:
        raise ValueError("Invalid high/mid expert counts.")
    widths = torch.full((experts,), int(low_width), dtype=torch.long)
    order = torch.argsort(expert_scores, descending=True, stable=True)
    widths[order[:n_high]] = int(high_width)
    widths[order[n_high:n_high + n_mid]] = int(mid_width)
    return widths


def layer_sp_weighted_targets(
    layer_scores: torch.Tensor,
    *,
    low_width: int,
    global_avg_width: float,
    high_width: int,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Per-layer mean-width targets from Layer-SP with exponent gamma."""

    if layer_scores.ndim != 1 or layer_scores.numel() == 0:
        raise ValueError("layer_scores must be a non-empty vector.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    scores = layer_scores.to(dtype=torch.float64).clamp(min=0.0)
    if float(scores.sum()) <= 0.0:
        return torch.full((int(scores.numel()),), float(global_avg_width), dtype=torch.float64)
    weights = (scores / scores.sum()).pow(float(gamma))
    scaled = weights / weights.mean()
    targets = float(low_width) + (float(global_avg_width) - float(low_width)) * scaled
    return targets.clamp(min=float(low_width), max=float(high_width))


def allocate_combo_expert_widths(
    layer_scores: torch.Tensor,
    expert_scores_by_layer: list[torch.Tensor],
    *,
    low_width: int,
    mid_width: int,
    high_width: int,
    global_avg_width: float,
    gamma: float = 2.0,
    min_fraction: float = 0.15,
    allow_mid: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Layer-SP water-fill plus direct tier search. Widths are in channels."""

    layers = int(layer_scores.numel())
    if len(expert_scores_by_layer) != layers:
        raise ValueError("expert_scores_by_layer must match layer_scores.")
    experts = int(expert_scores_by_layer[0].numel())
    targets = layer_sp_weighted_targets(
        layer_scores, low_width=low_width, global_avg_width=global_avg_width, high_width=high_width, gamma=gamma,
    )
    budgets = (targets * experts).tolist()
    order = torch.argsort(layer_scores, descending=True, stable=True).tolist()
    widths = torch.zeros((layers, experts), dtype=torch.long)
    actual = [0.0] * layers
    counts: list[list[int]] = [[0, 0, 0] for _ in range(layers)]
    remainder = 0.0
    for row in order:
        available = float(budgets[row]) + remainder
        n_high, n_mid, n_low = search_expert_tier_counts(
            available,
            experts,
            low_width,
            mid_width,
            high_width,
            min_fraction=min_fraction,
            allow_mid=allow_mid,
        )
        layer_widths = assign_expert_tier_widths(
            expert_scores_by_layer[row], n_high, n_mid, low_width, mid_width, high_width,
        )
        used = float(layer_widths.sum().item())
        widths[row] = layer_widths
        actual[row] = used
        counts[row] = [n_high, n_mid, n_low]
        remainder = available - used
    diagnostics: dict[str, object] = {
        "gamma": float(gamma),
        "min_fraction": float(min_fraction),
        "allow_mid": bool(allow_mid),
        "layer_target_widths": [float(value) for value in targets.tolist()],
        "layer_budget_init": [float(value) for value in budgets],
        "layer_budget_actual": actual,
        "expert_tier_counts_by_layer": counts,
        "leftover_final": float(remainder),
    }
    return widths, diagnostics


def allocate_layer_quantile_high_counts(
    layer_scores: torch.Tensor,
    *,
    n_experts: int,
    total_high: int,
    min_fraction: float = 0.15,
) -> torch.Tensor:
    """Assign per-layer High-expert counts from Layer-SP order only.

    Top layers receive ``h_max``, bottom layers receive ``h_min``, and at most
    one cut layer absorbs the integer remainder. The map is isotonic in Layer-SP.
    Magnitudes are never used.
    """

    if layer_scores.ndim != 1 or layer_scores.numel() == 0:
        raise ValueError("layer_scores must be a non-empty vector.")
    if not bool(torch.isfinite(layer_scores).all()):
        raise ValueError("layer_scores must be finite.")
    layers = int(layer_scores.numel())
    experts = int(n_experts)
    target = int(total_high)
    if experts <= 0:
        raise ValueError("n_experts must be positive.")
    if not 0 <= target <= layers * experts:
        raise ValueError("total_high is outside the feasible range.")
    h_min = int(min_fraction * experts)
    if h_min < 0 or 2 * h_min > experts or target < layers * h_min or target > layers * (experts - h_min):
        h_min = 0
    h_max = experts - h_min
    if not layers * h_min <= target <= layers * h_max:
        raise RuntimeError("Unable to satisfy Layer-SP quantile budget under the expert floor.")
    delta = h_max - h_min
    counts = torch.full((layers,), h_min, dtype=torch.long)
    if delta == 0:
        if target != layers * h_min:
            raise RuntimeError("Unable to satisfy Layer-SP quantile budget when h_min == h_max.")
        return counts
    need = target - layers * h_min
    n_rich = need // delta
    remainder = need % delta
    order = torch.argsort(layer_scores, descending=True, stable=True).tolist()
    for layer in order[:n_rich]:
        counts[layer] = h_max
    if remainder:
        counts[order[n_rich]] = h_min + remainder
    return counts


def allocate_quantile_layer_expert_widths(
    layer_scores: torch.Tensor,
    expert_scores_by_layer: list[torch.Tensor],
    *,
    low_width: int,
    high_width: int,
    global_avg_width: float,
    min_fraction: float = 0.15,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Nested quantile allocation: Layer-SP two-level counts, then Expert-SP prefix."""

    layers = int(layer_scores.numel())
    if len(expert_scores_by_layer) != layers:
        raise ValueError("expert_scores_by_layer must match layer_scores.")
    experts = int(expert_scores_by_layer[0].numel())
    gap = int(high_width) - int(low_width)
    if gap <= 0:
        raise ValueError("high_width must exceed low_width.")
    extra = int(round(float(global_avg_width) - float(low_width)))
    numerator = layers * experts * extra
    if numerator % gap != 0:
        raise ValueError("Global High-expert count must be an integer.")
    total_high = numerator // gap
    high_counts = allocate_layer_quantile_high_counts(
        layer_scores,
        n_experts=experts,
        total_high=total_high,
        min_fraction=min_fraction,
    )
    widths = torch.zeros((layers, experts), dtype=torch.long)
    actual = [0.0] * layers
    counts: list[list[int]] = [[0, 0, 0] for _ in range(layers)]
    for row in range(layers):
        n_high = int(high_counts[row].item())
        layer_widths = assign_expert_tier_widths(
            expert_scores_by_layer[row],
            n_high,
            0,
            low_width,
            low_width,
            high_width,
        )
        widths[row] = layer_widths
        actual[row] = float(layer_widths.sum().item())
        counts[row] = [n_high, 0, experts - n_high]
    realized_min = int(high_counts.min().item()) if layers else 0
    realized_max = int(high_counts.max().item()) if layers else 0
    diagnostics: dict[str, object] = {
        "gamma": None,
        "min_fraction": float(min_fraction),
        "allow_mid": False,
        "high_count_floor": realized_min,
        "high_count_cap": realized_max,
        "layer_high_counts": [int(value) for value in high_counts.tolist()],
        "layer_target_widths": [float(low_width) + float(count) * float(gap) / float(experts)
                                for count in high_counts.tolist()],
        "layer_budget_actual": actual,
        "expert_tier_counts_by_layer": counts,
        "leftover_final": 0.0,
        "total_high": int(total_high),
    }
    return widths, diagnostics
