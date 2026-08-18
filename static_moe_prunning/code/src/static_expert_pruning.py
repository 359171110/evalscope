from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict, Iterable, Mapping

import torch

from .channel_runtime import (
    ChannelTable,
    LayerChannelTable,
    channel_layer_to_device,
    compute_expert_outputs_with_channel_prefixes,
)
from .model_structure import iter_moe_layer_bindings
from .runtime_pruner import (
    compute_moe_weighted_hidden_states,
    compute_optional_shared_expert_output,
    route_qwen3_topk,
)

from .budgeted_micro_expert import (
    apply_hierarchical_completion,
    prefix_mask_from_widths,
)


@dataclass
class StaticExpertRuntimeStats:
    """Separate checkpoint structure from route-weighted executed compute."""

    profile_widths: torch.Tensor
    num_blocks: int
    routed_slots: int = 0
    routed_blocks: int = 0
    width_histogram: Dict[int, int] = field(default_factory=dict)
    layer_routed_slots: Dict[int, int] = field(default_factory=dict)
    layer_routed_blocks: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.profile_widths.ndim != 2:
            raise ValueError("profile_widths must have shape [layers, experts].")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")
        if bool(
            ((self.profile_widths < 0) | (self.profile_widths > self.num_blocks)).any()
        ):
            raise ValueError("profile_widths contains an invalid width.")

    def update(self, layer_idx: int, routed_widths: torch.Tensor) -> None:
        values = routed_widths.detach().to(torch.long).cpu()
        slots = int(values.numel())
        blocks = int(values.sum().item())
        self.routed_slots += slots
        self.routed_blocks += blocks
        layer = int(layer_idx)
        self.layer_routed_slots[layer] = self.layer_routed_slots.get(layer, 0) + slots
        self.layer_routed_blocks[layer] = self.layer_routed_blocks.get(layer, 0) + blocks
        unique, counts = torch.unique(values, return_counts=True)
        for width, count in zip(unique.tolist(), counts.tolist()):
            self.width_histogram[int(width)] = (
                self.width_histogram.get(int(width), 0) + int(count)
            )

    def structural_pruning_ratio(self) -> float:
        maximum = int(self.profile_widths.numel()) * int(self.num_blocks)
        kept = int(self.profile_widths.to(torch.long).sum().item())
        return 0.0 if maximum == 0 else 1.0 - kept / maximum

    def routed_pruning_ratio(self) -> float:
        maximum = int(self.routed_slots) * int(self.num_blocks)
        return 0.0 if maximum == 0 else 1.0 - self.routed_blocks / maximum

    def aggregate_width_histogram(self) -> Dict[int, int]:
        return dict(sorted(self.width_histogram.items()))

    def routed_pruning_by_layer(self) -> Dict[int, float]:
        return {
            layer: 1.0
            - self.layer_routed_blocks[layer]
            / (self.layer_routed_slots[layer] * self.num_blocks)
            for layer in sorted(self.layer_routed_slots)
            if self.layer_routed_slots[layer] > 0
        }


def _validate_block_tensor(block_values: torch.Tensor) -> tuple[int, int, int]:
    if block_values.ndim != 3:
        raise ValueError("block_values must have shape [layers, experts, blocks].")
    layers, experts, blocks = (int(size) for size in block_values.shape)
    if layers <= 0 or experts <= 0 or blocks <= 0:
        raise ValueError("block_values dimensions must be positive.")
    if not bool(torch.isfinite(block_values).all()):
        raise ValueError("block_values must be finite.")
    if bool((block_values < 0).any()):
        raise ValueError("block_values must be non-negative.")
    if blocks > 1 and bool((block_values[..., :-1] < block_values[..., 1:]).any()):
        raise ValueError("prefix marginal block values must be non-increasing.")
    return layers, experts, blocks


def build_protected_min_widths(
    *,
    num_layers: int,
    num_experts: int,
    num_blocks: int,
    protected_experts: Iterable[tuple[int, int, int]],
) -> torch.Tensor:
    """Build physical-expert width floors for matched-budget protection tests."""

    if num_layers <= 0 or num_experts <= 0 or num_blocks <= 0:
        raise ValueError("profile dimensions must be positive.")
    floors = torch.zeros((num_layers, num_experts), dtype=torch.long)
    seen: set[tuple[int, int]] = set()
    for layer_idx, expert_idx, width in protected_experts:
        layer = int(layer_idx)
        expert = int(expert_idx)
        floor = int(width)
        key = (layer, expert)
        if key in seen:
            raise ValueError(f"duplicate protected physical expert: {key}.")
        if not 0 <= layer < num_layers:
            raise ValueError(f"protected layer index out of range: {layer}.")
        if not 0 <= expert < num_experts:
            raise ValueError(f"protected expert index out of range: {expert}.")
        if not 0 <= floor <= num_blocks:
            raise ValueError(f"protected width out of range: {floor}.")
        floors[layer, expert] = floor
        seen.add(key)
    return floors


def allocate_static_prefix_widths(
    block_values: torch.Tensor,
    *,
    total_blocks: int,
    min_blocks_per_expert: int = 0,
    min_widths: torch.Tensor | None = None,
    max_widths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Allocate an exact global equal-cost budget under prefix constraints.

    ``block_values[l, e, j]`` is the marginal value of retaining block ``j``
    of physical expert ``e`` in layer ``l``.  Non-increasing marginals make a
    global next-block greedy allocation exact for the equal-cost problem.
    """

    layers, experts, num_blocks = _validate_block_tensor(block_values)
    minimum = int(min_blocks_per_expert)
    if not 0 <= minimum <= num_blocks:
        raise ValueError("min_blocks_per_expert must be in [0, num_blocks].")

    device = block_values.device
    if min_widths is None:
        floors = torch.full(
            (layers, experts), minimum, device=device, dtype=torch.long
        )
    else:
        if min_widths.shape != (layers, experts):
            raise ValueError("min_widths must have shape [layers, experts].")
        if min_widths.is_floating_point() and not bool(
            (min_widths == min_widths.round()).all()
        ):
            raise ValueError("min_widths must contain integer widths.")
        floors = min_widths.to(device=device, dtype=torch.long)
        floors = torch.maximum(floors, torch.full_like(floors, minimum))
        if bool(((floors < 0) | (floors > num_blocks)).any()):
            raise ValueError("min_widths must lie between 0 and num_blocks.")

    if max_widths is None:
        caps = torch.full(
            (layers, experts), num_blocks, device=device, dtype=torch.long
        )
    else:
        if max_widths.shape != (layers, experts):
            raise ValueError("max_widths must have shape [layers, experts].")
        if max_widths.is_floating_point() and not bool(
            (max_widths == max_widths.round()).all()
        ):
            raise ValueError("max_widths must contain integer widths.")
        caps = max_widths.to(device=device, dtype=torch.long)
        if bool(((caps < 0) | (caps > num_blocks)).any()):
            raise ValueError(
                "max_widths must lie between 0 and num_blocks."
            )
    if bool((floors > caps).any()):
        raise ValueError("min_widths must not exceed max_widths.")

    requested = int(total_blocks)
    minimum_total = int(floors.sum().item())
    maximum_total = int(caps.sum().item())
    if not minimum_total <= requested <= maximum_total:
        raise ValueError(
            f"total_blocks must be in [{minimum_total}, {maximum_total}], "
            f"got {requested}."
        )

    widths = floors.clone()
    remaining = requested - minimum_total
    if remaining == 0:
        return widths

    # Non-increasing marginals make one global stable sort equivalent to the
    # iterative next-block greedy algorithm.  Stable ties preserve each
    # expert's earlier block before its later equal-valued block, so counting
    # selected entries per expert yields a feasible prefix solution directly.
    block_ids = torch.arange(num_blocks, device=device).view(1, 1, num_blocks)
    eligible = (block_ids >= floors.unsqueeze(-1)) & (block_ids < caps.unsqueeze(-1))
    eligible_indices = torch.nonzero(eligible.reshape(-1), as_tuple=False).flatten()
    eligible_values = block_values.reshape(-1).index_select(0, eligible_indices)
    order = torch.argsort(eligible_values, descending=True, stable=True)
    selected = eligible_indices.index_select(0, order[:remaining])
    selected_experts = torch.div(selected, num_blocks, rounding_mode="floor")
    increments = torch.bincount(
        selected_experts, minlength=layers * experts
    ).reshape(layers, experts)
    widths += increments.to(widths.dtype)
    return widths


def allocate_static_prefix_widths_per_layer(
    block_values: torch.Tensor,
    *,
    total_blocks_by_layer: torch.Tensor,
    min_blocks_per_expert: int = 0,
    min_widths: torch.Tensor | None = None,
    max_widths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Allocate an exact independent prefix budget for every MoE layer."""

    layers, experts, _ = _validate_block_tensor(block_values)
    budgets = total_blocks_by_layer
    if not isinstance(budgets, torch.Tensor) or budgets.ndim != 1 or int(budgets.numel()) != layers:
        raise ValueError("total_blocks_by_layer must have shape [layers].")
    if budgets.is_floating_point() and not bool((budgets == budgets.round()).all()):
        raise ValueError("total_blocks_by_layer must contain integers.")
    budgets = budgets.to(dtype=torch.long, device="cpu")
    if bool((budgets < 0).any()):
        raise ValueError("total_blocks_by_layer must be non-negative.")
    for name, widths in (("min_widths", min_widths), ("max_widths", max_widths)):
        if widths is not None and widths.shape != (layers, experts):
            raise ValueError(f"{name} must have shape [layers, experts].")

    allocated = []
    for layer_idx in range(layers):
        try:
            layer_widths = allocate_static_prefix_widths(
                block_values[layer_idx : layer_idx + 1],
                total_blocks=int(budgets[layer_idx].item()),
                min_blocks_per_expert=min_blocks_per_expert,
                min_widths=None if min_widths is None else min_widths[layer_idx : layer_idx + 1],
                max_widths=None if max_widths is None else max_widths[layer_idx : layer_idx + 1],
            )
        except ValueError as error:
            raise ValueError(f"layer {layer_idx}: {error}") from error
        allocated.append(layer_widths)
    widths = torch.cat(allocated, dim=0)
    if widths.sum(dim=1).cpu().tolist() != budgets.tolist():
        raise RuntimeError("per-layer allocator failed to satisfy an exact layer budget.")
    return widths


def allocate_compute_calibrated_prefix_widths(
    block_values: torch.Tensor,
    route_counts: torch.Tensor,
    *,
    total_blocks: int,
    target_routed_pruning_ratio: float,
    min_blocks_per_expert: int = 0,
    min_widths: torch.Tensor | None = None,
    max_widths: torch.Tensor | None = None,
    search_iterations: int = 64,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Allocate an exact structural budget near a train-only routed-compute target.

    A scalar Lagrange multiplier penalizes or rewards blocks according to the
    corresponding physical expert's train route count.  Because every block of
    one expert has the same compute cost, subtracting that cost preserves the
    expert's non-increasing prefix marginals.  The structural block count remains
    exact while binary search finds the closest discrete routed-compute point.
    """

    layers, experts, num_blocks = _validate_block_tensor(block_values)
    if route_counts.shape != (layers, experts):
        raise ValueError("route_counts must have shape [layers, experts].")
    routes = route_counts.to(device=block_values.device, dtype=torch.float64)
    if not bool(torch.isfinite(routes).all()) or bool((routes < 0).any()):
        raise ValueError("route_counts must be finite and non-negative.")
    if float(routes.sum().item()) <= 0.0:
        raise ValueError("route_counts must contain positive routed mass.")
    target_pruning = float(target_routed_pruning_ratio)
    if not 0.0 <= target_pruning <= 1.0:
        raise ValueError("target_routed_pruning_ratio must be in [0, 1].")
    iterations = int(search_iterations)
    if iterations <= 0:
        raise ValueError("search_iterations must be positive.")

    values = block_values.to(dtype=torch.float64)
    value_scale = float(values.abs().max().item())
    if value_scale <= 0.0:
        value_scale = 1.0
    normalized_values = values / value_scale
    positive_routes = routes[routes > 0]
    route_scale = float(positive_routes.mean().item())
    normalized_routes = routes / route_scale
    maximum_routed_blocks = float(routes.sum().item()) * float(num_blocks)
    target_retained = (1.0 - target_pruning) * maximum_routed_blocks

    candidates: list[tuple[float, float, float, torch.Tensor]] = []

    def evaluate(multiplier: float) -> tuple[float, torch.Tensor]:
        adjusted = normalized_values - float(multiplier) * normalized_routes.unsqueeze(-1)
        minimum = float(adjusted.min().item())
        if minimum < 0.0:
            adjusted = adjusted - minimum
        widths = allocate_static_prefix_widths(
            adjusted.to(dtype=block_values.dtype),
            total_blocks=total_blocks,
            min_blocks_per_expert=min_blocks_per_expert,
            min_widths=min_widths,
            max_widths=max_widths,
        )
        retained = float((routes * widths.to(routes.dtype)).sum().item())
        prefix = torch.arange(num_blocks, device=values.device).view(1, 1, -1)
        selected = prefix < widths.to(device=values.device).unsqueeze(-1)
        objective = float(values.masked_select(selected).sum().item())
        candidates.append((abs(retained - target_retained), -objective, abs(multiplier), widths))
        return retained, widths

    bound = 1.0
    retained_low, _ = evaluate(-bound)
    retained_high, _ = evaluate(bound)
    expansions = 0
    while not (retained_low >= target_retained >= retained_high) and expansions < 40:
        bound *= 2.0
        retained_low, _ = evaluate(-bound)
        retained_high, _ = evaluate(bound)
        expansions += 1

    bracketed = retained_low >= target_retained >= retained_high
    low = -bound
    high = bound
    if bracketed:
        for _ in range(iterations):
            mid = (low + high) / 2.0
            retained, _ = evaluate(mid)
            if retained > target_retained:
                low = mid
            else:
                high = mid

    _, _, _, best_widths = min(candidates, key=lambda item: item[:3])
    achieved_retained = float((routes * best_widths.to(routes.dtype)).sum().item())
    achieved_pruning = 1.0 - achieved_retained / maximum_routed_blocks
    return best_widths, {
        "target_routed_pruning_ratio": target_pruning,
        "achieved_train_routed_pruning_ratio": achieved_pruning,
        "absolute_train_compute_error": abs(achieved_pruning - target_pruning),
        "maximum_train_routed_blocks": maximum_routed_blocks,
        "target_train_retained_blocks": target_retained,
        "achieved_train_retained_blocks": achieved_retained,
        "search_iterations": iterations,
        "bracket_expansions": expansions,
        "target_bracketed": bracketed,
    }


def allocate_fold_constrained_prefix_widths(
    block_values: torch.Tensor,
    route_count_folds: torch.Tensor,
    reference_widths: torch.Tensor,
    *,
    total_blocks: int,
    min_blocks_per_expert: int = 0,
    min_widths: torch.Tensor | None = None,
    max_widths: torch.Tensor | None = None,
    dual_iterations: int = 512,
    dual_step_size: float = 2.0,
    relative_tolerance: float = 1.0e-10,
    extra_block_cost_constraints: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Allocate exact prefix widths under per-fold compute non-inferiority.

    Every train route fold defines a separate retained-compute constraint.  The
    candidate may not retain more routed expert blocks than ``reference_widths``
    on any fold.  A projected multi-dual search repeatedly solves the exact
    equal-structure prefix problem with fold-cost penalties. Observed folds use
    a constant cost for every block of one expert. Optional extra constraints
    may provide block-specific costs when those costs are non-decreasing along
    each expert prefix, which also preserves non-increasing adjusted marginals.

    The routine returns the best feasible discrete solution by original utility.
    If the hard floors make the constraints infeasible, it returns the solution
    with the smallest maximum relative violation and records a failed feasibility
    certificate in the audit instead of silently relaxing a fold budget.
    """

    layers, experts, num_blocks = _validate_block_tensor(block_values)
    if route_count_folds.ndim != 3 or tuple(route_count_folds.shape[1:]) != (
        layers,
        experts,
    ):
        raise ValueError(
            "route_count_folds must have shape [folds, layers, experts]."
        )
    if int(route_count_folds.shape[0]) <= 0:
        raise ValueError("route_count_folds must contain at least one fold.")
    folds = route_count_folds.to(
        device=block_values.device, dtype=torch.float64
    )
    if not bool(torch.isfinite(folds).all()) or bool((folds < 0).any()):
        raise ValueError("route_count_folds must be finite and non-negative.")
    fold_totals = folds.sum(dim=(1, 2))
    if bool((fold_totals <= 0).any()):
        raise ValueError("every route-count fold must contain positive mass.")
    distributions = folds / fold_totals.view(-1, 1, 1)
    observed_fold_count = int(distributions.shape[0])

    if reference_widths.shape != (layers, experts):
        raise ValueError("reference_widths must have shape [layers, experts].")
    if reference_widths.is_floating_point() and not bool(
        (reference_widths == reference_widths.round()).all()
    ):
        raise ValueError("reference_widths must contain integer widths.")
    reference = reference_widths.to(
        device=block_values.device, dtype=torch.long
    )
    if bool(((reference < 0) | (reference > num_blocks)).any()):
        raise ValueError("reference_widths must lie between 0 and num_blocks.")
    if int(reference.sum().item()) != int(total_blocks):
        raise ValueError("reference_widths must match total_blocks exactly.")

    iterations = int(dual_iterations)
    if iterations <= 0:
        raise ValueError("dual_iterations must be positive.")
    step_size = float(dual_step_size)
    if not torch.isfinite(torch.tensor(step_size)) or step_size <= 0.0:
        raise ValueError("dual_step_size must be finite and positive.")
    tolerance = float(relative_tolerance)
    if not torch.isfinite(torch.tensor(tolerance)) or tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative.")

    block_costs = distributions.unsqueeze(-1).expand(
        -1, -1, -1, num_blocks
    ).clone()
    extra_constraint_count = 0
    if extra_block_cost_constraints is not None:
        extra = extra_block_cost_constraints.to(
            device=block_values.device, dtype=torch.float64
        )
        if extra.ndim != 4 or tuple(extra.shape[1:]) != (
            layers,
            experts,
            num_blocks,
        ):
            raise ValueError(
                "extra_block_cost_constraints must have shape "
                "[constraints, layers, experts, blocks]."
            )
        if int(extra.shape[0]) <= 0:
            raise ValueError(
                "extra_block_cost_constraints must contain at least one constraint."
            )
        if not bool(torch.isfinite(extra).all()) or bool((extra < 0).any()):
            raise ValueError(
                "extra_block_cost_constraints must be finite and non-negative."
            )
        if num_blocks > 1 and bool((extra[..., 1:] < extra[..., :-1]).any()):
            raise ValueError(
                "extra block costs must be non-decreasing within every expert prefix."
            )
        extra_constraint_count = int(extra.shape[0])
        block_costs = torch.cat((block_costs, extra), dim=0)

    values = block_values.to(dtype=torch.float64)
    value_scale = float(values.abs().max().item())
    if value_scale <= 0.0:
        value_scale = 1.0
    normalized_values = values / value_scale
    positive_costs = block_costs[block_costs > 0]
    cost_scale = float(positive_costs.mean().item())
    normalized_costs = block_costs / cost_scale
    prefix = torch.arange(num_blocks, device=values.device).view(1, 1, -1)
    reference_selected = prefix < reference.unsqueeze(-1)
    reference_costs = (
        block_costs * reference_selected.to(torch.float64).unsqueeze(0)
    ).sum(dim=(1, 2, 3))
    budget_scale = reference_costs.abs().clamp_min(
        torch.finfo(torch.float64).eps
    )
    lambdas = torch.zeros(
        int(block_costs.shape[0]),
        device=block_values.device,
        dtype=torch.float64,
    )

    candidates: list[
        tuple[float, float, float, int, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    feasible_iterations: list[int] = []

    for iteration in range(iterations + 1):
        penalty = torch.einsum("f,fleb->leb", lambdas, normalized_costs)
        adjusted = normalized_values - penalty
        minimum = float(adjusted.min().item())
        if minimum < 0.0:
            adjusted = adjusted - minimum
        widths = allocate_static_prefix_widths(
            adjusted.to(dtype=block_values.dtype),
            total_blocks=total_blocks,
            min_blocks_per_expert=min_blocks_per_expert,
            min_widths=min_widths,
            max_widths=max_widths,
        )
        selected = prefix < widths.to(device=values.device).unsqueeze(-1)
        candidate_costs = (
            block_costs * selected.to(torch.float64).unsqueeze(0)
        ).sum(dim=(1, 2, 3))
        relative_violation = (candidate_costs - reference_costs) / budget_scale
        positive_violation = relative_violation.clamp_min(0.0)
        maximum_violation = float(positive_violation.max().item())
        summed_violation = float(positive_violation.sum().item())
        objective = float(values.masked_select(selected).sum().item())
        if maximum_violation <= tolerance:
            feasible_iterations.append(iteration)
        candidates.append(
            (
                maximum_violation,
                summed_violation,
                -objective,
                iteration,
                widths.detach().clone(),
                candidate_costs.detach().clone(),
                lambdas.detach().clone(),
            )
        )
        if iteration == iterations:
            break
        # Signed projected subgradient updates allow a fold with slack to shed
        # excess penalty while violated folds receive increasing pressure.
        scheduled_step = step_size / float(iteration + 1) ** 0.5
        lambdas = (lambdas + scheduled_step * relative_violation).clamp_min(0.0)

    feasible = [item for item in candidates if item[0] <= tolerance]
    if feasible:
        best = min(feasible, key=lambda item: (item[2], item[3]))
    else:
        best = min(candidates, key=lambda item: item[:4])
    (
        maximum_violation,
        summed_violation,
        negative_objective,
        selected_iteration,
        best_widths,
        best_costs,
        best_lambdas,
    ) = best
    del negative_objective
    cost_delta = best_costs - reference_costs
    relative_delta = cost_delta / budget_scale
    return best_widths, {
        "fold_count": int(block_costs.shape[0]),
        "observed_fold_count": observed_fold_count,
        "extra_block_cost_constraint_count": extra_constraint_count,
        "all_fold_constraints_satisfied": bool(maximum_violation <= tolerance),
        "reference_retained_cost_by_fold": [
            float(value) for value in reference_costs.cpu().tolist()
        ],
        "candidate_retained_cost_by_fold": [
            float(value) for value in best_costs.cpu().tolist()
        ],
        "candidate_minus_reference_retained_cost_by_fold": [
            float(value) for value in cost_delta.cpu().tolist()
        ],
        "candidate_minus_reference_relative_cost_by_fold": [
            float(value) for value in relative_delta.cpu().tolist()
        ],
        "maximum_relative_fold_violation": float(maximum_violation),
        "summed_relative_fold_violation": float(summed_violation),
        "relative_tolerance": tolerance,
        "dual_iterations": iterations,
        "dual_step_size": step_size,
        "selected_iteration": int(selected_iteration),
        "feasible_iteration_count": len(feasible_iterations),
        "selected_dual_multipliers": [
            float(value) for value in best_lambdas.cpu().tolist()
        ],
    }


def build_reference_centered_route_envelope_costs(
    route_count_folds: torch.Tensor,
    reference_widths: torch.Tensor,
    *,
    num_blocks: int,
    expansion: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Build a separable route-envelope cost around frozen reference widths.

    Blocks already present in the reference use the lower empirical route
    envelope, while blocks beyond the reference frontier use the upper
    envelope. Candidate-minus-reference retained cost is therefore bounded by
    ``upper * positive(width-reference) + lower * negative(width-reference)``.
    An optional range expansion pads both sides without using evaluation routes.
    """

    if route_count_folds.ndim != 3:
        raise ValueError("route_count_folds must have shape [folds, layers, experts].")
    folds = route_count_folds.to(dtype=torch.float64)
    if int(folds.shape[0]) <= 0:
        raise ValueError("route_count_folds must contain at least one fold.")
    if not bool(torch.isfinite(folds).all()) or bool((folds < 0).any()):
        raise ValueError("route_count_folds must be finite and non-negative.")
    totals = folds.sum(dim=(1, 2))
    if bool((totals <= 0).any()):
        raise ValueError("every route-count fold must contain positive mass.")
    distributions = folds / totals.view(-1, 1, 1)
    layers, experts = int(folds.shape[1]), int(folds.shape[2])
    if reference_widths.shape != (layers, experts):
        raise ValueError("reference_widths must have shape [layers, experts].")
    reference = reference_widths.to(dtype=torch.long)
    blocks = int(num_blocks)
    if blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if bool(((reference < 0) | (reference > blocks)).any()):
        raise ValueError("reference_widths must lie between 0 and num_blocks.")
    padding = float(expansion)
    if not torch.isfinite(torch.tensor(padding)) or padding < 0.0:
        raise ValueError("expansion must be finite and non-negative.")

    observed_lower = distributions.min(dim=0).values
    observed_upper = distributions.max(dim=0).values
    observed_range = observed_upper - observed_lower
    lower = (observed_lower - padding * observed_range).clamp_min(0.0)
    upper = observed_upper + padding * observed_range
    block_index = torch.arange(blocks).view(1, 1, -1)
    below_reference = block_index < reference.unsqueeze(-1)
    costs = torch.where(
        below_reference,
        lower.unsqueeze(-1),
        upper.unsqueeze(-1),
    ).unsqueeze(0)
    return costs, {
        "type": "reference_centered_coordinate_envelope",
        "observed_fold_count": int(distributions.shape[0]),
        "expansion": padding,
        "observed_lower_mass": float(observed_lower.sum().item()),
        "observed_upper_mass": float(observed_upper.sum().item()),
        "expanded_lower_mass": float(lower.sum().item()),
        "expanded_upper_mass": float(upper.sum().item()),
    }


def allocate_route_envelope_constrained_prefix_widths(
    block_values: torch.Tensor,
    route_count_folds: torch.Tensor,
    reference_widths: torch.Tensor,
    *,
    total_blocks: int,
    envelope_expansion: float = 0.0,
    min_blocks_per_expert: int = 0,
    min_widths: torch.Tensor | None = None,
    max_widths: torch.Tensor | None = None,
    dual_iterations: int = 512,
    dual_step_size: float = 2.0,
    relative_tolerance: float = 1.0e-10,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Enforce observed-fold and reference-centered route-envelope budgets."""

    _, _, num_blocks = _validate_block_tensor(block_values)
    envelope_costs, envelope_audit = build_reference_centered_route_envelope_costs(
        route_count_folds,
        reference_widths,
        num_blocks=num_blocks,
        expansion=envelope_expansion,
    )
    widths, audit = allocate_fold_constrained_prefix_widths(
        block_values,
        route_count_folds,
        reference_widths,
        total_blocks=total_blocks,
        min_blocks_per_expert=min_blocks_per_expert,
        min_widths=min_widths,
        max_widths=max_widths,
        dual_iterations=dual_iterations,
        dual_step_size=dual_step_size,
        relative_tolerance=relative_tolerance,
        extra_block_cost_constraints=envelope_costs,
    )
    envelope_index = int(audit["fold_count"]) - 1
    relative_delta = audit["candidate_minus_reference_relative_cost_by_fold"]
    retained_delta = audit["candidate_minus_reference_retained_cost_by_fold"]
    audit["route_envelope"] = {
        **envelope_audit,
        "constraint_index": envelope_index,
        "constraint_retained_cost_delta": float(retained_delta[envelope_index]),
        "constraint_relative_delta": float(relative_delta[envelope_index]),
        "constraint_satisfied": bool(
            float(relative_delta[envelope_index]) <= float(relative_tolerance)
        ),
    }
    return widths, audit


def aggregate_route_count_folds(
    route_count_folds: torch.Tensor,
    *,
    aggregation: str = "mean",
    cvar_alpha: float = 0.75,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Convert train-fold route counts into one robust compute-cost distribution.

    Each fold is normalized before aggregation so longer calibration windows do
    not receive more weight.  ``mean`` is the existing consensus baseline;
    ``worst_case`` takes the coordinate-wise maximum; and ``cvar`` takes the
    upper-tail CVaR of each physical expert's fold-wise route probability.  The
    resulting vector is renormalized to sum to one because the compute anchor
    is expressed as a pruning ratio.  This creates a static, auditable
    distributionally-robust cost proxy without using validation/test routes.
    """

    if route_count_folds.ndim != 3:
        raise ValueError("route_count_folds must have shape [folds, layers, experts].")
    if route_count_folds.shape[0] <= 0:
        raise ValueError("route_count_folds must contain at least one fold.")
    counts = route_count_folds.to(dtype=torch.float64)
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
        raise ValueError("route_count_folds must be finite and non-negative.")
    fold_totals = counts.sum(dim=(1, 2))
    if bool((fold_totals <= 0).any()):
        raise ValueError("every route-count fold must contain positive routed mass.")
    distributions = counts / fold_totals.view(-1, 1, 1)
    selected = str(aggregation).lower()
    if selected == "mean":
        aggregate = distributions.mean(dim=0)
    elif selected == "worst_case":
        aggregate = distributions.max(dim=0).values
    elif selected == "cvar":
        alpha = float(cvar_alpha)
        if not 0.0 <= alpha < 1.0:
            raise ValueError("cvar_alpha must be in [0, 1).")
        tail_mass = 1.0 - alpha
        sorted_values = distributions.sort(dim=0, descending=True).values
        folds = int(distributions.shape[0])
        # Integrate the empirical upper tail exactly, including a fractional
        # final fold when (1-alpha)*folds is not an integer.
        remaining = tail_mass * folds
        aggregate = torch.zeros_like(sorted_values[0])
        for fold_idx in range(folds):
            weight = min(1.0, max(0.0, remaining - fold_idx))
            if weight == 0.0:
                break
            aggregate += weight * sorted_values[fold_idx]
        aggregate /= max(remaining, torch.finfo(aggregate.dtype).eps)
    else:
        raise ValueError("aggregation must be one of: mean, worst_case, cvar.")
    total = aggregate.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        raise ValueError("aggregated route distribution must have positive mass.")
    aggregate = aggregate / total
    return aggregate, {
        "aggregation": selected,
        "fold_count": int(distributions.shape[0]),
        "cvar_alpha": float(cvar_alpha),
        "aggregate_mass_before_normalization": float(total.item()),
    }


def build_layer_routing_entropy_prior(
    route_distribution: torch.Tensor,
    *,
    gamma: float = 0.0,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build a layer-level prior from train routing concentration.

    A positive ``gamma`` allocates relatively more capacity to high-entropy
    layers, whose routing is dispersed and therefore harder to protect with a
    sparse global expert budget.  The prior is normalized to mean one so it
    changes placement but not the scale of the utility objective.
    """

    if route_distribution.ndim != 2:
        raise ValueError("route_distribution must have shape [layers, experts].")
    if not bool(torch.isfinite(route_distribution).all()) or bool(
        (route_distribution < 0).any()
    ):
        raise ValueError("route_distribution must be finite and non-negative.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    gamma_f = float(gamma)
    layer_mass = route_distribution.sum(dim=1, keepdim=True).clamp_min(eps)
    probabilities = route_distribution / layer_mass
    entropy = -(
        probabilities.clamp_min(eps) * probabilities.clamp_min(eps).log()
    ).sum(dim=1)
    entropy = entropy / max(torch.log(torch.tensor(float(route_distribution.shape[1]))).item(), eps)
    centered = (entropy / entropy.mean().clamp_min(eps)).clamp_min(eps)
    prior = centered.pow(gamma_f)
    prior = prior / prior.mean().clamp_min(eps)
    return prior, {
        "gamma": gamma_f,
        "mean_normalized_entropy": float(entropy.mean().item()),
        "minimum_normalized_entropy": float(entropy.min().item()),
        "maximum_normalized_entropy": float(entropy.max().item()),
    }


def build_static_block_values(
    coverage_scores: torch.Tensor,
    *,
    route_counts: torch.Tensor | None = None,
    amp: torch.Tensor | None = None,
    aimer: torch.Tensor | None = None,
    mode: str = "rms",
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Build frozen physical-expert block marginals from train-only priors.

    Modes intentionally remain simple baselines.  ``rms`` uses channel-block
    coverage alone; ``route_rms`` weights it by expert usage; and
    ``dual_route_rms`` additionally applies the geometric AMP/AIMER prior.
    """

    _validate_block_tensor(coverage_scores)
    if eps <= 0:
        raise ValueError("eps must be positive.")
    values = coverage_scores.to(dtype=torch.float64)
    selected_mode = str(mode)
    layer_expert_shape = coverage_scores.shape[:2]

    if selected_mode == "rms":
        pass
    elif selected_mode in {"route_rms", "dual_route_rms"}:
        if route_counts is None or route_counts.shape != layer_expert_shape:
            raise ValueError("route_counts must have shape [layers, experts].")
        if not bool(torch.isfinite(route_counts).all()) or bool((route_counts < 0).any()):
            raise ValueError("route_counts must be finite and non-negative.")
        values = values * route_counts.to(values.device, values.dtype).unsqueeze(-1)
        if selected_mode == "dual_route_rms":
            if amp is None or amp.shape != layer_expert_shape:
                raise ValueError("amp must have shape [layers, experts].")
            if aimer is None or aimer.shape != layer_expert_shape:
                raise ValueError("aimer must have shape [layers, experts].")
            amp_f = amp.to(values.device, values.dtype)
            aimer_f = aimer.to(values.device, values.dtype)
            if not bool(torch.isfinite(amp_f).all()) or not bool(
                torch.isfinite(aimer_f).all()
            ):
                raise ValueError("amp and aimer must be finite.")
            if bool((amp_f < 0).any()) or bool((aimer_f < 0).any()):
                raise ValueError("amp and aimer must be non-negative.")
            dual_prior = (amp_f.clamp_min(eps) * aimer_f.clamp_min(eps)).sqrt()
            values = values * dual_prior.unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported static block value mode: {selected_mode}")

    # Multiplication by a non-negative expert-level scalar preserves the
    # required prefix monotonicity; validate to catch malformed source caches.
    _validate_block_tensor(values)
    return values.to(dtype=coverage_scores.dtype)


def build_static_profile(
    coverage_scores: torch.Tensor,
    *,
    mode: str,
    total_blocks: int,
    total_blocks_by_layer: torch.Tensor | None = None,
    route_counts: torch.Tensor | None = None,
    amp: torch.Tensor | None = None,
    aimer: torch.Tensor | None = None,
    min_blocks_per_expert: int = 0,
    min_widths: torch.Tensor | None = None,
    max_widths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build one frozen static-width profile at an exact structural budget."""

    layers, experts, blocks = _validate_block_tensor(coverage_scores)
    selected_mode = str(mode)
    if selected_mode == "uniform":
        # Level-wise decreasing values force the equal-cost allocator to give
        # every expert block j before assigning any expert block j+1.
        levels = torch.arange(
            blocks,
            0,
            -1,
            device=coverage_scores.device,
            dtype=coverage_scores.dtype,
        )
        values = levels.view(1, 1, blocks).expand(layers, experts, blocks)
    else:
        values = build_static_block_values(
            coverage_scores,
            route_counts=route_counts,
            amp=amp,
            aimer=aimer,
            mode=selected_mode,
        )
    if total_blocks_by_layer is not None:
        budgets = total_blocks_by_layer.to(dtype=torch.long, device="cpu")
        if int(budgets.sum().item()) != int(total_blocks):
            raise ValueError("total_blocks_by_layer must sum to total_blocks.")
        return allocate_static_prefix_widths_per_layer(
            values,
            total_blocks_by_layer=budgets,
            min_blocks_per_expert=min_blocks_per_expert,
            min_widths=min_widths,
            max_widths=max_widths,
        )
    return allocate_static_prefix_widths(
        values,
        total_blocks=total_blocks,
        min_blocks_per_expert=min_blocks_per_expert,
        min_widths=min_widths,
        max_widths=max_widths,
    )


def profile_widths_by_layer(
    profile_widths: torch.Tensor,
    *,
    layer_ids: list[int] | tuple[int, ...] | None = None,
) -> Dict[int, torch.Tensor]:
    if profile_widths.ndim != 2:
        raise ValueError("profile_widths must have shape [layers, experts].")
    if layer_ids is None:
        resolved_ids = list(range(profile_widths.shape[0]))
    else:
        resolved_ids = [int(layer_id) for layer_id in layer_ids]
        if len(resolved_ids) != profile_widths.shape[0]:
            raise ValueError("layer_ids length must match profile_widths layers.")
        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("layer_ids must be unique.")
    return {
        layer_id: profile_widths[row].detach().cpu().to(torch.long)
        for row, layer_id in enumerate(resolved_ids)
    }


def validate_static_profile_payload(payload: Mapping[str, object]) -> torch.Tensor:
    """Validate the invariants required before any test-split evaluation."""

    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported static profile schema_version.")
    construction = payload.get("profile_construction", "calibrated")
    calibration_split = payload.get("calibration_split")
    if construction == "calibration_free":
        if calibration_split != "not_applicable":
            raise ValueError(
                "calibration-free profiles must use calibration_split='not_applicable'."
            )
    elif construction == "calibrated":
        if calibration_split != "train":
            raise ValueError("static profiles must be calibrated on the train split.")
    else:
        raise ValueError("Unsupported profile_construction mode.")
    if payload.get("calibration_frozen_before_evaluation") is not True:
        raise ValueError("profile must be frozen before evaluation.")
    if payload.get("test_metrics_used_for_profile") is not False:
        raise ValueError("test metrics must not be used to construct a profile.")
    widths = payload.get("profile_widths")
    if not isinstance(widths, torch.Tensor) or widths.ndim != 2:
        raise ValueError("profile_widths must be a [layers, experts] tensor.")
    layers, experts = (int(size) for size in widths.shape)
    if layers != int(payload.get("num_layers", -1)):
        raise ValueError("num_layers does not match profile_widths.")
    if experts != int(payload.get("num_experts", -1)):
        raise ValueError("num_experts does not match profile_widths.")
    layer_ids = payload.get("layer_ids")
    if not isinstance(layer_ids, (list, tuple)) or len(layer_ids) != layers:
        raise ValueError("layer_ids must match profile_widths layers.")
    if len({int(layer_id) for layer_id in layer_ids}) != layers:
        raise ValueError("layer_ids must be unique.")
    num_blocks = int(payload.get("num_blocks", -1))
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if bool(((widths < 0) | (widths > num_blocks)).any()):
        raise ValueError("profile_widths contains an invalid width.")
    actual_total = int(widths.to(torch.long).sum().item())
    if actual_total != int(payload.get("total_blocks", -1)):
        raise ValueError("total_blocks does not match profile_widths.")
    expected_maximum = layers * experts * num_blocks
    if expected_maximum != int(payload.get("maximum_blocks", -1)):
        raise ValueError("maximum_blocks does not match profile dimensions.")
    retained_expert_mask = payload.get("retained_expert_mask")
    if retained_expert_mask is not None:
        if not isinstance(retained_expert_mask, torch.Tensor):
            raise ValueError("retained_expert_mask must be a tensor.")
        if retained_expert_mask.shape != widths.shape:
            raise ValueError("retained_expert_mask must match profile_widths.")
        retained_expert_mask = retained_expert_mask.to(dtype=torch.bool, device="cpu")
        if not bool((widths[retained_expert_mask] == num_blocks).all()):
            raise ValueError("retained experts must keep full width.")
        if not bool((widths[~retained_expert_mask] == 0).all()):
            raise ValueError("deleted experts must have zero width.")
        expected_retained = payload.get("retained_experts_by_layer")
        actual_retained = retained_expert_mask.sum(dim=1).tolist()
        if expected_retained is not None and expected_retained != actual_retained:
            raise ValueError("retained_experts_by_layer does not match retained_expert_mask.")
    return widths.detach().cpu().to(torch.long)


def gather_static_widths(
    layer_widths: torch.Tensor,
    selected_experts: torch.Tensor,
) -> torch.Tensor:
    """Gather fixed widths by physical expert ID, never by router rank."""

    if layer_widths.ndim != 1:
        raise ValueError("layer_widths must have shape [experts].")
    if selected_experts.ndim < 1:
        raise ValueError("selected_experts must have at least one dimension.")
    indices = selected_experts.to(dtype=torch.long)
    if bool(((indices < 0) | (indices >= layer_widths.numel())).any()):
        raise ValueError("selected_experts contains an out-of-range physical expert ID.")
    device_widths = layer_widths.to(device=indices.device, dtype=torch.long)
    return device_widths[indices]


def _static_expert_core(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_widths: torch.Tensor,
    channel_layer: LayerChannelTable,
    *,
    correction_mode: str = "none",
    max_correction_ratio: float | None = 0.20,
    moe_backend: str = "torch_index_add",
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
    """Execute routed experts with a frozen physical-expert width profile."""

    if selected_experts.shape != routing_weights.shape:
        raise ValueError("selected_experts and routing_weights must have the same shape.")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must have shape [tokens, routed_slots].")
    if hidden_states.ndim != 2 or hidden_states.shape[0] != selected_experts.shape[0]:
        raise ValueError("hidden_states must have shape [tokens, hidden_dim].")

    num_blocks = int(channel_layer.block_sizes.numel())
    widths = gather_static_widths(layer_widths, selected_experts)
    if bool(((widths < 0) | (widths > num_blocks)).any()):
        raise ValueError("static profile width exceeds the channel table block range.")
    block_keep_mask = prefix_mask_from_widths(widths, num_blocks)
    full_mask = widths == num_blocks

    full_hidden, _, _ = compute_moe_weighted_hidden_states(
        hidden_states,
        experts,
        selected_experts,
        routing_weights,
        keep_mask=full_mask,
        moe_backend=moe_backend,
    )
    partial_outputs = compute_expert_outputs_with_channel_prefixes(
        hidden_states,
        experts,
        selected_experts,
        block_keep_mask & ~full_mask.unsqueeze(-1),
        channel_layer,
    )
    partial_hidden = (routing_weights.unsqueeze(-1) * partial_outputs).sum(dim=1)
    observed_hidden = full_hidden + partial_hidden

    coverage_scores = channel_layer.block_coverage_scores.to(
        device=routing_weights.device, dtype=torch.float32
    )[selected_experts]
    retained_coverage = (
        coverage_scores * block_keep_mask.to(coverage_scores.dtype)
    ).sum(dim=-1).clamp(0.0, 1.0)
    retained_coverage = torch.where(
        full_mask, torch.ones_like(retained_coverage), retained_coverage
    )

    mode = str(correction_mode)
    if mode == "none":
        output = observed_hidden
        completion_aux: dict[str, torch.Tensor] = {}
    else:
        if mode == "global":
            local_weight = torch.zeros_like(retained_coverage)
            reliability_mode = "none"
        elif mode == "agreement_global":
            local_weight = torch.zeros_like(retained_coverage)
            reliability_mode = "directional_agreement"
        elif mode == "local":
            local_weight = torch.ones_like(retained_coverage)
            reliability_mode = "none"
        elif mode == "hierarchical":
            local_weight = retained_coverage
            reliability_mode = "none"
        else:
            raise ValueError(f"Unsupported correction_mode: {mode}")
        output, completion_aux = apply_hierarchical_completion(
            partial_outputs,
            routing_weights,
            retained_coverage,
            local_weight=local_weight,
            observed_override=observed_hidden,
            eps=eps,
            max_correction_ratio=max_correction_ratio,
            reliability_mode=reliability_mode,
        )

    return output, {
        **completion_aux,
        "widths": widths,
        "full_mask": full_mask,
        "block_keep_mask": block_keep_mask,
        "retained_coverage": retained_coverage,
        "correction_mode": mode,
    }


@contextmanager
def patch_qwen3_moe_blocks_static_expert(
    model,
    profile_widths: Mapping[int, torch.Tensor],
    channel_table: ChannelTable,
    *,
    retained_experts_by_layer: Mapping[int, torch.Tensor] | None = None,
    correction_mode: str = "none",
    max_correction_ratio: float | None = 0.20,
    runtime_stats: StaticExpertRuntimeStats | None = None,
    moe_backend: str = "torch_index_add",
):
    """Patch Qwen3 MoE layers to emulate a structurally shrunk checkpoint."""

    originals = []
    patched_layers = 0
    for binding in iter_moe_layer_bindings(model):
        layer_idx = int(binding.layer_idx)
        if layer_idx not in profile_widths or layer_idx not in channel_table:
            continue
        target = binding.patch_target
        original = target.forward
        widths = profile_widths[layer_idx].detach().cpu().to(torch.long)
        retained_experts = None
        if retained_experts_by_layer is not None and layer_idx in retained_experts_by_layer:
            retained_experts = retained_experts_by_layer[layer_idx].detach().cpu().to(torch.bool)
            if retained_experts.ndim != 1 or int(retained_experts.numel()) != int(widths.numel()):
                raise ValueError(f"layer {layer_idx} retained expert mask shape does not match profile widths.")
        if retained_experts is None:
            retained_experts = widths > 0
        if int(retained_experts.sum().item()) < int(binding.top_k):
            raise ValueError(f"layer {layer_idx} retained expert mask leaves fewer than top_k experts.")
        channels = channel_table[layer_idx]
        num_blocks = int(channels.block_sizes.numel())
        full_width_profile = bool((widths == num_blocks).all())
        if not full_width_profile:
            try:
                runtime_device = next(target.parameters()).device
            except StopIteration:
                runtime_device = next(model.parameters()).device
            channels = channel_layer_to_device(channels, runtime_device)

        if binding.kind == "mlp":
            top_k = binding.top_k
            norm = binding.norm_topk_prob

            if full_width_profile:

                def _forward(
                    self,
                    hidden_states,
                    _layer_idx=layer_idx,
                    _top_k=top_k,
                    _num_blocks=num_blocks,
                    _original=original,
                ):
                    if runtime_stats is not None:
                        batch, sequence = hidden_states.shape[:2]
                        routed_widths = torch.full(
                            (batch * sequence, _top_k),
                            _num_blocks,
                            device=hidden_states.device,
                            dtype=torch.long,
                        )
                        runtime_stats.update(_layer_idx, routed_widths)
                    return _original(hidden_states)

            else:

                def _forward(
                    self,
                    hidden_states,
                    _layer_idx=layer_idx,
                    _widths=widths,
                    _channels=channels,
                    _top_k=top_k,
                    _norm=norm,
                    _retained_experts=retained_experts,
                ):
                    batch, sequence, hidden_dim = hidden_states.shape
                    flat = hidden_states.reshape(-1, hidden_dim)
                    router_logits, gate, selected = route_qwen3_topk(
                        self.gate,
                        flat,
                        top_k=_top_k,
                        norm_topk_prob=_norm,
                        retained_expert_mask=_retained_experts,
                    )
                    output, aux = _static_expert_core(
                        flat,
                        self.experts,
                        selected,
                        gate,
                        _widths,
                        _channels,
                        correction_mode=correction_mode,
                        max_correction_ratio=max_correction_ratio,
                        moe_backend=moe_backend,
                    )
                    shared = compute_optional_shared_expert_output(
                        flat,
                        shared_expert=getattr(self, "shared_expert", None),
                        shared_expert_gate=getattr(self, "shared_expert_gate", None),
                    )
                    if shared is not None:
                        output = output + shared
                    if runtime_stats is not None:
                        runtime_stats.update(_layer_idx, aux["widths"])
                    return output.reshape(batch, sequence, hidden_dim), router_logits

        else:

            if full_width_profile:

                def _forward(
                    self,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    _layer_idx=layer_idx,
                    _num_blocks=num_blocks,
                    _original=original,
                ):
                    if runtime_stats is not None:
                        runtime_stats.update(
                            _layer_idx,
                            torch.full_like(top_k_index, _num_blocks),
                        )
                    return _original(hidden_states, top_k_index, top_k_weights)

            else:

                def _forward(
                    self,
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    _layer_idx=layer_idx,
                    _widths=widths,
                    _channels=channels,
                ):
                    output, aux = _static_expert_core(
                        hidden_states,
                        self,
                        top_k_index,
                        top_k_weights,
                        _widths,
                        _channels,
                        correction_mode=correction_mode,
                        max_correction_ratio=max_correction_ratio,
                        moe_backend=moe_backend,
                    )
                    if runtime_stats is not None:
                        runtime_stats.update(_layer_idx, aux["widths"])
                    return output

        originals.append((target, original))
        target.forward = MethodType(_forward, target)
        patched_layers += 1

    if patched_layers == 0:
        raise ValueError("No Qwen3 MoE layers matched the static profile and channel table.")
    try:
        yield model
    finally:
        for target, original in originals:
            target.forward = original
