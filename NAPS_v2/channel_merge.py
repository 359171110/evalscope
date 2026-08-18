from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ChannelMergeConfig:
    target_cap: int = 32
    min_fit_rows: int = 8
    min_holdout_rows: int = 1
    min_abs_correlation: float = 0.35
    coefficient_cap: float = 2.0
    max_update_ratio: float = 0.05
    min_relative_fit_improvement: float = 1.0e-4
    holdout_relative_tolerance: float = 0.0
    epsilon: float = 1.0e-12


def _validate_inputs(
    fit_responses: torch.Tensor,
    fit_route_weights: torch.Tensor,
    holdout_responses: torch.Tensor,
    holdout_route_weights: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    channel_utility: torch.Tensor,
    zero_mask: torch.Tensor,
    config: ChannelMergeConfig,
) -> tuple[int, int]:
    if fit_responses.ndim != 2 or holdout_responses.ndim != 2:
        raise ValueError("Fit and holdout responses must be rank-two tensors")
    if fit_responses.shape[1] != holdout_responses.shape[1]:
        raise ValueError("Fit and holdout responses must have the same channel width")
    source_width = int(fit_responses.shape[1])
    if down.ndim != 2 or down.shape[1] != source_width:
        raise ValueError("Down projection and response channels are not aligned")
    for label, responses, weights in (
        ("fit", fit_responses, fit_route_weights),
        ("holdout", holdout_responses, holdout_route_weights),
    ):
        if weights.ndim != 1 or weights.shape[0] != responses.shape[0]:
            raise ValueError(f"{label.capitalize()} route weights must align with response rows")
        if not bool(torch.isfinite(responses).all()) or not bool(torch.isfinite(weights).all()):
            raise ValueError(f"{label.capitalize()} responses and route weights must be finite")
    retained = retained.to(torch.long)
    if retained.ndim != 1 or not 0 < retained.numel() < source_width:
        raise ValueError("Retained channels must be a non-empty strict subset of the source width")
    if not torch.equal(torch.unique(retained).sort().values, retained.sort().values):
        raise ValueError("Retained channels must not contain duplicates")
    if int(retained.min().item()) < 0 or int(retained.max().item()) >= source_width:
        raise ValueError("Retained channel index is outside the source width")
    if channel_utility.shape != (source_width,) or zero_mask.shape != (source_width,):
        raise ValueError("Channel utility and zero mask must align with the source width")
    if not bool(torch.isfinite(channel_utility).all()) or bool((channel_utility < 0).any()):
        raise ValueError("Channel utility must be finite and non-negative")
    if zero_mask.dtype != torch.bool:
        raise ValueError("Zero mask must be boolean")
    if config.target_cap <= 0 or config.min_fit_rows <= 0 or config.min_holdout_rows <= 0:
        raise ValueError("Target cap and row thresholds must be positive")
    if not 0.0 <= config.min_abs_correlation <= 1.0:
        raise ValueError("Minimum absolute correlation must be between zero and one")
    if config.coefficient_cap <= 0.0 or config.max_update_ratio <= 0.0:
        raise ValueError("Coefficient cap and maximum update ratio must be positive")
    if config.min_relative_fit_improvement < 0.0 or config.holdout_relative_tolerance < 0.0:
        raise ValueError("Fit improvement and holdout tolerance must be non-negative")
    return source_width, int(retained.numel())


def _weighted_relative_loss(
    full_output: torch.Tensor,
    approximate_output: torch.Tensor,
    route_weights: torch.Tensor,
    epsilon: float,
) -> float:
    residual, denominator = _weighted_loss_components(
        full_output,
        approximate_output,
        route_weights,
        epsilon,
    )
    return residual / denominator


def _weighted_loss_components(
    full_output: torch.Tensor,
    approximate_output: torch.Tensor,
    route_weights: torch.Tensor,
    epsilon: float,
) -> tuple[float, float]:
    factors = route_weights.float().square()
    residual = ((full_output.float() - approximate_output.float()).square().sum(1) * factors).sum()
    denominator = (full_output.float().square().sum(1) * factors).sum().clamp_min(epsilon)
    return float(residual.item()), float(denominator.item())


def apply_channel_merge_plan(
    down: torch.Tensor,
    retained: torch.Tensor,
    plan: dict[str, Any],
) -> torch.Tensor:
    retained = retained.to(device=down.device, dtype=torch.long)
    output = down.float().index_select(1, retained)
    if not bool(plan.get("accepted", False)):
        return output.to(down.dtype)
    targets = [int(value) for value in plan.get("target_channels", [])]
    representatives = [int(value) for value in plan.get("representative_channels", [])]
    coefficients = [float(value) for value in plan.get("coefficients", [])]
    scale = float(plan.get("trust_region_scale", 0.0))
    if not (len(targets) == len(representatives) == len(coefficients)):
        raise ValueError("Merge targets, representatives, and coefficients do not align")
    retained_positions = {int(channel): position for position, channel in enumerate(retained.tolist())}
    for target, representative, coefficient in zip(targets, representatives, coefficients):
        position = retained_positions.get(representative)
        if position is None:
            raise ValueError(f"Merge representative {representative} is not retained")
        if target in retained_positions:
            raise ValueError(f"Merge target {target} is already retained")
        output[:, position] += scale * coefficient * down[:, target].float()
    return output.to(down.dtype)


def evaluate_channel_merge_plan(
    responses: torch.Tensor,
    route_weights: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    plan: dict[str, Any],
    epsilon: float = 1.0e-12,
) -> dict[str, float]:
    if responses.ndim != 2 or down.ndim != 2 or responses.shape[1] != down.shape[1]:
        raise ValueError("Responses and down projection are not channel-aligned")
    if route_weights.ndim != 1 or route_weights.shape[0] != responses.shape[0]:
        raise ValueError("Route weights must align with response rows")
    retained = retained.to(device=down.device, dtype=torch.long)
    if responses.shape[0] == 0:
        return {
            "denominator": 0.0,
            "baseline_residual": 0.0,
            "candidate_residual": 0.0,
            "baseline_loss": 0.0,
            "candidate_loss": 0.0,
        }
    full_output = responses.float() @ down.float().transpose(0, 1)
    retained_responses = responses.float().index_select(1, retained)
    baseline_down = down.float().index_select(1, retained)
    candidate_down = apply_channel_merge_plan(down, retained, plan).float()
    baseline_residual, denominator = _weighted_loss_components(
        full_output,
        retained_responses @ baseline_down.transpose(0, 1),
        route_weights,
        epsilon,
    )
    candidate_residual, candidate_denominator = _weighted_loss_components(
        full_output,
        retained_responses @ candidate_down.transpose(0, 1),
        route_weights,
        epsilon,
    )
    if candidate_denominator != denominator:
        raise RuntimeError("Baseline and candidate loss denominators differ")
    return {
        "denominator": denominator,
        "baseline_residual": baseline_residual,
        "candidate_residual": candidate_residual,
        "baseline_loss": baseline_residual / denominator,
        "candidate_loss": candidate_residual / denominator,
    }


def fit_channel_merge_plan(
    fit_responses: torch.Tensor,
    fit_route_weights: torch.Tensor,
    holdout_responses: torch.Tensor,
    holdout_route_weights: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    channel_utility: torch.Tensor,
    zero_mask: torch.Tensor,
    config: ChannelMergeConfig | None = None,
) -> dict[str, Any]:
    config = config or ChannelMergeConfig()
    source_width, retained_width = _validate_inputs(
        fit_responses,
        fit_route_weights,
        holdout_responses,
        holdout_route_weights,
        down,
        retained,
        channel_utility,
        zero_mask,
        config,
    )
    retained = retained.to(device=down.device, dtype=torch.long)
    zero_mask = zero_mask.to(device=down.device, dtype=torch.bool)
    channel_utility = channel_utility.to(device=down.device, dtype=torch.float32)
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "method": "channel_sparse_response_merge",
        "config": asdict(config),
        "source_width": source_width,
        "retained_width": retained_width,
        "target_channels": [],
        "representative_channels": [],
        "coefficients": [],
        "fit_correlations": [],
        "trust_region_scale": 0.0,
        "update_ratio_raw": 0.0,
        "update_ratio_final": 0.0,
        "fit_baseline_loss": None,
        "fit_candidate_loss": None,
        "fit_denominator": 0.0,
        "fit_baseline_residual": 0.0,
        "fit_candidate_residual": 0.0,
        "holdout_baseline_loss": None,
        "holdout_candidate_loss": None,
        "holdout_denominator": 0.0,
        "holdout_baseline_residual": 0.0,
        "holdout_candidate_residual": 0.0,
        "relative_fit_improvement": 0.0,
        "relative_holdout_change": 0.0,
        "accepted": False,
        "fallback_reason": None,
    }
    if fit_responses.shape[0] < config.min_fit_rows:
        diagnostics["fallback_reason"] = "insufficient_fit_rows"
        return diagnostics
    if holdout_responses.shape[0] < config.min_holdout_rows:
        diagnostics["fallback_reason"] = "insufficient_holdout_rows"
        return diagnostics

    retained_mask = torch.zeros(source_width, dtype=torch.bool, device=down.device)
    retained_mask[retained] = True
    candidates = torch.where(~retained_mask & ~zero_mask & (channel_utility > 0))[0]
    if candidates.numel() == 0:
        diagnostics["fallback_reason"] = "no_active_pruned_targets"
        return diagnostics
    ordered_targets = sorted(
        candidates.tolist(),
        key=lambda channel: (-float(channel_utility[channel].item()), int(channel)),
    )[:config.target_cap]
    active_representatives = retained[~zero_mask[retained]].sort().values
    if active_representatives.numel() == 0:
        diagnostics["fallback_reason"] = "no_active_retained_representatives"
        return diagnostics
    available_mask = torch.ones(active_representatives.numel(), dtype=torch.bool, device=down.device)
    selected_targets: list[int] = []
    selected_representatives: list[int] = []
    selected_coefficients: list[float] = []
    selected_correlations: list[float] = []
    target_tensor = torch.tensor(ordered_targets, dtype=torch.long, device=down.device)
    factors = fit_route_weights.float().abs().unsqueeze(1)
    representative_matrix = fit_responses.float().index_select(1, active_representatives) * factors
    target_matrix = fit_responses.float().index_select(1, target_tensor) * factors
    representative_norm_sq = representative_matrix.square().sum(0)
    target_norm_sq = target_matrix.square().sum(0)
    dot_products = representative_matrix.transpose(0, 1) @ target_matrix
    correlations = dot_products / (
        representative_norm_sq.sqrt().unsqueeze(1)
        * target_norm_sq.sqrt().unsqueeze(0)
    ).clamp_min(config.epsilon)
    coefficients = dot_products / representative_norm_sq.unsqueeze(1).clamp_min(config.epsilon)
    for target_position, target in enumerate(ordered_targets):
        qualified = correlations[:, target_position].abs().masked_fill(~available_mask, -torch.inf)
        representative_position = int(torch.argmax(qualified).item())
        absolute_correlation = float(qualified[representative_position].item())
        if not torch.isfinite(qualified[representative_position]) or absolute_correlation < config.min_abs_correlation:
            continue
        representative = int(active_representatives[representative_position].item())
        coefficient = float(coefficients[representative_position, target_position].item())
        coefficient = max(-config.coefficient_cap, min(config.coefficient_cap, coefficient))
        correlation = float(correlations[representative_position, target_position].item())
        selected_targets.append(int(target))
        selected_representatives.append(int(representative))
        selected_coefficients.append(float(coefficient))
        selected_correlations.append(correlation)
        available_mask[representative_position] = False
        if not bool(available_mask.any()):
            break
    if not selected_targets:
        diagnostics["fallback_reason"] = "no_similarity_qualified_pairs"
        return diagnostics

    baseline_down = down.float().index_select(1, retained)
    retained_positions = {int(channel): position for position, channel in enumerate(retained.tolist())}
    delta = torch.zeros_like(baseline_down)
    for target, representative, coefficient in zip(
        selected_targets,
        selected_representatives,
        selected_coefficients,
    ):
        delta[:, retained_positions[representative]] += coefficient * down[:, target].float()
    raw_ratio = float(
        (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(baseline_down).clamp_min(config.epsilon)).item()
    )
    scale = min(1.0, config.max_update_ratio / max(raw_ratio, config.epsilon))
    candidate_down = baseline_down + scale * delta

    losses: dict[str, tuple[float, float, float]] = {}
    for split, responses, route_weights in (
        ("fit", fit_responses, fit_route_weights),
        ("holdout", holdout_responses, holdout_route_weights),
    ):
        full_output = responses.float() @ down.float().transpose(0, 1)
        retained_responses = responses.float().index_select(1, retained)
        baseline_output = retained_responses @ baseline_down.transpose(0, 1)
        candidate_output = retained_responses @ candidate_down.transpose(0, 1)
        baseline_residual, denominator = _weighted_loss_components(
            full_output, baseline_output, route_weights, config.epsilon
        )
        candidate_residual, candidate_denominator = _weighted_loss_components(
            full_output, candidate_output, route_weights, config.epsilon
        )
        if candidate_denominator != denominator:
            raise RuntimeError("Baseline and candidate loss denominators differ")
        losses[split] = (baseline_residual, candidate_residual, denominator)
    fit_baseline_residual, fit_candidate_residual, fit_denominator = losses["fit"]
    holdout_baseline_residual, holdout_candidate_residual, holdout_denominator = losses["holdout"]
    fit_baseline = fit_baseline_residual / fit_denominator
    fit_candidate = fit_candidate_residual / fit_denominator
    holdout_baseline = holdout_baseline_residual / holdout_denominator
    holdout_candidate = holdout_candidate_residual / holdout_denominator
    relative_fit_improvement = (fit_baseline - fit_candidate) / max(fit_baseline, config.epsilon)
    relative_holdout_change = holdout_candidate / max(holdout_baseline, config.epsilon) - 1.0
    accepted = (
        relative_fit_improvement >= config.min_relative_fit_improvement
        and relative_holdout_change <= config.holdout_relative_tolerance
    )
    diagnostics.update({
        "target_channels": selected_targets,
        "representative_channels": selected_representatives,
        "coefficients": selected_coefficients,
        "fit_correlations": selected_correlations,
        "trust_region_scale": scale,
        "update_ratio_raw": raw_ratio,
        "update_ratio_final": raw_ratio * scale,
        "fit_baseline_loss": fit_baseline,
        "fit_candidate_loss": fit_candidate,
        "fit_denominator": fit_denominator,
        "fit_baseline_residual": fit_baseline_residual,
        "fit_candidate_residual": fit_candidate_residual,
        "holdout_baseline_loss": holdout_baseline,
        "holdout_candidate_loss": holdout_candidate,
        "holdout_denominator": holdout_denominator,
        "holdout_baseline_residual": holdout_baseline_residual,
        "holdout_candidate_residual": holdout_candidate_residual,
        "relative_fit_improvement": relative_fit_improvement,
        "relative_holdout_change": relative_holdout_change,
        "accepted": bool(accepted),
        "fallback_reason": None if accepted else "fit_or_holdout_gate_rejected",
    })
    return diagnostics