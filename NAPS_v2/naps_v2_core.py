from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NapsV2Config:
    effective_zero_threshold: float = 1.0e-12
    aimer_epsilon: float = 1.0e-8
    swap_fractions: tuple[tuple[int, float], ...] = (
        (2, 0.03),
        (4, 0.04),
        (8, 0.05),
        (16, 0.06),
        (32, 0.07),
    )
    coverage_top_fraction: float = 0.80
    compensation_channel_cap_b9: int = 32
    compensation_channel_cap_b6: int = 64
    ridge_scale: float = 1.0e-3
    representatives_per_target: int = 2
    minimum_uncovered_mass: float = 0.05
    trust_region_by_probe_count: tuple[tuple[int, float], ...] = (
        (2, 0.02),
        (8, 0.03),
        (16, 0.04),
    )


def effective_zero_mask(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if gate.ndim != 2 or up.shape != gate.shape or down.ndim != 2 or down.shape[1] != gate.shape[0]:
        raise ValueError("gate, up and down tensors are not channel-aligned")
    values = torch.stack(
        (gate.float().abs().amax(1), up.float().abs().amax(1), down.float().abs().transpose(0, 1).amax(1))
    )
    return values.amax(0) < float(threshold)


def stable_concat_score(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    config: NapsV2Config,
) -> torch.Tensor:
    if gate.ndim != 2 or up.shape != gate.shape or down.shape[1] != gate.shape[0]:
        raise ValueError("gate, up and down tensors are not channel-aligned")
    values = torch.cat((gate.float(), up.float(), down.float().transpose(0, 1)), dim=1)
    score = values.square().mean(1).sqrt() / (values.abs().mean(1) + config.aimer_epsilon)
    score[effective_zero_mask(gate, up, down, config.effective_zero_threshold)] = -torch.inf
    return score


def native_route(
    probes: torch.Tensor,
    router: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if probes.ndim != 2 or router.ndim != 2 or probes.shape[1] != router.shape[1]:
        raise ValueError("probes and router must have shapes [tokens, hidden] and [experts, hidden]")
    if not 0 < top_k <= router.shape[0]:
        raise ValueError("top_k must be within the number of experts")
    logits = probes.float() @ router.float().transpose(0, 1)
    selected = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :top_k]
    weights = torch.softmax(logits.gather(1, selected), dim=1)
    return logits, selected, weights


def swiglu_response(
    probes: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    gate_response = F.linear(probes.float(), gate.float())
    if activation == "silu":
        activated_gate = F.silu(gate_response)
    elif activation == "gelu_pytorch_tanh":
        activated_gate = F.gelu(gate_response, approximate="tanh")
    else:
        raise ValueError(f"Unsupported NAPS-v2 expert activation: {activation!r}")
    return activated_gate * F.linear(probes.float(), up.float())


def weighted_output_loss(
    full_output: torch.Tensor, retained_output: torch.Tensor, weights: torch.Tensor | None = None
) -> float:
    residual = (full_output.float() - retained_output.float()).square().sum(1)
    denominator = full_output.float().square().sum(1)
    if weights is not None:
        factors = weights.float().square()
        residual = residual * factors
        denominator = denominator * factors
    return float((residual.sum() / denominator.sum().clamp_min(1.0e-12)).item())


def output_for_set(responses: torch.Tensor, down: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    ids = channels.to(device=responses.device, dtype=torch.long)
    return responses.float().index_select(1, ids) @ down.float().index_select(1, ids).transpose(0, 1)


def dynamic_swap_fraction(native_probe_count: int, config: NapsV2Config) -> float:
    count = int(native_probe_count)
    for upper_bound, fraction in config.swap_fractions:
        if count <= upper_bound:
            return float(fraction)
    return 0.08


def trust_region_limit(native_probe_count: int, config: NapsV2Config) -> float:
    count = int(native_probe_count)
    for upper_bound, limit in config.trust_region_by_probe_count:
        if count <= upper_bound:
            return float(limit)
    return 0.05


def build_probe_sets(
    probes: torch.Tensor,
    selected_experts: torch.Tensor,
    selected_weights: torch.Tensor,
    expert_id: int,
) -> dict[str, Any]:
    rows, slots = torch.where(selected_experts == int(expert_id))
    native_rows = rows.to(torch.long)
    native_probes = probes.index_select(0, native_rows)
    native_weights = selected_weights[rows, slots].float()
    anchor = probes[int(expert_id):int(expert_id) + 1]
    self_naturally_routed = bool((selected_experts[int(expert_id)] == int(expert_id)).any().item())
    if native_rows.numel() and bool((native_rows == int(expert_id)).any().item()):
        coverage_rows = native_rows
        coverage_weights = native_weights
        anchor_added = False
    else:
        coverage_rows = torch.cat((native_rows, torch.tensor([expert_id], device=probes.device)))
        coverage_weights = torch.ones(coverage_rows.numel(), device=probes.device)
        anchor_added = True
    coverage_probes = probes.index_select(0, coverage_rows)
    return {
        "native_rows": native_rows,
        "native_probes": native_probes,
        "native_weights": native_weights,
        "coverage_rows": coverage_rows,
        "coverage_probes": coverage_probes,
        "coverage_weights": coverage_weights,
        "self_naturally_routed": self_naturally_routed,
        "anchor_added": anchor_added,
    }


def select_v2_mask(
    aimer_order: torch.Tensor,
    aimer_scores: torch.Tensor,
    responses: torch.Tensor,
    zero_mask: torch.Tensor,
    retained_channels: int,
    native_probe_count: int,
    config: NapsV2Config,
) -> tuple[torch.Tensor, dict[str, Any]]:
    channel_count = int(aimer_order.numel())
    if aimer_scores.shape != aimer_order.shape or zero_mask.shape != aimer_order.shape:
        raise ValueError("ranking and zero mask must be aligned")
    if not 0 < retained_channels < channel_count or responses.shape[1] != channel_count:
        raise ValueError("invalid retained width or response shape")
    baseline_keep = aimer_order[:retained_channels].to(torch.long)
    baseline_prune = aimer_order[retained_channels:].to(torch.long)
    active_keep = baseline_keep[~zero_mask[baseline_keep]]
    rescue = baseline_prune[~zero_mask[baseline_prune]]
    swap_fraction = dynamic_swap_fraction(native_probe_count, config)
    requested = min(round(swap_fraction * channel_count), int(rescue.numel()), int(active_keep.numel()))
    if requested:
        rescue_scores = responses.float().abs().mean(0)
        rescue_order = sorted(rescue.tolist(), key=lambda c: (-float(rescue_scores[c]), -float(aimer_scores[c]), c))
        swap_in = torch.tensor(rescue_order[:requested], dtype=torch.long, device=aimer_order.device)
        keep_order = active_keep.tolist()
        swap_out = torch.tensor(keep_order[-requested:], dtype=torch.long, device=aimer_order.device)
        selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=aimer_order.device)
        selected_mask[baseline_keep] = True
        selected_mask[swap_out] = False
        selected_mask[swap_in] = True
    else:
        swap_in = torch.empty(0, dtype=torch.long, device=aimer_order.device)
        swap_out = torch.empty(0, dtype=torch.long, device=aimer_order.device)
        selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=aimer_order.device)
        selected_mask[baseline_keep] = True
    selected = aimer_order[selected_mask[aimer_order]]
    if selected.numel() < retained_channels:
        selected = torch.cat(
            (selected, aimer_order[~selected_mask[aimer_order]][:retained_channels - selected.numel()])
        )
    selected = selected[:retained_channels]
    final_mask = torch.zeros(channel_count, dtype=torch.bool, device=aimer_order.device)
    final_mask[selected] = True
    order = torch.cat((selected, aimer_order[~final_mask[aimer_order]]))
    diagnostics = {
        "swap_fraction": swap_fraction,
        "requested_swaps": int(round(swap_fraction * channel_count)),
        "feasible_swaps": requested,
        "actual_swaps": int(swap_in.numel()),
        "swap_in_channels": swap_in,
        "swap_out_channels": swap_out,
        "retained": selected,
        "baseline_retained": baseline_keep,
        "rescue_candidates": rescue,
        "capacity_limited": requested < round(swap_fraction * channel_count),
    }
    return order, diagnostics


def select_compensation_targets(
    output_mass: torch.Tensor,
    retained: torch.Tensor,
    zero_mask: torch.Tensor,
    top_fraction: float,
    channel_cap: int,
) -> tuple[torch.Tensor, float]:
    active_pruned = torch.ones_like(zero_mask, dtype=torch.bool)
    active_pruned[retained.to(torch.long)] = False
    active_pruned &= ~zero_mask
    candidates = torch.where(active_pruned)[0]
    if not candidates.numel():
        return candidates, 0.0
    ranked = sorted(candidates.tolist(), key=lambda c: (-float(output_mass[c]), c))
    total = output_mass[candidates].sum().clamp_min(1.0e-12)
    chosen: list[int] = []
    cumulative = 0.0
    for channel in ranked[:int(channel_cap)]:
        chosen.append(channel)
        cumulative += float(output_mass[channel].item())
        if cumulative >= float(top_fraction) * float(total.item()):
            break
    return torch.tensor(chosen, dtype=torch.long, device=output_mass.device), cumulative / float(total.item())


def output_coverage(
    responses: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    zero_mask: torch.Tensor,
) -> dict[str, Any]:
    if responses.ndim != 2 or down.ndim != 2 or responses.shape[1] != down.shape[1]:
        raise ValueError("responses and down projection are not channel-aligned")
    response_mass = responses.float().abs().mean(0)
    channel_mass = response_mass * torch.linalg.vector_norm(down.float(), dim=0)
    channel_mass = channel_mass.masked_fill(zero_mask, 0.0)
    retained_mask = torch.zeros_like(zero_mask, dtype=torch.bool)
    retained_mask[retained.to(torch.long)] = True
    total_mass = channel_mass.sum()
    retained_mass = channel_mass[retained_mask].sum()
    coverage = float((retained_mass / total_mass.clamp_min(1.0e-12)).item()) if total_mass.item() else 1.0
    return {
        "channel_output_mass": channel_mass,
        "total_output_mass": float(total_mass.item()),
        "retained_output_mass": float(retained_mass.item()),
        "pruned_output_mass": float((total_mass - retained_mass).item()),
        "output_coverage": coverage,
        "uncovered_output_mass": max(0.0, 1.0 - coverage),
    }


def compensate_expert(
    responses: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    zero_mask: torch.Tensor,
    output_mass: torch.Tensor,
    native_probe_count: int,
    config: NapsV2Config,
    native_responses: torch.Tensor | None = None,
    native_weights: torch.Tensor | None = None,
    retained_active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    retained = retained.to(torch.long)
    active_retained = retained[~zero_mask[retained]] if retained_active is None else retained_active.to(torch.long)
    cap = config.compensation_channel_cap_b6 if retained.numel(
    ) * 2 <= down.shape[1] else config.compensation_channel_cap_b9
    targets, target_fraction = select_compensation_targets(
        output_mass, retained, zero_mask, config.coverage_top_fraction, cap
    )
    original = down.float().index_select(1, active_retained)
    diagnostics: dict[str, Any] = {
        "compensation_target_count": int(targets.numel()),
        "compensation_target_mass_fraction": target_fraction,
        "ridge_lambda": 0.0,
        "representatives_per_target": config.representatives_per_target,
        "update_ratio_raw": 0.0,
        "update_ratio_final": 0.0,
        "fallback_reason": None,
        "target_channels": targets,
        "representative_channels": [],
        "coefficients": [],
        "trust_region_scale": 0.0,
    }
    total_mass = output_mass.masked_fill(zero_mask, 0.0).sum()
    retained_mass = output_mass.index_select(0, active_retained).sum()
    uncovered_mass = float((1.0 - retained_mass / total_mass.clamp_min(1.0e-12)).item()) if total_mass.item() else 0.0
    diagnostics["uncovered_output_mass"] = max(0.0, uncovered_mass)
    if uncovered_mass < config.minimum_uncovered_mass:
        diagnostics["fallback_reason"] = "coverage_above_threshold"
        return down.float().index_select(1, retained).to(down.dtype), diagnostics
    if not targets.numel() or responses.shape[0] == 0 or active_retained.numel() == 0:
        diagnostics["fallback_reason"] = "empty_compensation_system"
        return down.float().index_select(1, retained).to(down.dtype), diagnostics
    retained_positions = {int(channel): index for index, channel in enumerate(retained.tolist())}
    active_positions = torch.tensor([retained_positions[int(c)] for c in active_retained.tolist()], device=down.device)
    h_retained = responses.float().index_select(1, active_retained)
    h_targets = responses.float().index_select(1, targets)
    probe_count = max(1, responses.shape[0])
    gram_trace = h_retained.square().sum() / probe_count
    ridge = config.ridge_scale * gram_trace / max(1, h_retained.shape[1])
    try:
        dual = h_retained @ h_retained.transpose(0, 1)
        dual = dual + probe_count * ridge * torch.eye(dual.shape[0], device=dual.device)
        coefficients = h_retained.transpose(0, 1) @ torch.linalg.solve(dual, h_targets)
        sparse = torch.zeros_like(coefficients)
        keep_count = min(config.representatives_per_target, coefficients.shape[0])
        representative_channels: list[list[int]] = []
        sparse_coefficients: list[list[float]] = []
        for target_position in range(coefficients.shape[1]):
            ids = torch.topk(coefficients[:, target_position].abs(), keep_count).indices
            sparse[ids, target_position] = coefficients[ids, target_position]
            representative_channels.append(active_retained.index_select(0, ids).tolist())
            sparse_coefficients.append(coefficients[ids, target_position].tolist())
        delta_active = down.float().index_select(1, targets) @ sparse.transpose(0, 1)
        delta = torch.zeros((down.shape[0], retained.numel()), dtype=torch.float32, device=down.device)
        delta[:, active_positions] = delta_active
        raw_ratio = float(
            (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(original).clamp_min(1.0e-12)).item()
        )
        limit = trust_region_limit(native_probe_count, config)
        scale = min(1.0, limit / max(raw_ratio, 1.0e-12))
        final_delta = delta * scale
        updated = down.float().index_select(1, retained) + final_delta
        if not torch.isfinite(updated).all():
            raise FloatingPointError("non-finite compensated down projection")
        full_output = responses.float() @ down.float().transpose(0, 1)
        mask_output = output_for_set(responses, down, retained)
        compensated_output = responses.float().index_select(1, retained) @ updated.transpose(0, 1)
        diagnostics.update({
            "ridge_lambda": float(ridge.item()),
            "update_ratio_raw": raw_ratio,
            "update_ratio_final": raw_ratio * scale,
            "representative_channels": representative_channels,
            "coefficients": sparse_coefficients,
            "trust_region_scale": scale,
            "mask_uniform_loss": weighted_output_loss(full_output, mask_output),
            "compensated_uniform_loss": weighted_output_loss(full_output, compensated_output),
        })
        if native_responses is not None and native_responses.shape[0]:
            native_full = native_responses.float() @ down.float().transpose(0, 1)
            native_mask = output_for_set(native_responses, down, retained)
            native_compensated = native_responses.float().index_select(1, retained) @ updated.transpose(0, 1)
            diagnostics.update({
                "mask_native_loss": weighted_output_loss(native_full, native_mask, native_weights),
                "compensated_native_loss": weighted_output_loss(native_full, native_compensated, native_weights),
            })
        else:
            diagnostics.update({"mask_native_loss": None, "compensated_native_loss": None})
        return updated.to(down.dtype), diagnostics
    except (RuntimeError, FloatingPointError):
        diagnostics["fallback_reason"] = "ridge_or_update_failure"
        return down.float().index_select(1, retained).to(down.dtype), diagnostics
