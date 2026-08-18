from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NapsConfig:
    effective_zero_threshold: float = 1.0e-12
    aimer_epsilon: float = 1.0e-8
    evidence_min: float = 2.0
    rank_min: float = 2.0
    evidence_saturation: float = 4.0
    rank_saturation: float = 4.0
    replacement_fraction: float = 0.025
    beta_epsilon: float = 1.0e-8
    beta_max: float = 1.0
    column_growth_max: float = 1.25
    expert_delta_max: float = 0.05
    swap_relative_tolerance: float = 1.0e-4


def effective_zero_mask(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if gate.ndim != 2 or up.shape != gate.shape or down.ndim != 2 or down.shape[1] != gate.shape[0]:
        raise ValueError("gate, up and down tensors are not channel-aligned")
    projection_max = torch.stack(
        (
            gate.float().abs().amax(dim=1),
            up.float().abs().amax(dim=1),
            down.float().abs().transpose(0, 1).amax(dim=1),
        )
    ).amax(dim=0)
    return projection_max < float(threshold)


def stable_concat_score(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    config: NapsConfig,
) -> torch.Tensor:
    if gate.ndim != 2 or up.shape != gate.shape or down.shape[1] != gate.shape[0]:
        raise ValueError("gate, up and down tensors are not channel-aligned")
    gate_f = gate.float()
    up_f = up.float()
    down_f = down.float().transpose(0, 1)
    values = torch.cat((gate_f, up_f, down_f), dim=1)
    mean_abs = values.abs().mean(dim=1)
    rms = values.square().mean(dim=1).sqrt()
    score = rms / (mean_abs + config.aimer_epsilon)
    score[effective_zero_mask(gate, up, down, config.effective_zero_threshold)] = -torch.inf
    return score


def native_route(
    probes: torch.Tensor,
    router: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if probes.ndim != 2 or router.ndim != 2 or probes.shape[1] != router.shape[1]:
        raise ValueError("probes and router must have shapes [tokens, hidden] and [experts, hidden]")
    logits = probes.float() @ router.float().transpose(0, 1)
    if top_k <= 0 or top_k > router.shape[0]:
        raise ValueError("top_k must be within the number of experts")
    selected = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :int(top_k)]
    top_logits = logits.gather(1, selected)
    weights = torch.softmax(top_logits, dim=1)
    return logits, selected, weights


def swiglu_response(probes: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(F.linear(probes.float(), gate.float())) * F.linear(probes.float(), up.float())


def effective_evidence(
    probes: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[float, float]:
    if probes.ndim != 2 or weights.ndim != 1 or probes.shape[0] != weights.numel():
        raise ValueError("probe and weight dimensions do not align")
    if probes.shape[0] == 0:
        return 0.0, 0.0
    weights_f = weights.float()
    n_eff = float(
        (weights_f.sum().square() / weights_f.square().sum().clamp_min(1.0e-12)).item()
    )
    normalized = F.normalize(probes.float(), p=2, dim=1, eps=1.0e-12)
    weighted = normalized * weights_f.sqrt().unsqueeze(1)
    gram = weighted @ weighted.transpose(0, 1)
    trace = torch.trace(gram)
    rank = float((trace.square() / gram.square().sum().clamp_min(1.0e-12)).item())
    return n_eff, rank


def weighted_output_loss(
    full_output: torch.Tensor,
    retained_output: torch.Tensor,
    weights: torch.Tensor | None,
) -> float:
    residual = (full_output.float() - retained_output.float()).square().sum(dim=1)
    denominator = full_output.float().square().sum(dim=1)
    if weights is not None:
        weight = weights.float().square()
        residual = residual * weight
        denominator = denominator * weight
    return float((residual.sum() / denominator.sum().clamp_min(1.0e-12)).item())


def output_for_set(
    responses: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
) -> torch.Tensor:
    ids = retained.to(device=responses.device, dtype=torch.long)
    return responses.float().index_select(1, ids) @ down.float().index_select(1, ids).transpose(0, 1)


def _swap_tolerance(value: float, relative: float) -> float:
    return max(1.0e-8, abs(value) * float(relative))


def select_mask(
    aimer_order: torch.Tensor,
    aimer_scores: torch.Tensor,
    responses: torch.Tensor,
    down: torch.Tensor,
    routed_weights: torch.Tensor,
    zero_mask: torch.Tensor,
    retained_channels: int,
    evidence_budget: int,
    config: NapsConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    channel_count = int(aimer_order.numel())
    if aimer_order.ndim != 1 or aimer_scores.shape != aimer_order.shape:
        raise ValueError("AIMER order and scores must be one-dimensional and aligned")
    if zero_mask.shape != aimer_order.shape or retained_channels <= 0 or retained_channels >= channel_count:
        raise ValueError("invalid channel mask or retained width")
    if responses.shape[1] != channel_count or down.shape[1] != channel_count:
        raise ValueError("responses/down channels do not match ranking")
    baseline_keep = aimer_order[:retained_channels].to(torch.long)
    baseline_prune = aimer_order[retained_channels:].to(torch.long)
    active_prune = baseline_prune[~zero_mask[baseline_prune]]
    if evidence_budget <= 0 or active_prune.numel() == 0 or responses.shape[0] == 0:
        return aimer_order, {"accepted_swaps": 0, "rescue_candidates": 0, "fallback": True}

    weights = routed_weights.float()
    activity = (responses.float().abs() * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1.0e-12)
    output_proxy = torch.sqrt(
        (responses.float().square() * weights[:, None].square()).sum(dim=0)
        / weights.square().sum().clamp_min(1.0e-12)
    ) * torch.linalg.vector_norm(down.float(), dim=0)
    candidate_count = min(int(evidence_budget), int(active_prune.numel()))
    rescue_activity = active_prune[torch.topk(activity[active_prune], candidate_count, sorted=True).indices]
    rescue_output = active_prune[torch.topk(output_proxy[active_prune], candidate_count, sorted=True).indices]
    rescue = torch.unique(torch.cat((rescue_activity, rescue_output)), sorted=False)
    drop_count = min(int(2 * evidence_budget), retained_channels)
    drops = baseline_keep[-drop_count:]

    full_output = responses.float() @ down.float().transpose(0, 1)
    selected = baseline_keep.clone()
    native = weighted_output_loss(full_output, output_for_set(responses, down, selected), weights)
    uniform = weighted_output_loss(full_output, output_for_set(responses, down, selected), None)
    accepted = 0
    while accepted < int(evidence_budget):
        best: tuple[float, float, int, int, torch.Tensor] | None = None
        selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=selected.device)
        selected_mask[selected] = True
        for rescue_id in rescue.tolist():
            if bool(selected_mask[rescue_id]):
                continue
            for drop_id in drops.tolist():
                if not bool(selected_mask[drop_id]):
                    continue
                proposal = selected[selected != drop_id]
                proposal = torch.cat((proposal, torch.tensor([rescue_id], device=proposal.device)))
                proposal_native = weighted_output_loss(full_output, output_for_set(responses, down, proposal), weights)
                proposal_uniform = weighted_output_loss(full_output, output_for_set(responses, down, proposal), None)
                tolerance = _swap_tolerance(native, config.swap_relative_tolerance)
                if proposal_native >= native - tolerance or proposal_uniform > uniform + 1.0e-8:
                    continue
                key = (
                    proposal_native,
                    proposal_uniform,
                    -float(aimer_scores[rescue_id].item()),
                    float(aimer_scores[drop_id].item()),
                )
                if best is None or key < best[:4]:
                    best = (*key, proposal)
        if best is None:
            break
        native, uniform, _, _, selected = best
        accepted += 1

    selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=selected.device)
    selected_mask[selected] = True
    remaining = aimer_order[~selected_mask[aimer_order]]
    order = torch.cat((selected, remaining))
    diagnostics = {
        "accepted_swaps": accepted,
        "rescue_candidates": int(rescue.numel()),
        "activity_candidates": int(rescue_activity.numel()),
        "output_candidates": int(rescue_output.numel()),
        "displaced_channels": int(accepted),
        "baseline_native_loss": weighted_output_loss(full_output, output_for_set(responses, down, baseline_keep), weights),
        "naps_native_loss": native,
        "baseline_uniform_loss": weighted_output_loss(full_output, output_for_set(responses, down, baseline_keep), None),
        "naps_uniform_loss": uniform,
        "fallback": accepted == 0,
        "retained": selected,
        "rescue": rescue,
        "displaced": baseline_keep[~selected_mask[baseline_keep]],
    }
    return order, diagnostics


def build_one_to_one_merge_plan(
    responses: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    displaced: torch.Tensor,
    routed_weights: torch.Tensor,
    config: NapsConfig,
) -> dict[str, Any]:
    if displaced.numel() == 0 or retained.numel() == 0:
        return {"pairs": [], "rejected": {}}
    retained = retained.to(torch.long)
    displaced = displaced.to(torch.long)
    weight_matrix = torch.diag(routed_weights.float().square())
    pairs: list[dict[str, float | int]] = []
    used: set[int] = set()
    rejected: dict[str, int] = {}
    candidates = []
    for p in displaced.tolist():
        a_p = responses[:, p]
        for r in retained.tolist():
            a_r = responses[:, r]
            beta = float(
                (a_r @ weight_matrix @ a_p / (a_r @ weight_matrix @ a_r + config.beta_epsilon)).item()
            )
            residual = (a_p - beta * a_r) @ weight_matrix @ (a_p - beta * a_r)
            cost = float(residual.item() * down[:, p].float().square().sum().item())
            if abs(beta) > config.beta_max:
                rejected["beta"] = rejected.get("beta", 0) + 1
                continue
            updated = down[:, r].float() + beta * down[:, p].float()
            growth = float((torch.linalg.vector_norm(updated) / torch.linalg.vector_norm(down[:, r]).clamp_min(1.0e-12)).item())
            if growth > config.column_growth_max:
                rejected["column_growth"] = rejected.get("column_growth", 0) + 1
                continue
            candidates.append((cost, p, r, beta, growth))
    for cost, p, r, beta, growth in sorted(candidates):
        if p in {pair["pruned"] for pair in pairs} or r in used:
            continue
        used.add(r)
        pairs.append({"pruned": p, "representative": r, "beta": beta, "column_growth": growth, "cost": cost})
    return {"pairs": pairs, "rejected": rejected}


def validate_merge_plan(
    responses: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    routed_weights: torch.Tensor,
    plan: dict[str, Any],
    config: NapsConfig,
) -> tuple[dict[str, Any], torch.Tensor]:
    retained = retained.to(torch.long)
    original = down.float().index_select(1, retained)
    merged = original.clone()
    retained_positions = {int(channel): index for index, channel in enumerate(retained.tolist())}
    accepted_pairs = []
    for pair in plan.get("pairs", []):
        representative = int(pair["representative"])
        position = retained_positions.get(representative)
        if position is None:
            continue
        pruned = int(pair["pruned"])
        beta = float(pair["beta"])
        merged[:, position] += beta * down[:, pruned].float()
        accepted_pairs.append(pair)
    delta_ratio = float(
        (torch.linalg.vector_norm(merged - original) / torch.linalg.vector_norm(original).clamp_min(1.0e-12)).item()
    )
    full_output = responses.float() @ down.float().transpose(0, 1)
    mask_output = responses.float().index_select(1, retained) @ original.transpose(0, 1)
    merge_output = responses.float().index_select(1, retained) @ merged.transpose(0, 1)
    mask_native = weighted_output_loss(full_output, mask_output, routed_weights)
    mask_uniform = weighted_output_loss(full_output, mask_output, None)
    merge_native = weighted_output_loss(full_output, merge_output, routed_weights)
    merge_uniform = weighted_output_loss(full_output, merge_output, None)
    fallback_reason = None
    if delta_ratio > config.expert_delta_max:
        fallback_reason = "expert_delta"
    elif merge_native > mask_native + 1.0e-8:
        fallback_reason = "native_loss"
    elif merge_uniform > mask_uniform + 1.0e-8:
        fallback_reason = "uniform_loss"
    if fallback_reason is not None:
        accepted_pairs = []
        merged = original
    validated = {
        "pairs": accepted_pairs,
        "rejected": plan.get("rejected", {}),
        "fallback_reason": fallback_reason,
        "expert_delta_ratio": delta_ratio,
        "mask_native_loss": mask_native,
        "merge_native_loss": merge_native,
        "mask_uniform_loss": mask_uniform,
        "merge_uniform_loss": merge_uniform,
    }
    return validated, merged
