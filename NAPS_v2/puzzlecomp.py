from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PuzzleCompConfig:
    similarity_threshold: float = 0.4
    reserve_fraction: float = 0.05
    epsilon: float = 1.0e-8
    acceptance_tolerance: float = 1.0e-6
    activation: str = "silu"


@dataclass(frozen=True)
class PuzzleStoragePlan:
    source_width: int
    retained_width: int
    reserve_channels: int
    core_channels_per_expert: int
    shared_residual_channels_per_pair: int
    effective_channels_per_expert: int
    stored_channels_per_pair: int
    materialized_channels_per_pair: int


def build_storage_plan(
    source_width: int,
    retained_width: int,
    reserve_fraction: float,
) -> PuzzleStoragePlan:
    if not 0 < retained_width < source_width:
        raise ValueError("retained_width must be positive and smaller than source_width")
    if not 0 < reserve_fraction < 0.5:
        raise ValueError("reserve_fraction must be between zero and one half")
    reserve_channels = round(float(reserve_fraction) * source_width)
    if not 0 < reserve_channels < retained_width:
        raise ValueError("reserve_fraction produces an invalid residual reserve")
    core_channels = retained_width - reserve_channels
    shared_residual_channels = 2 * reserve_channels
    effective_channels = core_channels + shared_residual_channels
    if effective_channels > source_width:
        raise ValueError("effective PuzzleComp width exceeds the source expert width")
    stored_channels = 2 * core_channels + shared_residual_channels
    if stored_channels != 2 * retained_width:
        raise RuntimeError("PuzzleComp pair storage does not preserve the retained-width budget")
    return PuzzleStoragePlan(
        source_width=source_width,
        retained_width=retained_width,
        reserve_channels=reserve_channels,
        core_channels_per_expert=core_channels,
        shared_residual_channels_per_pair=shared_residual_channels,
        effective_channels_per_expert=effective_channels,
        stored_channels_per_pair=stored_channels,
        materialized_channels_per_pair=2 * effective_channels,
    )


def split_ranked_channels(
    ranking: torch.Tensor,
    plan: PuzzleStoragePlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ranking = ranking.to(torch.long)
    if ranking.ndim != 1 or ranking.numel() != plan.source_width:
        raise ValueError("ranking must contain every source channel exactly once")
    expected = torch.arange(plan.source_width, device=ranking.device)
    if not torch.equal(torch.sort(ranking).values, expected):
        raise ValueError("ranking must be a permutation of source channel indices")
    core_end = plan.core_channels_per_expert
    retained_end = plan.retained_width
    return ranking[:core_end], ranking[core_end:retained_end], ranking[retained_end:]


def activation_weight_saliency(weight: torch.Tensor, input_rms: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or input_rms.ndim != 1 or weight.shape[1] != input_rms.numel():
        raise ValueError("weight and input_rms must have shapes [output, input] and [input]")
    return weight.float().abs() * input_rms.float().unsqueeze(0)


def channel_input_saliency(weight: torch.Tensor, channel_rms: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or channel_rms.ndim != 1 or weight.shape[0] != channel_rms.numel():
        raise ValueError("weight and channel_rms must have shapes [channels, output] and [channels]")
    return weight.float().abs() * channel_rms.float().unsqueeze(1)


def puzzle_merge_weight_pair(
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    left_saliency: torch.Tensor,
    right_saliency: torch.Tensor,
    config: PuzzleCompConfig,
) -> dict[str, torch.Tensor]:
    if (
        left_weight.shape != right_weight.shape
        or left_weight.shape != left_saliency.shape
        or left_weight.shape != right_saliency.shape
    ):
        raise ValueError("paired weights and saliency tensors must have identical shapes")
    left = left_weight.float()
    right = right_weight.float()
    relative_difference = (left.abs() - right.abs()).abs() / (
        left.abs() + right.abs() + config.epsilon
    )
    similarity_mask = relative_difference <= config.similarity_threshold
    left_saliency_mask = left_saliency.float() >= right_saliency.float()
    right_saliency_mask = ~left_saliency_mask
    left_mask = similarity_mask | left_saliency_mask
    right_mask = similarity_mask | right_saliency_mask
    merged_magnitude = torch.where(
        similarity_mask,
        0.5 * (left.abs() + right.abs()),
        torch.where(left_saliency_mask, left.abs(), right.abs()),
    )
    left_sign = torch.where(left < 0, -torch.ones_like(left), torch.ones_like(left))
    right_sign = torch.where(right < 0, -torch.ones_like(right), torch.ones_like(right))
    return {
        "merged_magnitude": merged_magnitude,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "left_sign": left_sign,
        "right_sign": right_sign,
        "left_reconstructed": left_sign * left_mask.float() * merged_magnitude,
        "right_reconstructed": right_sign * right_mask.float() * merged_magnitude,
        "similarity_mask": similarity_mask,
    }


def _channel_responses(
    probes: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    gate_output = F.linear(probes.float(), gate.float())
    if activation == "silu":
        activated_gate = F.silu(gate_output)
    elif activation == "gelu_pytorch_tanh":
        activated_gate = F.gelu(gate_output, approximate="tanh")
    else:
        raise ValueError(f"Unsupported PuzzleComp activation: {activation!r}")
    return activated_gate * F.linear(probes.float(), up.float())


def _expert_output(
    probes: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    return _channel_responses(probes, gate, up, activation) @ down.float().transpose(0, 1)


def _weighted_output_loss(
    full_output: torch.Tensor,
    approximate_output: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> float:
    residual = (full_output.float() - approximate_output.float()).square().sum(dim=1)
    denominator = full_output.float().square().sum(dim=1)
    if weights is not None:
        factors = weights.float().square()
        residual = residual * factors
        denominator = denominator * factors
    return float((residual.sum() / denominator.sum().clamp_min(1.0e-12)).item())


def _select_shared_channels(
    left_gate: torch.Tensor,
    left_up: torch.Tensor,
    left_down: torch.Tensor,
    right_gate: torch.Tensor,
    right_up: torch.Tensor,
    right_down: torch.Tensor,
    left_probes: torch.Tensor,
    right_probes: torch.Tensor,
    left_core: torch.Tensor,
    right_core: torch.Tensor,
    left_zero_mask: torch.Tensor,
    right_zero_mask: torch.Tensor,
    count: int,
    activation: str,
) -> torch.Tensor:
    source_width = left_gate.shape[0]
    eligible = torch.ones(source_width, dtype=torch.bool, device=left_gate.device)
    eligible[left_core] = False
    eligible[right_core] = False
    eligible &= ~(left_zero_mask & right_zero_mask)
    candidates = torch.where(eligible)[0]
    if candidates.numel() < count:
        return torch.empty(0, dtype=torch.long, device=left_gate.device)
    left_responses = _channel_responses(left_probes, left_gate, left_up, activation)
    right_responses = _channel_responses(right_probes, right_gate, right_up, activation)
    left_mass = left_responses.abs().mean(dim=0) * left_down.float().norm(dim=0)
    right_mass = right_responses.abs().mean(dim=0) * right_down.float().norm(dim=0)
    left_mass = left_mass / left_mass.sum().clamp_min(1.0e-12)
    right_mass = right_mass / right_mass.sum().clamp_min(1.0e-12)
    pair_mass = left_mass + right_mass
    order = torch.argsort(pair_mass[candidates], descending=True, stable=True)
    return candidates.index_select(0, order[:count])


def _merge_shared_channels(
    left_gate: torch.Tensor,
    left_up: torch.Tensor,
    left_down: torch.Tensor,
    right_gate: torch.Tensor,
    right_up: torch.Tensor,
    right_down: torch.Tensor,
    left_probes: torch.Tensor,
    right_probes: torch.Tensor,
    shared_channels: torch.Tensor,
    config: PuzzleCompConfig,
) -> dict[str, Any]:
    left_gate_shared = left_gate.index_select(0, shared_channels)
    left_up_shared = left_up.index_select(0, shared_channels)
    left_down_shared = left_down.index_select(1, shared_channels)
    right_gate_shared = right_gate.index_select(0, shared_channels)
    right_up_shared = right_up.index_select(0, shared_channels)
    right_down_shared = right_down.index_select(1, shared_channels)
    left_hidden_rms = left_probes.float().square().mean(dim=0).sqrt()
    right_hidden_rms = right_probes.float().square().mean(dim=0).sqrt()
    left_response_rms = _channel_responses(
        left_probes, left_gate_shared, left_up_shared, config.activation
    ).square().mean(dim=0).sqrt()
    right_response_rms = _channel_responses(
        right_probes, right_gate_shared, right_up_shared, config.activation
    ).square().mean(dim=0).sqrt()
    gate_pair = puzzle_merge_weight_pair(
        left_gate_shared,
        right_gate_shared,
        activation_weight_saliency(left_gate_shared, left_hidden_rms),
        activation_weight_saliency(right_gate_shared, right_hidden_rms),
        config,
    )
    up_pair = puzzle_merge_weight_pair(
        left_up_shared,
        right_up_shared,
        activation_weight_saliency(left_up_shared, left_hidden_rms),
        activation_weight_saliency(right_up_shared, right_hidden_rms),
        config,
    )
    left_down_channels = left_down_shared.transpose(0, 1)
    right_down_channels = right_down_shared.transpose(0, 1)
    down_pair = puzzle_merge_weight_pair(
        left_down_channels,
        right_down_channels,
        channel_input_saliency(left_down_channels, left_response_rms),
        channel_input_saliency(right_down_channels, right_response_rms),
        config,
    )
    return {
        "left_gate": gate_pair["left_reconstructed"],
        "left_up": up_pair["left_reconstructed"],
        "left_down": down_pair["left_reconstructed"].transpose(0, 1),
        "right_gate": gate_pair["right_reconstructed"],
        "right_up": up_pair["right_reconstructed"],
        "right_down": down_pair["right_reconstructed"].transpose(0, 1),
        "packed": {"gate": gate_pair, "up": up_pair, "down": down_pair},
        "diagnostics": {
            "gate_similarity_fraction": float(gate_pair["similarity_mask"].float().mean().item()),
            "up_similarity_fraction": float(up_pair["similarity_mask"].float().mean().item()),
            "down_similarity_fraction": float(down_pair["similarity_mask"].float().mean().item()),
            "left_gate_density": float(gate_pair["left_mask"].float().mean().item()),
            "right_gate_density": float(gate_pair["right_mask"].float().mean().item()),
            "left_up_density": float(up_pair["left_mask"].float().mean().item()),
            "right_up_density": float(up_pair["right_mask"].float().mean().item()),
            "left_down_density": float(down_pair["left_mask"].float().mean().item()),
            "right_down_density": float(down_pair["right_mask"].float().mean().item()),
        },
    }


def _baseline_expert(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    ranking: torch.Tensor,
    retained_width: int,
) -> dict[str, torch.Tensor]:
    ids = ranking[:retained_width].to(torch.long)
    return {
        "gate": gate.index_select(0, ids).float(),
        "up": up.index_select(0, ids).float(),
        "down": down.index_select(1, ids).float(),
    }


def pairwise_puzzle_compensate(
    left_gate: torch.Tensor,
    left_up: torch.Tensor,
    left_down: torch.Tensor,
    right_gate: torch.Tensor,
    right_up: torch.Tensor,
    right_down: torch.Tensor,
    left_ranking: torch.Tensor,
    right_ranking: torch.Tensor,
    retained_width: int,
    left_probes: torch.Tensor,
    right_probes: torch.Tensor,
    left_native_probes: torch.Tensor | None = None,
    right_native_probes: torch.Tensor | None = None,
    left_native_weights: torch.Tensor | None = None,
    right_native_weights: torch.Tensor | None = None,
    left_zero_mask: torch.Tensor | None = None,
    right_zero_mask: torch.Tensor | None = None,
    config: PuzzleCompConfig | None = None,
) -> dict[str, Any]:
    config = config or PuzzleCompConfig()
    if left_gate.shape != left_up.shape or right_gate.shape != right_up.shape:
        raise ValueError("gate and up tensors must be channel-aligned")
    if left_gate.shape != right_gate.shape:
        raise ValueError("paired experts must have identical tensor shapes")
    source_width, hidden_size = left_gate.shape
    if left_down.shape != (hidden_size, source_width) or right_down.shape != left_down.shape:
        raise ValueError("down tensors must have shape [hidden, source_width]")
    if left_ranking.numel() != source_width or right_ranking.numel() != source_width:
        raise ValueError("rankings must cover the complete source width")
    if left_probes.ndim != 2 or right_probes.ndim != 2:
        raise ValueError("probe tensors must be rank two")
    if left_probes.shape[1] != hidden_size or right_probes.shape[1] != hidden_size:
        raise ValueError("probe hidden size does not match expert weights")

    plan = build_storage_plan(source_width, retained_width, config.reserve_fraction)
    left_zero_mask = (
        torch.zeros(source_width, dtype=torch.bool, device=left_gate.device)
        if left_zero_mask is None else left_zero_mask.to(torch.bool)
    )
    right_zero_mask = (
        torch.zeros(source_width, dtype=torch.bool, device=right_gate.device)
        if right_zero_mask is None else right_zero_mask.to(torch.bool)
    )
    left_core, left_sacrificed, _ = split_ranked_channels(left_ranking, plan)
    right_core, right_sacrificed, _ = split_ranked_channels(right_ranking, plan)
    shared_channels = _select_shared_channels(
        left_gate,
        left_up,
        left_down,
        right_gate,
        right_up,
        right_down,
        left_probes,
        right_probes,
        left_core,
        right_core,
        left_zero_mask,
        right_zero_mask,
        plan.shared_residual_channels_per_pair,
        config.activation,
    )
    baseline_left = _baseline_expert(left_gate, left_up, left_down, left_ranking, retained_width)
    baseline_right = _baseline_expert(right_gate, right_up, right_down, right_ranking, retained_width)
    diagnostics: dict[str, Any] = {
        "config": asdict(config),
        "storage_plan": asdict(plan),
        "left_core_channels": left_core.tolist(),
        "right_core_channels": right_core.tolist(),
        "left_sacrificed_channels": left_sacrificed.tolist(),
        "right_sacrificed_channels": right_sacrificed.tolist(),
        "shared_channels": shared_channels.tolist(),
        "fallback_reason": None,
        "accepted": False,
    }
    if shared_channels.numel() != plan.shared_residual_channels_per_pair:
        diagnostics["fallback_reason"] = "insufficient_shared_channels"
        return {
            "accepted": False,
            "left": baseline_left,
            "right": baseline_right,
            "packed_residual": None,
            "diagnostics": diagnostics,
        }
    merged = _merge_shared_channels(
        left_gate,
        left_up,
        left_down,
        right_gate,
        right_up,
        right_down,
        left_probes,
        right_probes,
        shared_channels,
        config,
    )
    candidate_left = {
        "gate": torch.cat((left_gate.index_select(0, left_core), merged["left_gate"]), dim=0).float(),
        "up": torch.cat((left_up.index_select(0, left_core), merged["left_up"]), dim=0).float(),
        "down": torch.cat((left_down.index_select(1, left_core), merged["left_down"]), dim=1).float(),
    }
    candidate_right = {
        "gate": torch.cat((right_gate.index_select(0, right_core), merged["right_gate"]), dim=0).float(),
        "up": torch.cat((right_up.index_select(0, right_core), merged["right_up"]), dim=0).float(),
        "down": torch.cat((right_down.index_select(1, right_core), merged["right_down"]), dim=1).float(),
    }
    diagnostics.update(merged["diagnostics"])

    accepted = True
    missing_native_probe = False
    for side, probes, native_probes, native_weights, full_weights, baseline, candidate in (
        (
            "left", left_probes, left_native_probes, left_native_weights,
            (left_gate, left_up, left_down), baseline_left, candidate_left,
        ),
        (
            "right", right_probes, right_native_probes, right_native_weights,
            (right_gate, right_up, right_down), baseline_right, candidate_right,
        ),
    ):
        full_output = _expert_output(probes, *full_weights, activation=config.activation)
        baseline_loss = _weighted_output_loss(
            full_output, _expert_output(probes, **baseline, activation=config.activation)
        )
        candidate_loss = _weighted_output_loss(
            full_output, _expert_output(probes, **candidate, activation=config.activation)
        )
        diagnostics[f"{side}_baseline_loss"] = baseline_loss
        diagnostics[f"{side}_candidate_loss"] = candidate_loss
        accepted &= candidate_loss <= baseline_loss + config.acceptance_tolerance
        if native_probes is not None and native_probes.shape[0]:
            native_full = _expert_output(native_probes, *full_weights, activation=config.activation)
            native_baseline_loss = _weighted_output_loss(
                native_full,
                _expert_output(native_probes, **baseline, activation=config.activation),
                native_weights,
            )
            native_candidate_loss = _weighted_output_loss(
                native_full,
                _expert_output(native_probes, **candidate, activation=config.activation),
                native_weights,
            )
            diagnostics[f"{side}_native_baseline_loss"] = native_baseline_loss
            diagnostics[f"{side}_native_candidate_loss"] = native_candidate_loss
            accepted &= native_candidate_loss <= native_baseline_loss + config.acceptance_tolerance
        else:
            diagnostics[f"{side}_native_gate_missing"] = True
            missing_native_probe = True

    if missing_native_probe:
        accepted = False
        diagnostics["fallback_reason"] = "missing_native_probes"
    elif not accepted:
        diagnostics["fallback_reason"] = "loss_gate_rejected"
    diagnostics["accepted"] = bool(accepted)
    return {
        "accepted": bool(accepted),
        "left": candidate_left if accepted else baseline_left,
        "right": candidate_right if accepted else baseline_right,
        "candidate_left": candidate_left,
        "candidate_right": candidate_right,
        "packed_residual": merged["packed"] if accepted else None,
        "diagnostics": diagnostics,
    }