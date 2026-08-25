"""Model-independent probe, selection, and ridge-folding primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CompensationResult:
    """Result of held-out ridge residual folding for one expert."""

    down: torch.Tensor
    accepted: bool
    ridge: float | None
    trust_ratio: float | None
    baseline_error: float
    compensated_error: float
    update_ratio: float


def rms_normalize(value: torch.Tensor, target_rms: float = 1.0, epsilon: float = 1.0e-8) -> torch.Tensor:
    """Normalize each row to a requested root-mean-square magnitude."""

    if value.ndim < 1:
        raise ValueError("value must have at least one dimension")
    rms = value.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(epsilon)
    return value * (float(target_rms) / rms).to(dtype=value.dtype)


def route_topk(
    logits: torch.Tensor,
    top_k: int,
    *,
    scoring: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts and normalize their weights from router logits."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [tokens, experts]")
    if not 0 < int(top_k) <= logits.shape[1]:
        raise ValueError("top_k must be in [1, number of experts]")
    if scoring not in {"softmax", "sigmoid"}:
        raise ValueError("scoring must be 'softmax' or 'sigmoid'")
    values, indices = torch.topk(logits, int(top_k), dim=-1)
    if scoring == "softmax":
        weights = torch.softmax(values.float(), dim=-1).to(dtype=logits.dtype)
    else:
        weights = torch.sigmoid(values.float())
        weights = (weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)).to(dtype=logits.dtype)
    return indices, weights


def build_router_probes(
    effective_directions: torch.Tensor,
    *,
    variants: int,
    sigmas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2),
    scale: float = 1.0,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Construct deterministic router-region probes with cyclic, near, and far directions."""

    if effective_directions.ndim != 2:
        raise ValueError("effective_directions must have shape [experts, hidden]")
    experts, hidden = effective_directions.shape
    if experts < 2:
        raise ValueError("At least two experts are required for router perturbations")
    if variants <= 0 or scale <= 0.0 or not sigmas or any(float(item) < 0.0 for item in sigmas):
        raise ValueError("variants, scale, and sigmas are invalid")
    directions = rms_normalize(effective_directions.float(), epsilon=epsilon)
    normalized = F.normalize(directions, dim=-1, eps=epsilon)
    cosine = normalized @ normalized.transpose(0, 1)
    nearest_cosine = cosine.clone()
    nearest_cosine.fill_diagonal_(float("-inf"))
    farthest_cosine = cosine.clone()
    farthest_cosine.fill_diagonal_(float("inf"))
    nearest = nearest_cosine.argmax(dim=-1)
    farthest = farthest_cosine.argmin(dim=-1)
    rows: list[torch.Tensor] = []
    for variant in range(int(variants)):
        sigma = float(sigmas[variant % len(sigmas)])
        if variant % 3 == 0:
            partner = torch.roll(torch.arange(experts, device=directions.device), shifts=-1)
        elif variant % 3 == 1:
            partner = nearest
        else:
            partner = farthest
        base = directions
        other = directions.index_select(0, partner)
        orthogonal = other - (other * base).sum(dim=-1, keepdim=True) * base / base.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(epsilon)
        rows.append(rms_normalize(base + sigma * orthogonal, target_rms=scale, epsilon=epsilon))
    return torch.stack(rows, dim=1).reshape(experts, int(variants), hidden)


def optimize_router_margin(
    initial_probe: torch.Tensor,
    expert_id: int,
    route_logits_fn: object,
    *,
    steps: int = 8,
    learning_rate: float = 0.05,
    proximity_weight: float = 0.01,
    target_rms: float = 1.0,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Move one probe toward a target router region using only model parameters."""

    if steps < 0 or learning_rate <= 0.0 or proximity_weight < 0.0:
        raise ValueError("Invalid router optimization settings")
    probe = initial_probe.detach().float().clone().requires_grad_(True)
    anchor = probe.detach().clone()
    for _ in range(int(steps)):
        logits = route_logits_fn(probe.unsqueeze(0))
        if logits.ndim != 2 or expert_id >= logits.shape[-1]:
            raise ValueError("route_logits_fn must return [tokens, experts]")
        target = logits[:, int(expert_id)]
        other = logits.masked_fill(
            F.one_hot(torch.tensor(expert_id, device=logits.device), logits.shape[-1]).bool().unsqueeze(0),
            float("-inf"),
        ).max(dim=-1).values
        loss = -(target - other).mean() + float(proximity_weight) * (probe - anchor).square().mean()
        gradient = torch.autograd.grad(loss, probe, only_inputs=True)[0]
        with torch.no_grad():
            probe -= float(learning_rate) * gradient
            probe.copy_(rms_normalize(probe, target_rms=target_rms, epsilon=epsilon))
        probe.requires_grad_(True)
    return probe.detach()


def collect_routed_rows(
    expert_input: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    *,
    activation: str,
    max_rows_per_expert: int,
) -> dict[int, torch.Tensor]:
    """Collect route-weighted expert channel responses grouped by expert."""

    if expert_input.ndim != 2:
        raise ValueError("expert_input must have shape [tokens, hidden]")
    if top_k_index.shape != top_k_weights.shape or top_k_index.shape[0] != expert_input.shape[0]:
        raise ValueError("Route tensors must align with expert_input")
    if gate_up_proj.ndim != 3 or gate_up_proj.shape[1] % 2:
        raise ValueError("gate_up_proj must have shape [experts, 2 * channels, hidden]")
    if max_rows_per_expert <= 0:
        raise ValueError("max_rows_per_expert must be positive")
    experts, packed_channels, hidden = gate_up_proj.shape
    channels = packed_channels // 2
    if expert_input.shape[1] != hidden:
        raise ValueError("expert_input hidden size does not match gate_up_proj")
    rows: dict[int, list[torch.Tensor]] = {expert: [] for expert in range(experts)}
    flat_index = top_k_index.reshape(-1).long()
    flat_weights = top_k_weights.reshape(-1).float()
    token_index = torch.arange(expert_input.shape[0], device=expert_input.device).repeat_interleave(top_k_index.shape[1])
    response_input = expert_input.float()
    for expert in range(experts):
        selected = flat_index == expert
        if not bool(selected.any()):
            continue
        token_rows = token_index[selected]
        weights = flat_weights[selected].clamp_min(0.0).unsqueeze(1)
        gate = gate_up_proj[expert, :channels].float()
        up = gate_up_proj[expert, channels:].float()
        gate_response = F.linear(response_input.index_select(0, token_rows), gate)
        up_response = F.linear(response_input.index_select(0, token_rows), up)
        if activation in {"silu", "swish"}:
            response = F.silu(gate_response) * up_response
        elif activation in {"gelu", "gelu_pytorch_tanh"}:
            response = F.gelu(gate_response, approximate="tanh") * up_response
        else:
            raise ValueError(f"Unsupported activation: {activation!r}")
        weighted = response * weights.sqrt()
        if weighted.shape[0] > max_rows_per_expert:
            order = torch.argsort(weights[:, 0], descending=True, stable=True)[:max_rows_per_expert]
            weighted = weighted.index_select(0, order)
        rows[expert].append(weighted)
    return {
        expert: torch.cat(chunks, dim=0) if chunks else torch.empty((0, channels), device=expert_input.device)
        for expert, chunks in rows.items()
    }


def build_source_balanced_matrices(
    source_rows: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    num_experts: int,
    max_rows_per_expert_per_origin: int,
) -> dict[int, torch.Tensor]:
    """Combine origin rows so each origin contributes equal total expert mass."""

    if max_rows_per_expert_per_origin <= 0:
        raise ValueError("max_rows_per_expert_per_origin must be positive")
    matrices: dict[int, list[torch.Tensor]] = {expert: [] for expert in range(num_experts)}
    for source_name, expert_rows in source_rows.items():
        del source_name
        for expert in range(num_experts):
            rows = expert_rows.get(expert)
            if rows is None or rows.ndim != 2 or rows.shape[0] == 0:
                continue
            rows = rows[:max_rows_per_expert_per_origin].float()
            rows = rows / float(max(rows.shape[0], 1)) ** 0.5
            matrices[expert].append(rows)
    return {
        expert: torch.cat(rows, dim=0) if rows else torch.empty((0, 0))
        for expert, rows in matrices.items()
    }


def output_energy_scores(activation_matrix: torch.Tensor, down_proj: torch.Tensor) -> torch.Tensor:
    """Score channels by activation energy times down-projection output energy."""

    if activation_matrix.ndim != 2 or down_proj.ndim != 2:
        raise ValueError("activation_matrix and down_proj must be matrices")
    if activation_matrix.shape[1] != down_proj.shape[1]:
        raise ValueError("activation channels must match down_proj columns")
    return activation_matrix.float().square().sum(dim=0) * down_proj.float().square().sum(dim=0)


def _ridge_coefficients(
    retained: torch.Tensor,
    pruned: torch.Tensor,
    ridge: float,
    epsilon: float,
) -> torch.Tensor:
    """Solve A_R C ~= A_P in dual form and return C with shape [K, P]."""

    if retained.ndim != 2 or pruned.ndim != 2 or retained.shape[0] != pruned.shape[0]:
        raise ValueError("retained and pruned matrices must share their row count")
    scale = retained.square().sum() / max(retained.shape[0], 1)
    gram = retained @ retained.transpose(0, 1)
    gram.diagonal().add_(float(ridge) * scale.clamp_min(epsilon))
    solved = torch.linalg.solve(gram, pruned)
    return retained.transpose(0, 1) @ solved


def _reconstruction_error(activation_matrix: torch.Tensor, down_proj: torch.Tensor, keep: torch.Tensor, ridge: float) -> float:
    keep = keep.long()
    prune_mask = torch.ones(activation_matrix.shape[1], dtype=torch.bool, device=activation_matrix.device)
    prune_mask[keep] = False
    prune = torch.arange(activation_matrix.shape[1], device=activation_matrix.device)[prune_mask]
    retained = activation_matrix.index_select(1, keep).float()
    full_output = activation_matrix.float() @ down_proj.float().transpose(0, 1)
    retained_down = down_proj.float().index_select(1, keep)
    if prune.numel() == 0:
        candidate_output = retained @ retained_down.transpose(0, 1)
    else:
        coefficients = _ridge_coefficients(retained, activation_matrix.index_select(1, prune).float(), ridge, 1.0e-8)
        effective_down = retained_down + down_proj.float().index_select(1, prune) @ coefficients.transpose(0, 1)
        candidate_output = retained @ effective_down.transpose(0, 1)
    denominator = full_output.square().sum().clamp_min(1.0e-8)
    return float(((full_output - candidate_output).square().sum() / denominator).item())


def recoverability_swap_refinement(
    activation_matrix: torch.Tensor,
    down_proj: torch.Tensor,
    initial_keep: torch.Tensor,
    *,
    ridge: float = 1.0e-3,
    band: int = 32,
    max_swaps: int | None = None,
) -> torch.Tensor:
    """Improve an energy-ranked keep set with local recoverability-aware swaps."""

    if activation_matrix.ndim != 2 or down_proj.ndim != 2:
        raise ValueError("activation_matrix and down_proj must be matrices")
    channels = activation_matrix.shape[1]
    if down_proj.shape[1] != channels:
        raise ValueError("down_proj columns must match activation channels")
    keep = torch.unique(initial_keep.long(), sorted=False)
    if keep.numel() == 0 or keep.numel() >= channels:
        raise ValueError("initial_keep must be non-empty and pruned")
    if band <= 0:
        raise ValueError("band must be positive")
    if max_swaps is None:
        max_swaps = min(32, max(1, int(round(0.05 * keep.numel()))))
    max_swaps = max(0, int(max_swaps))
    all_channels = torch.arange(channels, device=activation_matrix.device)
    baseline_scores = output_energy_scores(activation_matrix, down_proj)
    current = _reconstruction_error(activation_matrix, down_proj, keep, ridge)
    for _ in range(max_swaps):
        keep_mask = torch.zeros(channels, dtype=torch.bool, device=keep.device)
        keep_mask[keep] = True
        pruned = all_channels[~keep_mask]
        if pruned.numel() == 0:
            break
        retained = activation_matrix.index_select(1, keep).float()
        pruned_matrix = activation_matrix.index_select(1, pruned).float()
        coefficients = _ridge_coefficients(retained, pruned_matrix, ridge, 1.0e-8)
        residual = pruned_matrix - retained @ coefficients
        residual_energy = residual.square().sum(dim=0)
        output_energy = down_proj.float().index_select(1, pruned).square().sum(dim=0)
        irrecoverability = residual_energy * output_energy
        candidate_band = min(int(band), 8)
        irrecoverable_order = torch.argsort(irrecoverability, descending=True, stable=True)[:candidate_band]
        prune_candidates = pruned.index_select(0, irrecoverable_order)
        keep_order = torch.argsort(baseline_scores.index_select(0, keep), descending=False, stable=True)[:candidate_band]
        keep_candidates = keep.index_select(0, keep_order)
        best_pair: tuple[int, int] | None = None
        best_value = current
        for prune_channel in prune_candidates.tolist():
            for keep_channel in keep_candidates.tolist():
                candidate = keep[keep != keep_channel]
                candidate = torch.cat((candidate, torch.tensor([prune_channel], device=keep.device)))
                value = _reconstruction_error(activation_matrix, down_proj, candidate, ridge)
                if value < best_value:
                    best_pair = (keep_channel, prune_channel)
                    best_value = value
        if best_pair is None:
            break
        keep = keep[keep != best_pair[0]]
        keep = torch.cat((keep, torch.tensor([best_pair[1]], device=keep.device)))
        current = best_value
    return torch.sort(keep).values


def fit_ridge_down(
    activation_train: torch.Tensor,
    activation_valid: torch.Tensor,
    down_proj: torch.Tensor,
    keep: torch.Tensor,
    *,
    ridge_grid: tuple[float, ...] = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0),
    trust_ratio_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
    epsilon: float = 1.0e-8,
) -> CompensationResult:
    """Fit and held-out validate the closed-form down-projection residual fold."""

    if activation_train.ndim != 2 or activation_valid.ndim != 2 or down_proj.ndim != 2:
        raise ValueError("activation matrices and down_proj must be matrices")
    if activation_train.shape[1] != down_proj.shape[1] or activation_valid.shape[1] != down_proj.shape[1]:
        raise ValueError("activation channel counts must match down_proj")
    keep = torch.sort(torch.unique(keep.long())).values
    if keep.numel() == 0 or keep.numel() >= down_proj.shape[1]:
        raise ValueError("keep must be non-empty and pruned")
    prune_mask = torch.ones(down_proj.shape[1], dtype=torch.bool, device=down_proj.device)
    prune_mask[keep] = False
    prune = torch.arange(down_proj.shape[1], device=down_proj.device)[prune_mask]
    retained_train = activation_train.float().index_select(1, keep)
    retained_valid = activation_valid.float().index_select(1, keep)
    target_valid = activation_valid.float() @ down_proj.float().transpose(0, 1)
    original_down = down_proj.float().index_select(1, keep)
    baseline_valid = retained_valid @ original_down.transpose(0, 1)
    baseline_error = float(((target_valid - baseline_valid).square().mean()).item())
    best: CompensationResult | None = None
    if prune.numel() == 0 or activation_valid.shape[0] == 0:
        return CompensationResult(original_down, False, None, None, baseline_error, baseline_error, 0.0)
    for ridge in ridge_grid:
        coefficients = _ridge_coefficients(
            retained_train,
            activation_train.float().index_select(1, prune),
            float(ridge),
            epsilon,
        )
        delta = down_proj.float().index_select(1, prune) @ coefficients.transpose(0, 1)
        for trust_ratio in trust_ratio_grid:
            ratio = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(original_down).clamp_min(epsilon))
            scale = min(1.0, float(trust_ratio) / max(ratio, epsilon))
            candidate_down = original_down + scale * delta
            candidate_output = retained_valid @ candidate_down.transpose(0, 1)
            error = float(((target_valid - candidate_output).square().mean()).item())
            candidate = CompensationResult(
                down=candidate_down,
                accepted=error < baseline_error,
                ridge=float(ridge),
                trust_ratio=float(trust_ratio),
                baseline_error=baseline_error,
                compensated_error=error,
                update_ratio=ratio * scale,
            )
            if best is None or candidate.compensated_error < best.compensated_error:
                best = candidate
    if best is None or not best.accepted:
        return CompensationResult(original_down, False, None, None, baseline_error, baseline_error, 0.0)
    return best


def slice_packed_experts(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    keep: torch.Tensor,
    compensated_down: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice packed expert tensors and optionally replace retained down columns."""

    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise ValueError("Packed expert tensors must be rank three")
    experts, packed_width, hidden = gate_up_proj.shape
    if down_proj.shape[0] != experts or down_proj.shape[1] != hidden or packed_width % 2:
        raise ValueError("Packed expert tensor shapes are inconsistent")
    channels = packed_width // 2
    keep = keep.long()
    if keep.ndim == 1:
        keep = keep.unsqueeze(0).expand(experts, -1)
    if keep.shape[0] != experts:
        raise ValueError("keep must have one row per expert")
    selected_gate_up = []
    selected_down = []
    for expert in range(experts):
        indices = keep[expert]
        if indices.numel() == 0 or bool((indices < 0).any()) or bool((indices >= channels).any()):
            raise ValueError("keep contains an invalid channel")
        packed_indices = torch.cat((indices, indices + channels))
        selected_gate_up.append(gate_up_proj[expert].index_select(0, packed_indices))
        selected_down.append(down_proj[expert].index_select(1, indices))
    output_down = torch.stack(selected_down)
    if compensated_down is not None:
        if compensated_down.shape != output_down.shape:
            raise ValueError("compensated_down shape must match sliced down projection")
        output_down = compensated_down.to(dtype=output_down.dtype, device=output_down.device)
    return torch.stack(selected_gate_up), output_down
