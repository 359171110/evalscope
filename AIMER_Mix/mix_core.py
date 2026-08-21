from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import torch

from AIMER_Channel.aimer_channel_core import coupled_channel_aimer_importance

DEFAULT_EPS = 1.0e-8
DEFAULT_EFFECTIVE_ZERO_THRESHOLD = 1.0e-12
EnergyMode = Literal["geom", "l2"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _channel_path_norms(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel L2 of gate rows, up rows, and down columns."""

    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise ValueError("gate, up, and down must be rank-2 expert tensors.")
    if gate.shape != up.shape:
        raise ValueError("gate and up must share the same shape.")
    if down.shape[0] != gate.shape[1] or down.shape[1] != gate.shape[0]:
        raise ValueError("down must be the transpose layout of gate/up channel rows.")
    gate_n = gate.detach().to(dtype=torch.float32).norm(dim=1)
    up_n = up.detach().to(dtype=torch.float32).norm(dim=1)
    down_n = down.detach().to(dtype=torch.float32).norm(dim=0)
    return gate_n, up_n, down_n


def path_mean_energies(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> tuple[float, float, float]:
    """Expert-level mean path energy ``(Ē_g, Ē_u, Ē_d)``."""

    gate_n, up_n, down_n = _channel_path_norms(gate, up, down)
    return float(gate_n.mean().item()), float(up_n.mean().item()), float(down_n.mean().item())


def energy_balance_alpha(
    mean_gate: float,
    mean_up: float,
    mean_down: float,
    eps: float = DEFAULT_EPS,
) -> float:
    """α = min(Ē_g, Ē_u, Ē_d) / max(Ē_g, Ē_u, Ē_d). Balanced experts stay near 1."""

    if float(eps) <= 0:
        raise ValueError("eps must be positive.")
    values = (float(mean_gate), float(mean_up), float(mean_down))
    peak = max(values)
    if peak < float(eps):
        return 1.0
    return min(values) / peak


def geom_channel_energy(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Geometric mean of the three projection L2 norms: ``(||g|| ||u|| ||d||)^{1/3}``."""

    gate_n, up_n, down_n = _channel_path_norms(gate, up, down)
    return (gate_n * up_n * down_n).clamp_min(0.0).pow(1.0 / 3.0)


def l2_channel_energy(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Coupled L2 used by the Magnitude baseline."""

    gate_n, up_n, down_n = _channel_path_norms(gate, up, down)
    return torch.sqrt(gate_n.square() + up_n.square() + down_n.square())


def descending_unit_ranks(scores: torch.Tensor) -> torch.Tensor:
    """Map scores to ``[0, 1]`` with 1 = most important. Average ranks for ties.

    Non-finite values (AIMER near-zero ``-inf``) receive rank 0.
    """

    if scores.ndim == 1:
        return _descending_unit_ranks_1d(scores)
    if scores.ndim == 2:
        return torch.stack([_descending_unit_ranks_1d(row) for row in scores])
    raise ValueError("scores must be [C] or [experts, C].")


def _descending_unit_ranks_1d(scores: torch.Tensor) -> torch.Tensor:
    n = int(scores.numel())
    out = torch.zeros(n, dtype=torch.float32, device=scores.device)
    finite = torch.isfinite(scores)
    count = int(finite.sum().item())
    if count == 0:
        return out
    idx = torch.nonzero(finite, as_tuple=False).flatten()
    vals = scores.index_select(0, idx).to(dtype=torch.float32)
    order = torch.argsort(vals, descending=True, stable=True)
    sorted_vals = vals.index_select(0, order)
    ordinal = torch.empty(count, dtype=torch.float32, device=scores.device)
    start = 0
    while start < count:
        end = start + 1
        while end < count and bool(torch.isclose(sorted_vals[end], sorted_vals[start], rtol=0.0, atol=0.0)):
            end += 1
        mean_rank = 0.5 * float(start + end - 1)
        ordinal[start:end] = mean_rank
        start = end
    placed = torch.empty(count, dtype=torch.float32, device=scores.device)
    placed[order] = ordinal
    denom = float(max(count - 1, 1))
    out[idx] = 1.0 - placed / denom
    return out


def mix_channel_importance(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    energy_mode: EnergyMode = "geom",
    eps: float = DEFAULT_EPS,
    effective_zero_threshold: float = DEFAULT_EFFECTIVE_ZERO_THRESHOLD,
) -> tuple[torch.Tensor, float]:
    """Return ``(mix scores, α)`` for one expert. Does not switch on model name."""

    if energy_mode not in {"geom", "l2"}:
        raise ValueError("energy_mode must be 'geom' or 'l2'.")
    aimer = coupled_channel_aimer_importance(
        gate,
        up,
        down,
        eps=eps,
        effective_zero_threshold=effective_zero_threshold,
    )
    energy = geom_channel_energy(gate, up, down) if energy_mode == "geom" else l2_channel_energy(gate, up, down)
    mean_gate, mean_up, mean_down = path_mean_energies(gate, up, down)
    alpha = energy_balance_alpha(mean_gate, mean_up, mean_down, eps=eps)
    mix = float(alpha) * descending_unit_ranks(aimer) + (1.0 - float(alpha)) * descending_unit_ranks(energy)
    return mix, float(alpha)


def packed_mix_channel_importance(
    gate_up: torch.Tensor,
    down: torch.Tensor,
    *,
    energy_mode: EnergyMode = "geom",
    eps: float = DEFAULT_EPS,
    effective_zero_threshold: float = DEFAULT_EFFECTIVE_ZERO_THRESHOLD,
) -> tuple[torch.Tensor, float]:
    if gate_up.ndim != 2 or down.ndim != 2:
        raise ValueError("packed gate_up and down must be rank-2 expert tensors.")
    width = int(down.shape[1])
    if gate_up.shape[0] != 2 * width or gate_up.shape[1] != down.shape[0]:
        raise ValueError("packed gate_up must have shape [2C, H] matching down [H, C].")
    return mix_channel_importance(
        gate_up[:width],
        gate_up[width:],
        down,
        energy_mode=energy_mode,
        eps=eps,
        effective_zero_threshold=effective_zero_threshold,
    )


def rank_channels_by_mix(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim == 1:
        return torch.argsort(scores, dim=0, descending=True, stable=True)
    if scores.ndim == 2:
        return torch.argsort(scores, dim=1, descending=True, stable=True)
    raise ValueError("scores must be [C] or [experts, C].")


def ranking_table(
    scores: torch.Tensor,
    block_size: int,
    expert_alpha: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int | float]:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [experts, channels].")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive.")
    width = int(scores.shape[1])
    if width % int(block_size):
        raise ValueError("channel count must be divisible by block_size.")
    ranked = rank_channels_by_mix(scores)
    num_blocks = width // int(block_size)
    num_experts = int(scores.shape[0])
    finite_scores = torch.nan_to_num(scores.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ranked_scores = torch.gather(finite_scores, 1, ranked)
    block_scores = ranked_scores.reshape(num_experts, num_blocks, int(block_size)).mean(dim=2)
    coverage = block_scores / block_scores.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    payload: dict[str, torch.Tensor | int | float] = {
        "ranked_indices": ranked.long().cpu(),
        "channel_scores": scores.to(dtype=torch.float32).cpu(),
        "block_relative_scores": block_scores.cpu(),
        "block_coverage_scores": coverage.cpu(),
        "block_sizes": torch.full((num_blocks, ), int(block_size), dtype=torch.long),
        "intermediate_size": width,
    }
    if expert_alpha is not None:
        alphas = expert_alpha.to(dtype=torch.float32).cpu()
        if alphas.numel() != num_experts:
            raise ValueError("expert_alpha must have one value per expert.")
        payload["expert_alpha"] = alphas
        payload["mean_alpha"] = float(alphas.mean().item())
    return payload


def retained_prefix(order: torch.Tensor, retained_channels: int) -> torch.Tensor:
    if order.ndim != 1:
        raise ValueError("order must be a 1-D ranking.")
    if not 0 < int(retained_channels) <= int(order.numel()):
        raise ValueError("retained_channels must be in (0, channel_count].")
    return order[:int(retained_channels)].long()


def validate_rankings(
    table: dict[int, dict[str, object]],
    num_layers: int,
    num_experts: int,
    width: int,
    layer_ids: tuple[int, ...] | list[int] | None = None,
) -> None:
    expected_ids = list(range(int(num_layers)) if layer_ids is None else [int(layer_id) for layer_id in layer_ids])
    if set(map(int, table)) != set(expected_ids):
        raise ValueError("Ranking table does not cover every requested MoE layer.")
    expected = torch.arange(width)
    for layer_id in expected_ids:
        ranking = table[layer_id]["ranked_indices"]
        if not isinstance(ranking, torch.Tensor) or tuple(ranking.shape) != (num_experts, width):
            raise ValueError(f"Layer {layer_id} ranking has an invalid shape.")
        if not torch.equal(torch.sort(ranking.long(), dim=1).values, expected.expand(num_experts, -1)):
            raise ValueError(f"Layer {layer_id} ranking rows must be complete channel permutations.")
