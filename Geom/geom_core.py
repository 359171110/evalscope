from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coupled_channel_geom(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Geometric mean of gate/up/down channel L2 norms in FP32.

    ``gate`` / ``up`` are ``[C, H]``; ``down`` is ``[H, C]``. The result is ``[C]``:

    ``S(c) = (||g_c||_2 * ||u_c||_2 * ||d_c||_2)^{1/3}``.
    """

    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise ValueError("gate, up, and down must be rank-2 expert tensors.")
    if gate.shape != up.shape:
        raise ValueError("gate and up must share the same shape.")
    if down.shape[0] != gate.shape[1] or down.shape[1] != gate.shape[0]:
        raise ValueError("down must be the transpose layout of gate/up channel rows.")
    gate_n = gate.detach().to(dtype=torch.float32).norm(dim=1)
    up_n = up.detach().to(dtype=torch.float32).norm(dim=1)
    down_n = down.detach().to(dtype=torch.float32).norm(dim=0)
    return (gate_n * up_n * down_n).clamp_min(0.0).pow(1.0 / 3.0)


def packed_channel_geom(gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Geom score for a packed ``[2C, H]`` gate-up tensor and ``[H, C]`` down."""

    if gate_up.ndim != 2 or down.ndim != 2:
        raise ValueError("packed gate_up and down must be rank-2 expert tensors.")
    width = int(down.shape[1])
    if gate_up.shape[0] != 2 * width or gate_up.shape[1] != down.shape[0]:
        raise ValueError("packed gate_up must have shape [2C, H] matching down [H, C].")
    return coupled_channel_geom(gate_up[:width], gate_up[width:], down)


def rank_channels_by_geom(scores: torch.Tensor) -> torch.Tensor:
    """Return channel indices sorted by descending geom score.

    Ties keep the lower original index first (stable argsort).
    """

    if scores.ndim == 1:
        return torch.argsort(scores, dim=0, descending=True, stable=True)
    if scores.ndim == 2:
        return torch.argsort(scores, dim=1, descending=True, stable=True)
    raise ValueError("scores must be [C] or [experts, C].")


def ranking_table(scores: torch.Tensor, block_size: int) -> dict[str, torch.Tensor | int]:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [experts, channels].")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive.")
    width = int(scores.shape[1])
    if width % int(block_size):
        raise ValueError("channel count must be divisible by block_size.")
    ranked = rank_channels_by_geom(scores)
    num_blocks = width // int(block_size)
    num_experts = int(scores.shape[0])
    ranked_scores = torch.gather(scores.to(dtype=torch.float32), 1, ranked)
    block_scores = ranked_scores.reshape(num_experts, num_blocks, int(block_size)).mean(dim=2)
    coverage = block_scores / block_scores.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return {
        "ranked_indices": ranked.long().cpu(),
        "channel_scores": scores.to(dtype=torch.float32).cpu(),
        "block_relative_scores": block_scores.cpu(),
        "block_coverage_scores": coverage.cpu(),
        "block_sizes": torch.full((num_blocks, ), int(block_size), dtype=torch.long),
        "intermediate_size": width,
    }


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
