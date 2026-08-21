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


def expert_rng_seed(seed: int, layer_id: int, expert_id: int) -> int:
    """Mix a global seed with layer/expert ids so call order cannot change the draw."""

    payload = f"{int(seed)}:{int(layer_id)}:{int(expert_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def random_channel_order(channel_count: int, *, seed: int, layer_id: int, expert_id: int) -> torch.Tensor:
    if channel_count <= 0:
        raise ValueError("channel_count must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(expert_rng_seed(seed, layer_id, expert_id))
    return torch.randperm(int(channel_count), generator=generator)


def build_layer_orders(
    layer_ids: tuple[int, ...] | list[int],
    num_experts: int,
    channel_count: int,
    *,
    seed: int,
) -> dict[int, torch.Tensor]:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    orders: dict[int, torch.Tensor] = {}
    for layer_id in layer_ids:
        rows = [
            random_channel_order(channel_count, seed=seed, layer_id=int(layer_id), expert_id=expert_id)
            for expert_id in range(int(num_experts))
        ]
        orders[int(layer_id)] = torch.stack(rows)
    return orders


def permutation_table(order: torch.Tensor, block_size: int) -> dict[str, torch.Tensor | int]:
    if order.ndim != 2:
        raise ValueError("order must have shape [experts, channels].")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive.")
    width = int(order.shape[1])
    if width % int(block_size):
        raise ValueError("channel count must be divisible by block_size.")
    num_blocks = width // int(block_size)
    num_experts = int(order.shape[0])
    return {
        "ranked_indices": order.long().cpu(),
        "block_relative_scores": torch.ones(num_experts, num_blocks, dtype=torch.float32),
        "block_coverage_scores": torch.full((num_experts, num_blocks), 1.0 / num_blocks, dtype=torch.float32),
        "block_sizes": torch.full((num_blocks, ), int(block_size), dtype=torch.long),
        "intermediate_size": width,
    }


def retained_prefix(order: torch.Tensor, retained_channels: int) -> torch.Tensor:
    if order.ndim != 1:
        raise ValueError("order must be a 1-D permutation.")
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
