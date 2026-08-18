from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.channel_runtime import _build_layer_channel_table_from_raw_scores, channel_table_to_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build matched random and random+pseudo-protection profiles.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pseudo-ranking-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float, default=0.50)
    parser.add_argument("--protection-ratio", type=float, default=0.10)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_random_orders(
    pseudo_order: torch.Tensor,
    *,
    retained_channels: int,
    protected_channels: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pseudo_order.ndim != 3:
        raise ValueError("pseudo_order must have shape [layers, experts, channels].")
    channel_count = int(pseudo_order.shape[-1])
    if not 0 <= protected_channels <= retained_channels <= channel_count:
        raise ValueError("channel counts must satisfy 0 <= protected <= retained <= total.")

    random_orders = torch.empty_like(pseudo_order)
    protected_orders = torch.empty_like(pseudo_order)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for layer_id in range(int(pseudo_order.shape[0])):
        for expert_id in range(int(pseudo_order.shape[1])):
            random_order = torch.randperm(channel_count, generator=generator)
            random_orders[layer_id, expert_id] = random_order

            protected = pseudo_order[layer_id, expert_id, :protected_channels]
            protected_mask = torch.zeros(channel_count, dtype=torch.bool)
            protected_mask[protected] = True
            random_remainder = random_order[~protected_mask[random_order]]
            protected_orders[layer_id, expert_id] = torch.cat((protected, random_remainder))
    return random_orders, protected_orders


def orders_to_scores(orders: torch.Tensor) -> dict[int, torch.Tensor]:
    channel_count = int(orders.shape[-1])
    scores_by_layer = {}
    descending = torch.arange(channel_count, 0, -1, dtype=torch.float32)
    for layer_id in range(int(orders.shape[0])):
        scores = torch.empty_like(orders[layer_id], dtype=torch.float32)
        scores.scatter_(1, orders[layer_id], descending.expand_as(scores))
        scores_by_layer[layer_id] = scores
    return scores_by_layer


def build_artifacts(
    *,
    model_path: Path,
    orders: torch.Tensor,
    method: str,
    mode: str,
    retained_channels: int,
    block_size: int,
    seed: int,
    pseudo_cache_sha256: str | None,
    protected_channels: int,
) -> tuple[dict, dict]:
    scores_by_layer = orders_to_scores(orders)
    tables = {
        layer_id: _build_layer_channel_table_from_raw_scores(scores, block_size)
        for layer_id, scores in scores_by_layer.items()
    }
    num_layers, num_experts, intermediate_size = map(int, orders.shape)
    retained_blocks = retained_channels // block_size
    num_blocks = intermediate_size // block_size
    widths = torch.full((num_layers, num_experts), retained_blocks, dtype=torch.long)
    channel_payload = {
        "schema_version": 1,
        "purpose": f"{method}_channel_ranking",
        "model_path": str(model_path),
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": block_size,
        "table": channel_table_to_payload(tables),
        "random": {
            "seed": seed,
            "protected_channels": protected_channels,
            "pseudo_ranking_cache_sha256": pseudo_cache_sha256,
        },
    }
    total_blocks = int(widths.sum())
    maximum_blocks = int(widths.numel() * num_blocks)
    profile = {
        "schema_version": 1,
        "method": method,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "profile_construction": "calibration_free_seeded_random",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": list(range(num_layers)),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "num_blocks": num_blocks,
        "channel_block_size": block_size,
        "intermediate_size": intermediate_size,
        "allocation_scope": "per_expert_fixed",
        "target_blocks_by_layer": widths.sum(dim=1).tolist(),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "total_blocks": total_blocks,
        "maximum_blocks": maximum_blocks,
        "target_pruning_ratio": 1.0 - retained_channels / intermediate_size,
        "actual_structural_pruning_ratio": 1.0 - total_blocks / maximum_blocks,
        "retained_expert_mask": None,
        "profile_widths": widths,
        "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
        "random": channel_payload["random"],
    }
    return channel_payload, profile


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    pseudo_cache_path = args.pseudo_ranking_cache.expanduser().resolve()
    pseudo_payload = torch.load(pseudo_cache_path, map_location="cpu", weights_only=True)
    layer_ids = sorted(int(layer_id) for layer_id in pseudo_payload["table"])
    pseudo_order = torch.stack([pseudo_payload["table"][layer_id]["ranked_indices"] for layer_id in layer_ids])
    intermediate_size = int(pseudo_order.shape[-1])
    retained_channels = int(round(intermediate_size * (1.0 - args.target_pruning_ratio)))
    protected_channels = int(round(intermediate_size * args.protection_ratio))
    if retained_channels % args.channel_block_size:
        raise ValueError("retained channel count must be block aligned.")
    random_orders, protected_orders = build_random_orders(
        pseudo_order,
        retained_channels=retained_channels,
        protected_channels=protected_channels,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pseudo_sha = file_sha256(pseudo_cache_path)
    variants = (
        ("random", "seeded_random_fixed_width", random_orders, None, 0),
        (
            "random_wick_protect",
            "seeded_random_fixed_width_with_wick_pseudo_protection",
            protected_orders,
            pseudo_sha,
            protected_channels,
        ),
    )
    for method, mode, orders, source_sha, protected_count in variants:
        channel, profile = build_artifacts(
            model_path=model_path,
            orders=orders,
            method=method,
            mode=mode,
            retained_channels=retained_channels,
            block_size=args.channel_block_size,
            seed=args.seed,
            pseudo_cache_sha256=source_sha,
            protected_channels=protected_count,
        )
        channel_path = args.output_dir / f"{method}_rankings.pt"
        profile_path = args.output_dir / f"{method}_50pct_per_expert.pt"
        torch.save(channel, channel_path)
        profile["cache_provenance"] = {"channel": {"sha256": file_sha256(channel_path), "role": method}}
        torch.save(profile, profile_path)
        print(channel_path.resolve())
        print(profile_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())