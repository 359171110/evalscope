from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.channel_runtime import _build_layer_channel_table_from_raw_scores, channel_table_to_payload
from WICK.build_wick_profile import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed-width backbone ranking with Pure-Pseudo protection.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backbone-cache", type=Path, required=True)
    parser.add_argument("--pseudo-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--retained-blocks", type=int, required=True)
    parser.add_argument("--protection-ratio", type=float, required=True)
    return parser.parse_args()


def cache_orders(payload: dict) -> torch.Tensor:
    table = payload.get("table")
    if not isinstance(table, dict) or not table:
        raise ValueError("ranking cache must contain a non-empty table.")
    layer_ids = sorted(int(layer_id) for layer_id in table)
    return torch.stack([table[layer_id]["ranked_indices"].to(torch.long) for layer_id in layer_ids])


def build_protected_orders(
    backbone_order: torch.Tensor,
    pseudo_order: torch.Tensor,
    *,
    protected_channels: int,
) -> torch.Tensor:
    if backbone_order.shape != pseudo_order.shape or backbone_order.ndim != 3:
        raise ValueError("backbone and pseudo orders must have aligned [layers, experts, channels] shapes.")
    channel_count = int(backbone_order.shape[-1])
    protected_count = int(protected_channels)
    if not 0 <= protected_count <= channel_count:
        raise ValueError("protected_channels must be in [0, channel_count].")

    combined = torch.empty_like(backbone_order)
    for layer_id in range(int(backbone_order.shape[0])):
        for expert_id in range(int(backbone_order.shape[1])):
            protected = pseudo_order[layer_id, expert_id, :protected_count]
            protected_mask = torch.zeros(channel_count, dtype=torch.bool)
            protected_mask[protected] = True
            backbone = backbone_order[layer_id, expert_id]
            combined[layer_id, expert_id] = torch.cat((protected, backbone[~protected_mask[backbone]]))
    return combined


def orders_to_scores(orders: torch.Tensor) -> dict[int, torch.Tensor]:
    channel_count = int(orders.shape[-1])
    descending = torch.arange(channel_count, 0, -1, dtype=torch.float32)
    scores_by_layer = {}
    for layer_id in range(int(orders.shape[0])):
        scores = torch.empty_like(orders[layer_id], dtype=torch.float32)
        scores.scatter_(1, orders[layer_id], descending.expand_as(scores))
        scores_by_layer[layer_id] = scores
    return scores_by_layer


def build_protected_artifacts(
    *,
    model_path: Path,
    orders: torch.Tensor,
    method: str,
    backbone: str,
    retained_blocks: int,
    protection_ratio: float,
    block_size: int,
    backbone_cache_sha256: str,
    pseudo_cache_sha256: str,
) -> tuple[dict, dict]:
    num_layers, num_experts, intermediate_size = map(int, orders.shape)
    if intermediate_size % int(block_size):
        raise ValueError("intermediate size must be divisible by block_size.")
    num_blocks = intermediate_size // int(block_size)
    retained = int(retained_blocks)
    if not 1 <= retained < num_blocks:
        raise ValueError("retained_blocks must be in [1, num_blocks).")
    ratio = float(protection_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("protection_ratio must be in [0, 1].")
    protected_channels = int(round(intermediate_size * ratio))
    retained_channels = retained * int(block_size)
    if protected_channels > retained_channels:
        raise ValueError("protected channels cannot exceed retained channels.")

    tables = {
        layer_id: _build_layer_channel_table_from_raw_scores(scores, int(block_size))
        for layer_id, scores in orders_to_scores(orders).items()
    }
    widths = torch.full((num_layers, num_experts), retained, dtype=torch.long)
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    metadata = {
        "backbone": backbone,
        "protection_source": "pure_pseudo_ranking_prefix",
        "protection_ratio": ratio,
        "protected_channels": protected_channels,
        "backbone_cache_sha256": backbone_cache_sha256,
        "pseudo_cache_sha256": pseudo_cache_sha256,
    }
    channel = {
        "schema_version": 1,
        "purpose": f"{method}_channel_ranking",
        "model_path": str(model_path),
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": int(block_size),
        "table": channel_table_to_payload(tables),
        "pseudo_protection": metadata,
    }
    profile = {
        "schema_version": 1,
        "method": method,
        "mode": f"{backbone}_fixed_width_with_pure_pseudo_protection",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": list(range(num_layers)),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "num_blocks": num_blocks,
        "channel_block_size": int(block_size),
        "intermediate_size": intermediate_size,
        "allocation_scope": "per_expert_fixed",
        "target_blocks_by_layer": widths.sum(dim=1).tolist(),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "total_blocks": total_blocks,
        "maximum_blocks": maximum_blocks,
        "target_pruning_ratio": 1.0 - retained / num_blocks,
        "actual_structural_pruning_ratio": 1.0 - total_blocks / maximum_blocks,
        "retained_expert_mask": None,
        "profile_widths": widths,
        "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
        "pseudo_protection": metadata,
    }
    return channel, profile


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    backbone_path = args.backbone_cache.expanduser().resolve()
    pseudo_path = args.pseudo_cache.expanduser().resolve()
    backbone_payload = torch.load(backbone_path, map_location="cpu", weights_only=True)
    pseudo_payload = torch.load(pseudo_path, map_location="cpu", weights_only=True)
    backbone_order = cache_orders(backbone_payload)
    pseudo_order = cache_orders(pseudo_payload)
    protected_channels = int(round(int(backbone_order.shape[-1]) * float(args.protection_ratio)))
    orders = build_protected_orders(backbone_order, pseudo_order, protected_channels=protected_channels)
    block_size = int(backbone_payload["block_size"])
    channel, profile = build_protected_artifacts(
        model_path=model_path,
        orders=orders,
        method=args.method,
        backbone=args.backbone,
        retained_blocks=int(args.retained_blocks),
        protection_ratio=float(args.protection_ratio),
        block_size=block_size,
        backbone_cache_sha256=file_sha256(backbone_path),
        pseudo_cache_sha256=file_sha256(pseudo_path),
    )

    args.output_channel_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel, args.output_channel_cache)
    profile["cache_provenance"] = {
        "channel": {"sha256": file_sha256(args.output_channel_cache), "role": args.method}
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output_profile)
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["width_histogram"] = {
        str(int(width)): int(count) for width, count in zip(*torch.unique(profile["profile_widths"], return_counts=True))
    }
    args.output_profile.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output_channel_cache.resolve())
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())