from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from Random.model_adapter import RandomModelAdapter
from Random.random_core import (
    build_layer_orders,
    file_sha256,
    permutation_table,
    validate_rankings,
)
from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload


def load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload["weight_map"].items()}


def build_channel_payload(
    *,
    model_path: Path,
    adapter: RandomModelAdapter,
    orders: dict[int, torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    architecture = adapter.architecture
    moe_ids = architecture.moe_layer_ids()
    tables = {layer_id: permutation_table(orders[layer_id], architecture.channel_alignment) for layer_id in moe_ids}
    validate_rankings(
        tables,
        len(moe_ids),
        architecture.num_experts,
        architecture.intermediate_size,
        layer_ids=moe_ids,
    )
    return {
        "schema_version": 1,
        "purpose": "random_channel_ranking",
        "method": "random",
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "architecture": adapter.metadata(),
        "model_provenance": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": architecture.channel_alignment,
        "score_mode": "independent_per_expert_seeded_permutation",
        "table": tables,
        "random": {
            "seed": int(seed),
            "rng_scheme": "sha256(seed:layer:expert) -> torch.Generator.manual_seed",
            "data_free": True,
        },
    }


def build_profile(
    *,
    model_path: Path,
    adapter: RandomModelAdapter,
    retained_channels: int,
    target_pruning_ratio: float,
    seed: int,
) -> dict[str, Any]:
    architecture = adapter.architecture
    moe_ids = architecture.moe_layer_ids()
    block_size = architecture.channel_alignment
    num_blocks = architecture.intermediate_size // block_size
    retained_blocks = retained_channels // block_size
    widths = torch.full((len(moe_ids), architecture.num_experts), retained_blocks, dtype=torch.long)
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    actual_ratio = 1.0 - retained_channels / architecture.intermediate_size
    profile = {
        "schema_version": 1,
        "method": "random",
        "mode": "per_expert_fixed_seeded_random",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": list(moe_ids),
        "num_layers": len(moe_ids),
        "num_experts": architecture.num_experts,
        "num_blocks": num_blocks,
        "channel_block_size": block_size,
        "intermediate_size": architecture.intermediate_size,
        "allocation_scope": "per_expert_fixed",
        "target_blocks_by_layer": widths.sum(dim=1).tolist(),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "total_blocks": total_blocks,
        "maximum_blocks": maximum_blocks,
        "target_pruning_ratio": float(target_pruning_ratio),
        "actual_structural_pruning_ratio": actual_ratio,
        "retained_channels": retained_channels,
        "retained_expert_mask": None,
        "profile_widths": widths,
        "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
        "random": {
            "seed": int(seed),
            "rng_scheme": "sha256(seed:layer:expert) -> torch.Generator.manual_seed",
            "data_free": True,
            "architecture": adapter.metadata(),
        },
    }
    validate_static_profile_payload(profile)
    return profile


def write_profile(profile: dict[str, Any], profile_path: Path) -> None:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, profile_path)
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["profile_file_sha256"] = file_sha256(profile_path)
    profile_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_retained_indices(channel: dict[str, Any], retained_channels: int, output_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "method": "random",
        "seed": int(channel["random"]["seed"]),
        "retained_channels": int(retained_channels),
        "layer_ids": sorted(int(layer_id) for layer_id in channel["table"]),
        "retained_indices": {
            str(int(layer_id)): table["ranked_indices"][:, :int(retained_channels)].tolist()
            for layer_id, table in channel["table"].items()
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_existing_cache(
    channel: dict[str, Any],
    *,
    model_path: Path,
    adapter: RandomModelAdapter,
    seed: int,
) -> None:
    if int(channel.get("schema_version", -1)) != 1 or channel.get("purpose") != "random_channel_ranking":
        raise ValueError("Unsupported Random channel ranking payload.")
    if Path(str(channel.get("model_path", ""))).resolve() != model_path:
        raise ValueError("Random rankings were built for a different model path.")
    if channel.get("model_family") != adapter.architecture.model_family:
        raise ValueError("Random rankings model family does not match the checkpoint.")
    if int(channel.get("random", {}).get("seed", -1)) != int(seed):
        raise ValueError("Random rankings seed does not match --seed.")
    provenance = channel.get("model_provenance", {})
    if provenance.get("config_sha256") != file_sha256(model_path / "config.json"):
        raise ValueError("Checkpoint config changed after Random ranking construction.")
    if provenance.get("weight_index_sha256") != file_sha256(model_path / "model.safetensors.index.json"):
        raise ValueError("Checkpoint weight index changed after Random ranking construction.")
    architecture = adapter.architecture
    validate_rankings(
        channel["table"],
        len(architecture.moe_layer_ids()),
        architecture.num_experts,
        architecture.intermediate_size,
        layer_ids=architecture.moe_layer_ids(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data-free random channel rankings and a static profile.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float)
    parser.add_argument("--retained-channels", type=int)
    parser.add_argument("--rounding", choices=("floor", "nearest", "ceil"), default="nearest")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.retained_channels is None) == (args.target_pruning_ratio is None):
        raise ValueError("Provide exactly one of --retained-channels or --target-pruning-ratio.")
    model_path = args.model_path.expanduser().resolve()
    channel_path = args.output_channel_cache.expanduser().resolve()
    profile_path = args.output_profile.expanduser().resolve()
    weight_map = load_weight_map(model_path)
    adapter = RandomModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.architecture
    if args.retained_channels is not None:
        retained_channels = int(args.retained_channels)
        target_ratio = 1.0 - retained_channels / architecture.intermediate_size
    else:
        target_ratio = float(args.target_pruning_ratio)
        retained_channels = architecture.width_for_pruning(target_ratio, args.rounding)
    architecture.validate_width(retained_channels)

    if channel_path.exists():
        channel = torch.load(channel_path, map_location="cpu", weights_only=True)
        validate_existing_cache(channel, model_path=model_path, adapter=adapter, seed=args.seed)
    else:
        orders = build_layer_orders(
            architecture.moe_layer_ids(),
            architecture.num_experts,
            architecture.intermediate_size,
            seed=args.seed,
        )
        channel = build_channel_payload(model_path=model_path, adapter=adapter, orders=orders, seed=args.seed)
        channel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(channel, channel_path)

    profile = build_profile(
        model_path=model_path,
        adapter=adapter,
        retained_channels=retained_channels,
        target_pruning_ratio=target_ratio,
        seed=args.seed,
    )
    profile["cache_provenance"] = {
        "channel": {
            "path": str(channel_path),
            "sha256": file_sha256(channel_path),
            "role": "random_ranking",
        }
    }
    write_profile(profile, profile_path)
    write_retained_indices(
        channel,
        retained_channels,
        profile_path.with_name(f"random_retained_{retained_channels}ch.json"),
    )
    if abs(profile["actual_structural_pruning_ratio"] - target_ratio) > 1.0e-12:
        print(
            f"WARNING: requested pruning ratio {target_ratio:.8f} was aligned to "
            f"{profile['actual_structural_pruning_ratio']:.8f} ({retained_channels} channels).",
            flush=True,
        )
    print(channel_path)
    print(profile_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
