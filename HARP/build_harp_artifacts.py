"""Build HARP Layer-SP/Expert-SP/Channel-SP pruning profiles."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from CSP.build_csp_artifacts_impl import build_layer_rankings, load_weight_map
from CSP.csp_core import file_sha256
from CSP.model_adapter import CSPModelAdapter
from HARP.harp_core import allocate_expert_widths, allocate_layer_upgrade_units, detect_anchor_layer
from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload


def load_named(model_path: Path, weight_map: dict[str, str], names: list[str]) -> dict[str, torch.Tensor]:
    """Load named tensors from safetensors shards."""

    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(weight_map[name], []).append(name)
    result: dict[str, torch.Tensor] = {}
    for shard, shard_names in grouped.items():
        with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
            for name in shard_names:
                result[name] = handle.get_tensor(name)
    return result


def whole_layer_score(tensors: dict[str, torch.Tensor]) -> float:
    """Compute Layer-SP over routed expert tensors only."""

    l1 = 0.0
    l2 = 0.0
    count = 0
    for tensor in tensors.values():
        value = tensor.detach().to(dtype=torch.float32)
        l1 += float(value.abs().sum().item())
        l2 += float(value.square().sum().item())
        count += tensor.numel()
    if l1 <= 0 or l2 <= 0:
        return float("-inf")
    return float(torch.log(torch.tensor(count * l2 / (l1 * l1), dtype=torch.float64)).item())


def build_profile(model_path: Path, output_cache: Path, output_profile: Path, budget_width: int, low_width: int, high_width: int) -> None:
    """Build and save a HARP profile."""

    weight_map = load_weight_map(model_path)
    adapter = CSPModelAdapter.from_checkpoint(model_path, weight_map)
    if not low_width < budget_width < high_width:
        raise ValueError("HARP requires low_width < budget_width < high_width.")
    for width in (low_width, budget_width, high_width):
        adapter.architecture.validate_width(width)
    layer_scores: list[float] = []
    layer_ids = adapter.architecture.moe_layer_ids()
    for layer_id in layer_ids:
        tensors = load_named(model_path, weight_map, adapter.routed_tensor_names(layer_id))
        layer_scores.append(whole_layer_score(tensors))
    channel = build_layer_rankings(model_path, adapter, weight_map, False, False)
    channel_path = output_cache.expanduser().resolve()
    channel_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "purpose": "harp_channel_ranking",
        "method": "harp",
        "model_path": str(model_path),
        "model_family": adapter.architecture.model_family,
        "model_provenance": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "block_size": adapter.architecture.channel_alignment,
        "score_mode": "per_expert_structural_participation_aimer_compatible_fp32",
        "architecture": adapter.metadata(),
        "table": channel,
        "layer_scores": layer_scores,
        "csp": {"data_free": True, "weight_only": True, "input_scale_mode": "none", "canonicalization": False},
        "harp": {"layer_score": "log(N_layer * ||Theta_layer||_2^2 / ||Theta_layer||_1^2)", "input_scale_mode": "none", "canonicalization": False},
    }
    torch.save(payload, channel_path)
    block = adapter.architecture.channel_alignment
    low_blocks, budget_blocks, high_blocks = (width // block for width in (low_width, budget_width, high_width))
    experts = adapter.architecture.num_experts
    layers = len(layer_ids)
    total_units = layers * experts * (budget_blocks - low_blocks)
    anchor = detect_anchor_layer(torch.tensor(layer_scores, dtype=torch.float64))
    layer_units = allocate_layer_upgrade_units(
        torch.tensor(layer_scores, dtype=torch.float64),
        total_units=total_units,
        max_units_per_layer=2 * experts,
        anchor_layer=anchor,
        anchor_min_units=experts if anchor is not None else 0,
    )
    widths = []
    expert_widths = []
    for row, layer_id in enumerate(layer_ids):
        table = channel[int(layer_id)]
        expert_scores = table["expert_structural_scores"]
        units = int(layer_units[row].item())
        layer_widths = allocate_expert_widths(expert_scores, low_blocks=low_blocks, target_units=units)
        widths.append(layer_widths)
        expert_widths.append(layer_widths.tolist())
    profile_widths = torch.stack(widths)
    target_blocks_by_layer = profile_widths.sum(dim=1).tolist()
    profile: dict[str, Any] = {
        "schema_version": 1, "method": "harp", "mode": "harp_layer_expert_channel_sp",
        "created_at": datetime.now(timezone.utc).isoformat(), "model_path": str(model_path), "model_family": adapter.architecture.model_family,
        "profile_construction": "calibration_free", "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True, "test_metrics_used_for_profile": False,
        "layer_ids": list(layer_ids), "num_layers": layers, "num_experts": experts,
        "num_blocks": adapter.architecture.intermediate_size // block, "channel_block_size": block,
        "intermediate_size": adapter.architecture.intermediate_size,
        "allocation_scope": "per_layer_expert_harp_layer_expert_channel_sp",
        "allocation_objective": "layer_sp_budget_then_expert_sp_upgrade_then_channel_sp_prefix",
        "target_blocks_by_layer": target_blocks_by_layer, "actual_blocks_by_layer": target_blocks_by_layer,
        "total_blocks": int(profile_widths.sum().item()), "maximum_blocks": layers * experts * (adapter.architecture.intermediate_size // block),
        "target_pruning_ratio": 1.0 - budget_width / adapter.architecture.intermediate_size,
        "actual_structural_pruning_ratio": 1.0 - budget_width / adapter.architecture.intermediate_size,
        "retained_channels": None, "budget_reference_width": budget_width,
        "width_options": [low_width, budget_width, high_width], "padded_intermediate_size": high_width,
        "retained_expert_mask": None, "profile_widths": profile_widths,
        "profile_sha256": hashlib.sha256(profile_widths.numpy().tobytes(order="C")).hexdigest(),
        "csp": {"data_free": True, "weight_only": True, "accumulator_dtype": "float32", "input_scale_mode": "none", "canonicalization": False, "architecture": adapter.metadata(),
                "harp": {"layer_scores": layer_scores, "layer_order_descending": sorted(layer_ids, key=lambda i: (-layer_scores[i], i)), "anchor_layer": anchor, "low_width": low_width, "budget_width": budget_width, "high_width": high_width, "expert_widths_by_layer": expert_widths}},
        "cache_provenance": {"channel": {"path": str(channel_path), "sha256": file_sha256(channel_path), "role": "harp_ranking"}},
    }
    validate_static_profile_payload(profile)
    output_profile.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, output_profile.expanduser().resolve())
    print(channel_path)
    print(output_profile.expanduser().resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--budget-width", type=int, required=True)
    parser.add_argument("--low-width", type=int, required=True)
    parser.add_argument("--high-width", type=int, required=True)
    args = parser.parse_args()
    build_profile(args.model_path.expanduser().resolve(), args.output_channel_cache, args.output_profile, args.budget_width, args.low_width, args.high_width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
