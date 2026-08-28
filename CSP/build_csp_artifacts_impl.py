from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from CSP.csp_core import (
    DEFAULT_FUNCTIONAL_VIABILITY_THRESHOLD,
    canonical_structural_score,
    file_sha256,
    canonical_structural_score_packed,
    ranking_table,
    validate_rankings,
)
from CSP.model_adapter import CSPModelAdapter
from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload


def load_weight_map(model_path: Path) -> dict[str, str]:
    """Load the safetensors name-to-shard map."""

    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload["weight_map"].items()}


def _load_named_tensors(model_path: Path, weight_map: dict[str, str], names: list[str]) -> dict[str, torch.Tensor]:
    shards: dict[str, list[str]] = {}
    for name in names:
        if name not in weight_map:
            raise KeyError(f"Missing CSP tensor {name}.")
        shards.setdefault(weight_map[name], []).append(name)
    loaded: dict[str, torch.Tensor] = {}
    for shard_name, shard_names in shards.items():
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in shard_names:
                loaded[name] = handle.get_tensor(name)
    return loaded


def score_separate_layer(
    adapter: CSPModelAdapter,
    tensors: dict[str, torch.Tensor],
    layer_id: int,
    apply_input_scale: bool,
    canonicalize: bool,
) -> torch.Tensor:
    """Score every separate-layout routed expert in one layer."""

    scale = tensors.get(adapter.input_scale_name(layer_id)) if apply_input_scale else None
    rows = []
    for expert_id in range(adapter.architecture.num_experts):
        rows.append(
            canonical_structural_score(
                tensors[adapter.gate_name(layer_id, expert_id)],
                tensors[adapter.up_name(layer_id, expert_id)],
                tensors[adapter.down_name(layer_id, expert_id)],
                input_scale=scale,
                functional_viability_threshold=DEFAULT_FUNCTIONAL_VIABILITY_THRESHOLD,
                canonicalize=canonicalize,
            )
        )
    return torch.stack(rows)


def score_packed_layer(
    adapter: CSPModelAdapter,
    tensors: dict[str, torch.Tensor],
    layer_id: int,
    apply_input_scale: bool,
    canonicalize: bool,
) -> torch.Tensor:
    """Score every packed routed expert in one layer."""

    gate_up = tensors[adapter.gate_up_name(layer_id)]
    down = tensors[adapter.down_name(layer_id)]
    if gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("Packed expert tensors must have a leading expert axis.")
    scale = tensors.get(adapter.input_scale_name(layer_id)) if apply_input_scale else None
    scores = torch.stack([
        canonical_structural_score_packed(
            gate_up[expert_id],
            down[expert_id],
            input_scale=scale,
            functional_viability_threshold=DEFAULT_FUNCTIONAL_VIABILITY_THRESHOLD,
            canonicalize=canonicalize,
        )
        for expert_id in range(adapter.architecture.num_experts)
    ])
    if scores.shape[1] != adapter.architecture.intermediate_size:
        raise ValueError("Packed CSP width does not match moe_intermediate_size.")
    return scores


def build_layer_rankings(
    model_path: Path,
    adapter: CSPModelAdapter,
    weight_map: dict[str, str],
    apply_input_scale: bool,
    canonicalize: bool,
) -> dict[int, dict[str, torch.Tensor | int]]:
    """Build complete rankings while loading only one MoE layer at a time."""

    tables: dict[int, dict[str, torch.Tensor | int]] = {}
    for layer_id in adapter.architecture.moe_layer_ids():
        tensors = _load_named_tensors(model_path, weight_map, adapter.scoring_tensor_names(layer_id))
        if apply_input_scale and adapter.input_scale_name(layer_id) not in tensors:
            raise KeyError(f"Missing required CSP input scale for layer {layer_id}.")
        if adapter.architecture.tensor_codec == "packed":
            scores = score_packed_layer(adapter, tensors, layer_id, apply_input_scale, canonicalize)
        else:
            scores = score_separate_layer(adapter, tensors, layer_id, apply_input_scale, canonicalize)
        tables[int(layer_id)] = ranking_table(scores, adapter.architecture.channel_alignment)
        print(f"scored_layer={layer_id}", flush=True)
    return tables


def build_channel_payload(
    *,
    model_path: Path,
    adapter: CSPModelAdapter,
    tables: dict[int, dict[str, torch.Tensor | int]],
    apply_input_scale: bool,
    canonicalize: bool,
) -> dict[str, Any]:
    """Create the immutable, data-free CSP ranking cache."""

    architecture = adapter.architecture
    moe_ids = architecture.moe_layer_ids()
    validate_rankings(tables, len(moe_ids), architecture.num_experts, architecture.intermediate_size, layer_ids=moe_ids)
    return {
        "schema_version": 1,
        "purpose": "csp_channel_ranking",
        "method": "csp",
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
        "score_mode": (
            "per_expert_canonical_structural_participation_fp32"
            if canonicalize else "per_expert_structural_participation_fp32"
        ),
        "table": tables,
        "csp": {
            "data_free": True,
            "weight_only": True,
            "accumulator_dtype": "float32",
            "functional_viability_threshold": DEFAULT_FUNCTIONAL_VIABILITY_THRESHOLD,
            "input_scale_mode": "gemma4_pre_feedforward_layernorm_2" if apply_input_scale else "none",
            "criterion": (
                "log_renyi2_divergence_from_uniform_canonical_signature"
                if canonicalize else "log_renyi2_divergence_from_uniform_raw_signature"
            ),
            "canonicalization": canonicalize,
        },
    }


def build_profile(
    *, model_path: Path, adapter: CSPModelAdapter, retained_channels: int, target_pruning_ratio: float,
    apply_input_scale: bool, canonicalize: bool
) -> dict[str, Any]:
    """Build a uniform-width calibration-free CSP pruning profile."""

    architecture = adapter.architecture
    architecture.validate_width(retained_channels)
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
        "method": "csp",
        "mode": (
            "per_expert_fixed_canonical_structural_participation"
            if canonicalize else "per_expert_fixed_structural_participation"
        ),
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
        "csp": {
            "data_free": True,
            "weight_only": True,
            "accumulator_dtype": "float32",
            "input_scale_mode": "gemma4_pre_feedforward_layernorm_2" if apply_input_scale else "none",
            "canonicalization": canonicalize,
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
    profile_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_retained_indices(channel: dict[str, Any], retained_channels: int, output_path: Path) -> None:
    output_path.write_text(
        json.dumps({
            "schema_version": 1,
            "method": "csp",
            "retained_channels": int(retained_channels),
            "layer_ids": sorted(int(layer_id) for layer_id in channel["table"]),
            "retained_indices": {
                str(int(layer_id)): table["ranked_indices"][:, :int(retained_channels)].tolist()
                for layer_id, table in channel["table"].items()
            },
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_existing_cache(
    channel: dict[str, Any], *, model_path: Path, adapter: CSPModelAdapter,
    apply_input_scale: bool, canonicalize: bool
) -> None:
    """Reject a stale or non-CSP cache before profile creation."""

    if int(channel.get("schema_version", -1)) != 1 or channel.get("purpose") != "csp_channel_ranking":
        raise ValueError("Unsupported CSP channel ranking payload.")
    if Path(str(channel.get("model_path", ""))).resolve() != model_path:
        raise ValueError("CSP rankings were built for a different model path.")
    if channel.get("model_family") != adapter.architecture.model_family:
        raise ValueError("CSP rankings model family does not match the checkpoint.")
    if channel.get("csp", {}).get("data_free") is not True or channel.get("csp", {}).get("weight_only") is not True:
        raise ValueError("CSP rankings must be data-free and weight-only.")
    expected_scale_mode = "gemma4_pre_feedforward_layernorm_2" if apply_input_scale else "none"
    if channel.get("csp", {}).get("input_scale_mode") != expected_scale_mode:
        raise ValueError("Existing CSP rankings use a different input-scale mode.")
    if channel.get("csp", {}).get("canonicalization") is not canonicalize:
        raise ValueError("Existing CSP rankings use a different canonicalization mode.")
    provenance = channel.get("model_provenance", {})
    if provenance.get("config_sha256") != file_sha256(model_path / "config.json"):
        raise ValueError("Checkpoint config changed after CSP ranking construction.")
    if provenance.get("weight_index_sha256") != file_sha256(model_path / "model.safetensors.index.json"):
        raise ValueError("Checkpoint weight index changed after CSP ranking construction.")
    architecture = adapter.architecture
    validate_rankings(channel["table"], len(architecture.moe_layer_ids()), architecture.num_experts, architecture.intermediate_size, layer_ids=architecture.moe_layer_ids())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data-free Canonical Structural Participation rankings and a static profile.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float)
    parser.add_argument("--retained-channels", type=int)
    parser.add_argument("--rounding", choices=("floor", "nearest", "ceil"), default="nearest")
    parser.add_argument("--apply-input-scale", choices=("auto", "always", "never"), default="auto")
    parser.add_argument(
        "--canonicalize", "--canonicalization", action="store_true",
        help="Canonicalize the up/down gauge before scoring. Disabled by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.retained_channels is None) == (args.target_pruning_ratio is None):
        raise ValueError("Provide exactly one of --retained-channels or --target-pruning-ratio.")
    model_path = args.model_path.expanduser().resolve()
    channel_path = args.output_channel_cache.expanduser().resolve()
    profile_path = args.output_profile.expanduser().resolve()
    weight_map = load_weight_map(model_path)
    adapter = CSPModelAdapter.from_checkpoint(model_path, weight_map)
    apply_input_scale = args.apply_input_scale == "always" or (
        args.apply_input_scale == "auto" and adapter.architecture.model_family == "gemma4"
    )
    canonicalize = bool(args.canonicalize)
    if apply_input_scale and adapter.input_scale_template is None:
        if args.apply_input_scale == "always":
            raise ValueError(f"{adapter.architecture.model_family} has no supported expert input scale.")
        apply_input_scale = False
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
        validate_existing_cache(
            channel, model_path=model_path, adapter=adapter,
            apply_input_scale=apply_input_scale, canonicalize=canonicalize,
        )
    else:
        channel = build_channel_payload(
            model_path=model_path,
            adapter=adapter,
            tables=build_layer_rankings(
                model_path, adapter, weight_map, apply_input_scale, canonicalize
            ),
            apply_input_scale=apply_input_scale,
            canonicalize=canonicalize,
        )
        channel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(channel, channel_path)
    profile = build_profile(
        model_path=model_path,
        adapter=adapter,
        retained_channels=retained_channels,
        target_pruning_ratio=target_ratio,
        apply_input_scale=apply_input_scale,
        canonicalize=canonicalize,
    )
    profile["cache_provenance"] = {"channel": {"path": str(channel_path), "sha256": file_sha256(channel_path), "role": "csp_ranking"}}
    write_profile(profile, profile_path)
    write_retained_indices(channel, retained_channels, profile_path.with_name(f"csp_retained_{retained_channels}ch.json"))
    if abs(profile["actual_structural_pruning_ratio"] - target_ratio) > 1.0e-12:
        print(f"WARNING: requested pruning ratio {target_ratio:.8f} was aligned to {profile['actual_structural_pruning_ratio']:.8f} ({retained_channels} channels).", flush=True)
    print(channel_path)
    print(profile_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
