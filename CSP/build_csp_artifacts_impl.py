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
    expert_structural_score,
    expert_structural_score_packed,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every separate-layout routed expert in one layer."""

    scale = tensors.get(adapter.input_scale_name(layer_id)) if apply_input_scale else None
    rows = []
    expert_scores = []
    for expert_id in range(adapter.architecture.num_experts):
        gate = tensors[adapter.gate_name(layer_id, expert_id)]
        up = tensors[adapter.up_name(layer_id, expert_id)]
        down = tensors[adapter.down_name(layer_id, expert_id)]
        rows.append(
            canonical_structural_score(
                gate,
                up,
                down,
                input_scale=scale,
                functional_viability_threshold=DEFAULT_FUNCTIONAL_VIABILITY_THRESHOLD,
                canonicalize=canonicalize,
            )
        )
        expert_scores.append(expert_structural_score(gate, up, down))
    return torch.stack(rows), torch.stack(expert_scores)


def score_packed_layer(
    adapter: CSPModelAdapter,
    tensors: dict[str, torch.Tensor],
    layer_id: int,
    apply_input_scale: bool,
    canonicalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    expert_scores = torch.stack([
        expert_structural_score_packed(gate_up[expert_id], down[expert_id])
        for expert_id in range(adapter.architecture.num_experts)
    ])
    if scores.shape[1] != adapter.architecture.intermediate_size:
        raise ValueError("Packed CSP width does not match moe_intermediate_size.")
    return scores, expert_scores


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
            scores, expert_scores = score_packed_layer(adapter, tensors, layer_id, apply_input_scale, canonicalize)
        else:
            scores, expert_scores = score_separate_layer(adapter, tensors, layer_id, apply_input_scale, canonicalize)
        table = ranking_table(scores, adapter.architecture.channel_alignment)
        table["expert_structural_scores"] = expert_scores.to(dtype=torch.float32).cpu()
        tables[int(layer_id)] = table
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
            if canonicalize else "per_expert_structural_participation_aimer_compatible_fp32"
        ),
        "table": tables,
        "csp": {
            "data_free": True,
            "weight_only": True,
            "accumulator_dtype": "float32",
            "functional_viability_threshold": DEFAULT_FUNCTIONAL_VIABILITY_THRESHOLD,
            "raw_ranking_eps": 1.0e-8 if not canonicalize and not apply_input_scale else None,
            "raw_ranking_compatibility": (
                "aimer_channel_fp32" if not canonicalize and not apply_input_scale else "not_applicable"
            ),
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


def build_heterogeneous_profile(
    *,
    model_path: Path,
    adapter: CSPModelAdapter,
    tables: dict[int, dict[str, torch.Tensor | int]],
    width_options: list[int],
    budget_width: int,
    apply_input_scale: bool,
    canonicalize: bool,
) -> dict[str, Any]:
    """Build an HSP-Hetero profile with fixed Expert-SP quantile tiers."""

    architecture = adapter.architecture
    if architecture.model_family not in {"qwen3", "qwen3.6", "gemma4", "deepseek_v2", "olmoe", "mixtral"}:
        raise ValueError("HSP-Hetero does not support this model family.")
    if canonicalize:
        raise ValueError("HSP-Hetero uses raw Expert-SP and raw Channel-SP; canonicalization is not supported.")
    block_size = architecture.channel_alignment
    options = sorted({int(width) for width in width_options})
    if len(options) != 3:
        raise ValueError("HSP-Hetero requires exactly three width options.")
    for width in options:
        architecture.validate_width(width)
    architecture.validate_width(int(budget_width))
    if options[1] != int(budget_width) or options[1] - options[0] != options[2] - options[1]:
        raise ValueError("HSP-Hetero width options must be symmetric around budget_width.")
    if architecture.num_experts % 4:
        raise ValueError("HSP-Hetero requires the number of experts to be divisible by four.")

    moe_ids = architecture.moe_layer_ids()
    widths_by_layer = []
    expert_scores_by_layer = []
    width_histograms = []
    target_blocks = architecture.num_experts * int(budget_width) // block_size
    for layer_id in moe_ids:
        expert_scores = tables[int(layer_id)].get("expert_structural_scores")
        if not isinstance(expert_scores, torch.Tensor) or tuple(expert_scores.shape) != (architecture.num_experts,):
            raise ValueError(f"Layer {layer_id} is missing HSP Expert-SP scores.")
        order = torch.argsort(expert_scores, descending=True, stable=True)
        quarter = architecture.num_experts // 4
        layer_widths = torch.full(
            (architecture.num_experts,), int(budget_width) // block_size, dtype=torch.long
        )
        layer_widths[order[:quarter]] = options[2] // block_size
        layer_widths[order[-quarter:]] = options[0] // block_size
        widths_by_layer.append(layer_widths)
        expert_scores_by_layer.append(expert_scores.tolist())
        counts = {str(width): int((layer_widths == width // block_size).sum().item()) for width in options}
        width_histograms.append(counts)

    widths = torch.stack(widths_by_layer)
    if widths.sum(dim=1).tolist() != [target_blocks] * len(moe_ids):
        raise RuntimeError("heterogeneous CSP allocation violated the exact per-layer budget.")
    num_blocks = architecture.intermediate_size // block_size
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    actual_ratio = 1.0 - int(budget_width) / architecture.intermediate_size
    profile = {
        "schema_version": 1,
        "method": "hsp",
        "mode": "hsp_hetero_raw_expert_sp_quantiles",
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
        "allocation_scope": "per_layer_expert_sp_quantiles",
        "allocation_objective": (
            f"top_25_percent_expert_sp_to_{options[2]}_middle_50_percent_to_{options[1]} "
            f"bottom_25_percent_to_{options[0]}"
        ),
        "target_blocks_by_layer": [target_blocks] * len(moe_ids),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "total_blocks": total_blocks,
        "maximum_blocks": maximum_blocks,
        "target_pruning_ratio": actual_ratio,
        "actual_structural_pruning_ratio": actual_ratio,
        "retained_channels": None,
        "budget_reference_width": int(budget_width),
        "width_options": options,
        "padded_intermediate_size": options[-1],
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
            "heterogeneous_allocation": {
                "expert_score": "log(N_E * ||Theta_e||_2^2 / ||Theta_e||_1^2)",
                "expert_score_canonicalization": False,
                "allocation": "layerwise_quantiles_25_50_25",
                "width_options": options,
                "budget_reference_width": int(budget_width),
                "expert_structural_scores_by_layer": expert_scores_by_layer,
                "width_histograms_by_layer": width_histograms,
            },
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


def write_retained_indices(channel: dict[str, Any], profile: dict[str, Any], output_path: Path) -> None:
    block_size = int(profile["channel_block_size"])
    layer_ids = [int(layer_id) for layer_id in profile["layer_ids"]]
    widths = profile["profile_widths"].to(dtype=torch.long)
    output_path.write_text(
        json.dumps({
            "schema_version": 1,
            "method": "csp",
            "retained_channels": profile.get("retained_channels"),
            "budget_reference_width": profile.get("budget_reference_width"),
            "width_options": profile.get("width_options"),
            "layer_ids": sorted(int(layer_id) for layer_id in channel["table"]),
            "widths_by_layer": (widths * block_size).tolist(),
            "retained_indices": {
                str(layer_id): [
                    channel["table"][layer_id]["ranked_indices"][expert_id, :int(widths[row, expert_id].item()) * block_size].tolist()
                    for expert_id in range(int(profile["num_experts"]))
                ]
                for row, layer_id in enumerate(layer_ids)
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
    parser.add_argument("--heterogeneous-widths", type=int, nargs="+")
    parser.add_argument("--budget-width", type=int)
    parser.add_argument("--rounding", choices=("floor", "nearest", "ceil"), default="nearest")
    parser.add_argument("--apply-input-scale", choices=("auto", "always", "never"), default="auto")
    parser.add_argument(
        "--canonicalize", "--canonicalization", action="store_true",
        help="Canonicalize the up/down gauge before scoring. Disabled by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    heterogeneous = args.heterogeneous_widths is not None or args.budget_width is not None
    if heterogeneous:
        if args.heterogeneous_widths is None or args.budget_width is None:
            raise ValueError("Provide both --heterogeneous-widths and --budget-width.")
        if args.retained_channels is not None or args.target_pruning_ratio is not None:
            raise ValueError("Heterogeneous CSP cannot be combined with a uniform width or pruning ratio.")
    elif (args.retained_channels is None) == (args.target_pruning_ratio is None):
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
    heterogeneous = args.heterogeneous_widths is not None or args.budget_width is not None
    if heterogeneous and apply_input_scale:
        if args.apply_input_scale == "always":
            raise ValueError("HSP-Hetero uses raw parameter signatures and does not support input scaling.")
        apply_input_scale = False
    if apply_input_scale and adapter.input_scale_template is None:
        if args.apply_input_scale == "always":
            raise ValueError(f"{adapter.architecture.model_family} has no supported expert input scale.")
        apply_input_scale = False
    architecture = adapter.architecture
    if heterogeneous:
        retained_channels = None
        target_ratio = 1.0 - int(args.budget_width) / architecture.intermediate_size
    elif args.retained_channels is not None:
        retained_channels = int(args.retained_channels)
        target_ratio = 1.0 - retained_channels / architecture.intermediate_size
    else:
        target_ratio = float(args.target_pruning_ratio)
        retained_channels = architecture.width_for_pruning(target_ratio, args.rounding)
    if retained_channels is not None:
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
    if heterogeneous:
        profile = build_heterogeneous_profile(
            model_path=model_path,
            adapter=adapter,
            tables=channel["table"],
            width_options=list(args.heterogeneous_widths),
            budget_width=int(args.budget_width),
            apply_input_scale=apply_input_scale,
            canonicalize=canonicalize,
        )
    else:
        profile = build_profile(
            model_path=model_path,
            adapter=adapter,
            retained_channels=int(retained_channels),
            target_pruning_ratio=target_ratio,
            apply_input_scale=apply_input_scale,
            canonicalize=canonicalize,
        )
    profile["cache_provenance"] = {"channel": {"path": str(channel_path), "sha256": file_sha256(channel_path), "role": "csp_ranking"}}
    write_profile(profile, profile_path)
    retained_name = (
        f"csp_retained_heterogeneous_budget{args.budget_width}ch.json"
        if heterogeneous else f"csp_retained_{retained_channels}ch.json"
    )
    write_retained_indices(channel, profile, profile_path.with_name(retained_name))
    if abs(profile["actual_structural_pruning_ratio"] - target_ratio) > 1.0e-12:
        print(f"WARNING: requested pruning ratio {target_ratio:.8f} was aligned to {profile['actual_structural_pruning_ratio']:.8f} ({retained_channels} channels).", flush=True)
    print(channel_path)
    print(profile_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
