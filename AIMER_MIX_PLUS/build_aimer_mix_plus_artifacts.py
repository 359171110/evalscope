from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from AIMER_Mix.mix_core import file_sha256, ranking_table, validate_rankings
from AIMER_Mix.model_adapter import AIMERMixModelAdapter
from AIMER_MIX_PLUS.plus_core import (
    SOURCE_NAMES,
    AIMERMixPlusConfig,
    PseudoSource,
    build_plus_ranking_from_order,
    rank_percentiles_from_order,
)
from AIMER_MIX_PLUS.source_cache import load_pseudo_source
from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload


def load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json is missing weight_map")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def parse_named_path(spec: str) -> tuple[str, Path]:
    name, separator, raw_path = spec.partition("=")
    if not separator or name not in SOURCE_NAMES or not raw_path:
        raise ValueError("source must use pp=PATH, prp=PATH, or layerprop=PATH")
    return name, Path(raw_path).expanduser().resolve()


def parse_named_float(spec: str, *, label: str) -> tuple[str, float]:
    name, separator, raw_value = spec.partition("=")
    if not separator or name not in SOURCE_NAMES or not raw_value:
        raise ValueError(f"{label} must use SOURCE=FLOAT")
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{label} value must be numeric") from error
    return name, value


def parse_named_float_map(specs: list[str], *, label: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for spec in specs:
        name, value = parse_named_float(spec, label=label)
        if name in values:
            raise ValueError(f"Duplicate {label} for source {name}")
        values[name] = value
    return values


def load_aimer_mix_cache(
    cache_path: Path,
    *,
    model_path: Path,
    adapter: AIMERMixModelAdapter,
) -> dict[str, Any]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("AIMER-Mix cache must contain a mapping")
    if payload.get("purpose") != "aimer_mix_ranking" or payload.get("method") != "aimer_mix":
        raise ValueError("Expected an AIMER-Mix channel ranking cache")
    if Path(str(payload.get("model_path", ""))).expanduser().resolve() != model_path:
        raise ValueError("AIMER-Mix cache was built for a different model path")
    architecture = adapter.architecture
    if payload.get("model_family") != architecture.model_family:
        raise ValueError("AIMER-Mix cache model family does not match the checkpoint")
    if int(payload.get("calibration_sequences", 0) or 0) != 0 or payload.get("test_metrics_used") is True:
        raise ValueError("AIMER-Mix base cache must be calibration-free")
    provenance = payload.get("model_provenance", {})
    if provenance.get("config_sha256") != file_sha256(model_path / "config.json"):
        raise ValueError("Checkpoint config changed after AIMER-Mix ranking construction")
    index_path = model_path / "model.safetensors.index.json"
    if provenance.get("weight_index_sha256") != file_sha256(index_path):
        raise ValueError("Checkpoint weight index changed after AIMER-Mix ranking construction")
    layer_ids = architecture.moe_layer_ids()
    validate_rankings(
        payload["table"],
        len(layer_ids),
        architecture.num_experts,
        architecture.intermediate_size,
        layer_ids=layer_ids,
    )
    return payload


def stack_base_orders(cache: dict[str, Any], layer_ids: tuple[int, ...]) -> torch.Tensor:
    return torch.stack([
        cache["table"][layer_id]["ranked_indices"].to(torch.long).cpu()
        for layer_id in layer_ids
    ])


def build_tables(
    orders: torch.Tensor,
    *,
    base_cache: dict[str, Any],
    layer_ids: tuple[int, ...],
    block_size: int,
) -> dict[int, dict[str, torch.Tensor | int | float]]:
    ranks = rank_percentiles_from_order(orders)
    tables: dict[int, dict[str, torch.Tensor | int | float]] = {}
    for position, layer_id in enumerate(layer_ids):
        base_row = base_cache["table"][layer_id]
        expert_alpha = base_row.get("expert_alpha")
        tables[layer_id] = ranking_table(ranks[position], block_size, expert_alpha=expert_alpha)
    return tables


def source_metadata(source: PseudoSource) -> dict[str, Any]:
    coverage = source.coverage if source.coverage is not None else torch.ones(source.order.shape[:2])
    stability = source.stability if source.stability is not None else torch.ones(source.order.shape[:2])
    return {
        "name": source.name,
        "base_weight": float(source.base_weight),
        "coverage_mean": float(coverage.float().mean().item()),
        "coverage_min": float(coverage.float().min().item()),
        "stability_mean": float(stability.float().mean().item()),
        "stability_min": float(stability.float().min().item()),
        **source.metadata,
    }


def summarize_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    flat = [record for layer in payload["diagnostics"] for record in layer]
    active = Counter(name for record in flat for name in record["active_sources"])
    swap_counts = torch.tensor([float(record.get("swap_count", 0)) for record in flat])
    pseudo_mass = torch.tensor([float(record.get("pseudo_mass", 0.0)) for record in flat])
    agreement_mass = torch.tensor([float(record.get("agreement_mass", 0.0)) for record in flat])
    layer_summary = []
    for layer in payload["diagnostics"]:
        layer_swaps = torch.tensor([float(record.get("swap_count", 0)) for record in layer])
        layer_summary.append({
            "layer_id": int(layer[0]["layer_id"]) if layer else -1,
            "swap_count_mean": float(layer_swaps.mean().item()) if layer else 0.0,
            "swap_count_max": int(layer_swaps.max().item()) if layer else 0,
        })
    return {
        "expert_count": len(flat),
        "active_source_expert_counts": dict(active),
        "swap_count_mean": float(swap_counts.mean().item()) if flat else 0.0,
        "swap_count_max": int(swap_counts.max().item()) if flat else 0,
        "pseudo_mass_mean": float(pseudo_mass.mean().item()) if flat else 0.0,
        "agreement_mass_mean": float(agreement_mass.mean().item()) if flat else 0.0,
        "layers": layer_summary,
    }


def build_channel_payload(
    *,
    model_path: Path,
    adapter: AIMERMixModelAdapter,
    tables: dict[int, dict[str, torch.Tensor | int | float]],
    retained_channels: int,
    base_cache_path: Path,
    base_cache: dict[str, Any],
    sources: list[PseudoSource],
    config: AIMERMixPlusConfig,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    architecture = adapter.architecture
    layer_ids = architecture.moe_layer_ids()
    validate_rankings(
        tables,
        len(layer_ids),
        architecture.num_experts,
        architecture.intermediate_size,
        layer_ids=layer_ids,
    )
    return {
        "schema_version": 1,
        "purpose": "aimer_mix_plus_ranking",
        "method": "aimer_mix_plus",
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
        "retained_channels": int(retained_channels),
        "ranking_is_width_specific": True,
        "table": tables,
        "aimer_mix_plus": {
            "data_free": True,
            "training_free": True,
            "base": "aimer_mix",
            "base_cache": {
                "path": str(base_cache_path),
                "sha256": file_sha256(base_cache_path),
                "energy_mode": base_cache.get("aimer_mix", {}).get("energy_mode"),
            },
            "sources": [source_metadata(source) for source in sources],
            "fusion_config": asdict(config),
            "diagnostic_summary": summarize_diagnostics(diagnostics),
            "diagnostics": diagnostics["diagnostics"],
            "compensation": "none",
        },
    }


def build_profile(
    *,
    model_path: Path,
    adapter: AIMERMixModelAdapter,
    retained_channels: int,
    target_pruning_ratio: float,
    channel_cache_path: Path,
    channel: dict[str, Any],
) -> dict[str, Any]:
    architecture = adapter.architecture
    layer_ids = architecture.moe_layer_ids()
    block_size = architecture.channel_alignment
    num_blocks = architecture.intermediate_size // block_size
    retained_blocks = retained_channels // block_size
    widths = torch.full((len(layer_ids), architecture.num_experts), retained_blocks, dtype=torch.long)
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    profile = {
        "schema_version": 1,
        "method": "aimer_mix_plus",
        "mode": "aimer_mix_core_with_multi_source_pseudo_boundary_rescue",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": list(layer_ids),
        "num_layers": len(layer_ids),
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
        "actual_structural_pruning_ratio": 1.0 - retained_channels / architecture.intermediate_size,
        "retained_channels": retained_channels,
        "retained_expert_mask": None,
        "profile_widths": widths,
        "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
        "cache_provenance": {
            "channel": {
                "path": str(channel_cache_path),
                "sha256": file_sha256(channel_cache_path),
                "role": "aimer_mix_plus_width_specific_ranking",
            }
        },
        "aimer_mix_plus": {
            "data_free": True,
            "base": "aimer_mix",
            "sources": channel["aimer_mix_plus"]["sources"],
            "fusion_config": channel["aimer_mix_plus"]["fusion_config"],
            "diagnostic_summary": channel["aimer_mix_plus"]["diagnostic_summary"],
            "architecture": adapter.metadata(),
        },
    }
    validate_static_profile_payload(profile)
    return profile


def write_json_summary(profile: dict[str, Any], profile_path: Path) -> None:
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["profile_file_sha256"] = file_sha256(profile_path)
    profile_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_retained_indices(channel: dict[str, Any], output_path: Path) -> None:
    retained_channels = int(channel["retained_channels"])
    payload = {
        "schema_version": 1,
        "method": "aimer_mix_plus",
        "retained_channels": retained_channels,
        "retained_indices": {
            str(layer_id): row["ranked_indices"][:, :retained_channels].tolist()
            for layer_id, row in channel["table"].items()
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build width-specific AIMER-Mix-Plus rankings and profile.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--aimer-mix-cache", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], help="Pseudo ranking source: SOURCE=PATH")
    parser.add_argument("--source-stability", action="append", default=[], help="Override: SOURCE=FLOAT")
    parser.add_argument("--source-coverage-floor", action="append", default=[], help="Fallback: SOURCE=FLOAT")
    parser.add_argument("--source-base-weight", action="append", default=[], help="Multiplier: SOURCE=FLOAT")
    parser.add_argument("--allow-source-model-path-mismatch", action="store_true")
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float)
    parser.add_argument("--retained-channels", type=int)
    parser.add_argument("--rounding", choices=("floor", "nearest", "ceil"), default="nearest")
    parser.add_argument("--boundary-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-boundary-channels", type=int, default=32)
    parser.add_argument("--maximum-boundary-fraction", type=float, default=0.35)
    parser.add_argument("--base-boundary-weight", type=float, default=0.75)
    parser.add_argument("--pseudo-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-floor", type=float, default=0.70)
    parser.add_argument("--agreement-bonus", type=float, default=0.15)
    parser.add_argument("--disagreement-penalty", type=float, default=0.05)
    parser.add_argument("--rank-temperature", type=float, default=1.0)
    parser.add_argument("--pp-weight", type=float, default=1.0)
    parser.add_argument("--prp-weight", type=float, default=1.0)
    parser.add_argument("--layerprop-weight", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.retained_channels is None) == (args.target_pruning_ratio is None):
        raise ValueError("Provide exactly one of --retained-channels or --target-pruning-ratio")
    model_path = args.model_path.expanduser().resolve()
    base_cache_path = args.aimer_mix_cache.expanduser().resolve()
    channel_path = args.output_channel_cache.expanduser().resolve()
    profile_path = args.output_profile.expanduser().resolve()
    for path in (channel_path, profile_path, profile_path.with_suffix(".json")):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}")
    weight_map = load_weight_map(model_path)
    adapter = AIMERMixModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.architecture
    if args.retained_channels is not None:
        retained_channels = int(args.retained_channels)
        target_ratio = 1.0 - retained_channels / architecture.intermediate_size
    else:
        target_ratio = float(args.target_pruning_ratio)
        retained_channels = architecture.width_for_pruning(target_ratio, args.rounding)
    architecture.validate_width(retained_channels)
    base_cache = load_aimer_mix_cache(base_cache_path, model_path=model_path, adapter=adapter)
    layer_ids = architecture.moe_layer_ids()
    base_orders = stack_base_orders(base_cache, layer_ids)

    source_paths = dict(parse_named_path(spec) for spec in args.source)
    if len(source_paths) != len(args.source):
        raise ValueError("Pseudo source names must be unique")
    stability = parse_named_float_map(args.source_stability, label="source stability")
    coverage_floor = parse_named_float_map(args.source_coverage_floor, label="coverage floor")
    base_weights = parse_named_float_map(args.source_base_weight, label="source base weight")
    sources = [
        load_pseudo_source(
            name=name,
            cache_path=path,
            layer_ids=layer_ids,
            num_experts=architecture.num_experts,
            channels=architecture.intermediate_size,
            model_path=model_path,
            base_weight=base_weights.get(name, 1.0),
            coverage_floor=coverage_floor.get(name, 0.35),
            stability=stability.get(name, 1.0),
            strict_model_path=not args.allow_source_model_path_mismatch,
        )
        for name, path in source_paths.items()
    ]
    config = AIMERMixPlusConfig(
        boundary_fraction=float(args.boundary_fraction),
        minimum_boundary_channels=int(args.minimum_boundary_channels),
        maximum_boundary_fraction=float(args.maximum_boundary_fraction),
        base_boundary_weight=float(args.base_boundary_weight),
        pseudo_weight=float(args.pseudo_weight),
        pseudo_floor=float(args.pseudo_floor),
        agreement_bonus=float(args.agreement_bonus),
        disagreement_penalty=float(args.disagreement_penalty),
        rank_temperature=float(args.rank_temperature),
        source_weights=(
            ("pp", float(args.pp_weight)),
            ("prp", float(args.prp_weight)),
            ("layerprop", float(args.layerprop_weight)),
        ),
    )
    orders, diagnostics = build_plus_ranking_from_order(
        base_orders,
        retained_channels=retained_channels,
        pseudo_sources=sources,
        config=config,
    )
    tables = build_tables(
        orders,
        base_cache=base_cache,
        layer_ids=layer_ids,
        block_size=architecture.channel_alignment,
    )
    channel = build_channel_payload(
        model_path=model_path,
        adapter=adapter,
        tables=tables,
        retained_channels=retained_channels,
        base_cache_path=base_cache_path,
        base_cache=base_cache,
        sources=sources,
        config=config,
        diagnostics=diagnostics,
    )
    channel_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel, channel_path)
    profile = build_profile(
        model_path=model_path,
        adapter=adapter,
        retained_channels=retained_channels,
        target_pruning_ratio=target_ratio,
        channel_cache_path=channel_path,
        channel=channel,
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, profile_path)
    write_json_summary(profile, profile_path)
    write_retained_indices(
        channel,
        profile_path.with_name(f"aimer_mix_plus_retained_{retained_channels}ch.json"),
    )
    print(channel_path)
    print(profile_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())