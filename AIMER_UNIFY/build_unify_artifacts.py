from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from AIMER_Mix.mix_core import file_sha256, validate_rankings
from AIMER_Mix.model_adapter import AIMERMixModelAdapter
from AIMER_MIX_PLUS.build_aimer_mix_plus_artifacts import (
    build_tables,
    load_aimer_mix_cache,
    load_weight_map,
    source_metadata,
    stack_base_orders,
)
from AIMER_MIX_PLUS.source_cache import load_pseudo_source
from AIMER_UNIFY.unify_core import UnifyConfig, build_unify_ranking_from_order
from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload


def parse_named_path(spec: str) -> tuple[str, Path]:
    name, separator, raw_path = spec.partition("=")
    if not separator or name not in {"prp", "layerprop"} or not raw_path:
        raise ValueError("source must use prp=PATH or layerprop=PATH")
    return name, Path(raw_path).expanduser().resolve()


def summarize_unify(diagnostics: dict[str, Any]) -> dict[str, Any]:
    summary = dict(diagnostics.get("diagnostic_summary", {}))
    flat = [record for layer in diagnostics.get("diagnostics", []) for record in layer]
    lambdas = [
        float(record["layerprop_lambda"])
        for record in flat
        if record.get("layerprop_lambda") is not None
    ]
    if lambdas:
        values = torch.tensor(lambdas)
        summary["layerprop_lambda_mean"] = float(values.mean().item())
        summary["layerprop_lambda_min"] = float(values.min().item())
        summary["layerprop_lambda_max"] = float(values.max().item())
    return summary


def build_channel_payload(
    *,
    model_path: Path,
    adapter: AIMERMixModelAdapter,
    tables: dict[int, dict[str, torch.Tensor | int | float]],
    retained_channels: int,
    base_cache_path: Path,
    base_cache: dict[str, Any],
    sources: list[Any],
    config: UnifyConfig,
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
        "purpose": "aimer_unify_ranking",
        "method": "aimer_unify",
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
        "aimer_unify": {
            "data_free": True,
            "training_free": True,
            "base": "aimer_mix",
            "gate": "mix_keepset_overlap_with_ffn_pseudo",
            "base_cache": {
                "path": str(base_cache_path),
                "sha256": file_sha256(base_cache_path),
                "energy_mode": base_cache.get("aimer_mix", {}).get("energy_mode"),
            },
            "sources": [source_metadata(source) for source in sources],
            "fusion_config": {"layerprop_tau": float(config.layerprop_tau)},
            "diagnostic_summary": summarize_unify(diagnostics),
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
        "method": "aimer_unify",
        "mode": "agreement_gated_mix_with_ffn_pseudo",
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
                "role": "aimer_unify_width_specific_ranking",
            }
        },
        "aimer_unify": {
            "data_free": True,
            "base": "aimer_mix",
            "sources": channel["aimer_unify"]["sources"],
            "fusion_config": channel["aimer_unify"]["fusion_config"],
            "diagnostic_summary": channel["aimer_unify"]["diagnostic_summary"],
            "architecture": adapter.metadata(),
        },
    }
    validate_static_profile_payload(profile)
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build agreement-gated AIMER-Unify rankings.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--aimer-mix-cache", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], help="prp=PATH or layerprop=PATH")
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--retained-channels", type=int, required=True)
    parser.add_argument("--layerprop-tau", type=float, default=8.0)
    parser.add_argument("--allow-source-model-path-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    retained_channels = int(args.retained_channels)
    architecture.validate_width(retained_channels)
    target_ratio = 1.0 - retained_channels / architecture.intermediate_size
    base_cache = load_aimer_mix_cache(base_cache_path, model_path=model_path, adapter=adapter)
    layer_ids = architecture.moe_layer_ids()
    base_orders = stack_base_orders(base_cache, layer_ids)
    source_paths = dict(parse_named_path(spec) for spec in args.source)
    if len(source_paths) != len(args.source):
        raise ValueError("Pseudo source names must be unique")
    if "prp" not in source_paths or "layerprop" not in source_paths:
        raise ValueError("Unified fusion requires both prp and layerprop sources")
    sources = [
        load_pseudo_source(
            name=name,
            cache_path=path,
            layer_ids=layer_ids,
            num_experts=architecture.num_experts,
            channels=architecture.intermediate_size,
            model_path=model_path,
            strict_model_path=not args.allow_source_model_path_mismatch,
        )
        for name, path in source_paths.items()
    ]
    config = UnifyConfig(layerprop_tau=float(args.layerprop_tau))
    orders, diagnostics = build_unify_ranking_from_order(
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
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["profile_file_sha256"] = file_sha256(profile_path)
    profile_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(channel_path)
    print(profile_path)
    print(json.dumps(channel["aimer_unify"]["diagnostic_summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
