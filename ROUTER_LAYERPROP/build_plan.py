"""Build a Router-conditioned Multi-origin LayerProp pruning plan."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .adapters import adapter_for_model
from .config import LayerPropConfig
from .planner import build_layer_plan, plan_summary
from .propagation import (
    collect_local_rows,
    run_refresh_propagation,
    run_router_long_propagation,
    run_source0_propagation,
    supported_layer_ids,
)
from .synthetic import embedding_probe_scale, synthetic_input_ids


def load_hf_model(model_path: Path, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Load a supported Transformers model only when plan construction is requested."""

    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    architectures = [str(item) for item in (getattr(config, "architectures", None) or [])]
    loader = AutoModelForImageTextToText if any("ConditionalGeneration" in item for item in architectures) else AutoModelForCausalLM
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    try:
        model = loader.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        model = loader.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
    return model.to(device).eval()


def _config_from_args(args: argparse.Namespace) -> LayerPropConfig:
    values = LayerPropConfig(
        propagation_mode=args.propagation_mode,
        num_pseudo_tokens=args.num_pseudo_tokens,
        sequence_length=args.sequence_length,
        probe_variants=args.probe_variants,
        refresh_stride=args.refresh_stride,
        refresh_horizon=args.refresh_horizon,
        max_rows_per_expert_per_origin=args.max_rows_per_expert,
        min_train_rows=args.min_train_rows,
        min_valid_rows=args.min_valid_rows,
        recoverability_band=args.recoverability_band,
        channel_multiple=args.channel_multiple,
    )
    values.validate()
    return values


def build_plan(model_path: Path, output_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Construct and save a complete data-free pruning plan."""

    config = _config_from_args(args)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model = load_hf_model(model_path, device, dtype)
    adapter = adapter_for_model(model)
    layer_ids = supported_layer_ids(adapter)
    requested_width = args.retained_channels
    if requested_width is None:
        requested_width = round(adapter.metadata.intermediate_size * (1.0 - args.pruning_ratio))
    retained_channels = max(
        config.channel_multiple,
        int(requested_width) // config.channel_multiple * config.channel_multiple,
    )
    retained_channels = min(
        retained_channels,
        adapter.metadata.intermediate_size - config.channel_multiple,
    )
    if retained_channels <= 0 or retained_channels >= adapter.metadata.intermediate_size:
        raise ValueError("retained width must be positive, aligned, and smaller than source width")
    input_ids = synthetic_input_ids(
        vocab_size=int(getattr(adapter.text_config(), "vocab_size")),
        num_sequences=config.num_sequences,
        sequence_length=config.sequence_length,
        bos_token_id=int(getattr(adapter.text_config(), "bos_token_id", 2) or 2),
        pad_token_id=int(getattr(adapter.text_config(), "pad_token_id", 0) or 0),
    )
    scale = embedding_probe_scale(model)
    print(
        f"router_layerprop_start family={adapter.metadata.family} layers={len(layer_ids)} "
        f"tokens={config.num_pseudo_tokens} sequence_length={config.sequence_length} retained={retained_channels}",
        flush=True,
    )
    use_router_long = config.propagation_mode == "long_short" and adapter.supports_refresh_propagation
    if use_router_long:
        train_rows, valid_rows, layer_scales = run_router_long_propagation(
            adapter,
            layer_ids,
            config,
            scale,
        )
    else:
        train_rows, valid_rows, layer_scales = run_source0_propagation(
            model,
            adapter,
            input_ids,
            layer_ids=layer_ids,
            config=config,
            device=device,
        )
    local_rows = collect_local_rows(adapter, layer_ids, config, layer_scales, scale)
    refresh_rows = (
        run_refresh_propagation(adapter, layer_ids, config, layer_scales, scale)
        if config.propagation_mode == "long_short"
        else {}
    )
    refresh_by_layer: dict[int, dict[str, dict[int, torch.Tensor]]] = {}
    for source_name, target_rows in refresh_rows.items():
        for target_layer, rows in target_rows.items():
            refresh_by_layer.setdefault(target_layer, {})[source_name] = rows
    plans: dict[int, dict[int, dict[str, Any]]] = {}
    for layer_id in layer_ids:
        gate_up, down = adapter.expert_weights(adapter.layers()[layer_id])
        if use_router_long:
            origin_rows: dict[str, dict[int, torch.Tensor]] = {
                "source0_long": train_rows[layer_id],
            }
            selection_source_rows: dict[str, dict[int, torch.Tensor]] = {
                "source0_long": train_rows[layer_id],
                "target_local": local_rows[layer_id],
            }
            selection_source_rows.update(refresh_by_layer.get(layer_id, {}))
            validation_source_rows: dict[str, dict[int, torch.Tensor]] = {
                "source0_long": valid_rows[layer_id],
                "target_local": local_rows[layer_id],
            }
            validation_source_rows.update(refresh_by_layer.get(layer_id, {}))
        else:
            origin_rows = {
                "source0_long": train_rows[layer_id],
                "target_local": local_rows[layer_id],
            }
            selection_source_rows = {}
            validation_source_rows = {}
        plans[layer_id] = build_layer_plan(
            source_rows=origin_rows,
            validation_rows=None if use_router_long else valid_rows[layer_id],
            validation_source_rows=validation_source_rows if use_router_long else None,
            fallback_source_rows={"target_local": local_rows[layer_id]} if use_router_long else None,
            selection_source_rows=selection_source_rows if use_router_long else None,
            down_proj=down,
            retained_channels=retained_channels,
            config=config,
        )
        print(f"router_layerprop_layer={layer_id} experts={adapter.metadata.num_experts}", flush=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": "router_conditioned_multi_origin_layerprop",
        "propagation_mode": config.propagation_mode,
        "data_free": True,
        "model_path": str(model_path.resolve()),
        "model_family": adapter.metadata.family,
        "metadata": asdict(adapter.metadata),
        "config": asdict(config),
        "retained_channels": retained_channels,
        "layer_ids": list(layer_ids),
        "layers": plans,
        "summary": plan_summary(plans),
        "provenance": {
            "probe_source": (
                "router_conditioned_source0_long_plus_stride_refresh_plus_target_local"
                if use_router_long
                else "router_region_directions_plus_deterministic_vocab_lattice"
            ),
            "calibration_corpus": None,
            "source0_split": "sequence_order_train_valid",
            "refresh_schedule": "stride_and_horizon_nearest_source" if refresh_rows else "disabled",
            "refresh_origins": sorted(refresh_rows),
            "refresh_supported": bool(adapter.supports_refresh_propagation),
            "layer_scales": layer_scales,
            "source0_origin": "router_conditioned_long_bank" if use_router_long else "vocab_lattice_native_forward",
            "effective_mode": (
                "long_short"
                if refresh_rows
                else "router_long_local"
                if use_router_long
                else "stable_fallback"
                if config.propagation_mode == "long_short"
                else "stable"
            ),
            "shared_expert_pruned": False,
            "original_model_kept_untouched_during_planning": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"router_layerprop_plan={output_path}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--propagation-mode", choices=("stable", "long_short"), default="long_short")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--retained-channels", type=int)
    parser.add_argument("--pruning-ratio", type=float, default=0.5)
    parser.add_argument("--num-pseudo-tokens", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--probe-variants", type=int, default=8)
    parser.add_argument("--refresh-stride", type=int, default=4)
    parser.add_argument("--refresh-horizon", type=int, default=8)
    parser.add_argument("--max-rows-per-expert", type=int, default=128)
    parser.add_argument("--min-train-rows", type=int, default=16)
    parser.add_argument("--min-valid-rows", type=int, default=8)
    parser.add_argument("--recoverability-band", type=int, default=32)
    parser.add_argument("--channel-multiple", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output_plan.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    model_path = args.model_path.expanduser().resolve()
    build_plan(model_path, output_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
