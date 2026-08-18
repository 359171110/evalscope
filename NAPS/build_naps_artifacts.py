from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from NAPS.naps_core import (
    NapsConfig,
    build_one_to_one_merge_plan,
    effective_evidence,
    effective_zero_mask,
    native_route,
    select_mask,
    stable_concat_score,
    swiglu_response,
    validate_merge_plan,
)
from PP.pure_pseudo_model_adapter import PurePseudoModelAdapter
from WICK.build_wick_profile import file_sha256, rms_norm_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NAPS-Mask rankings and bounded merge plans.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retained-channels", type=int, required=True)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload["weight_map"].items()}


def load_tensor(model_path: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(f"Missing checkpoint tensor: {name}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def load_norm(
    model_path: Path,
    weight_map: dict[str, str],
    adapter: PurePseudoModelAdapter,
    layer_id: int,
) -> tuple[str, torch.Tensor]:
    for name in adapter.norm_names(layer_id):
        if name in weight_map:
            return name, load_tensor(model_path, weight_map, name)
    raise KeyError(f"No pre-MoE RMSNorm tensor found for layer {layer_id}")


def iter_expert_weights(
    model_path: Path,
    weight_map: dict[str, str],
    adapter: PurePseudoModelAdapter,
    layer_id: int,
    device: torch.device,
):
    if adapter.expert_gate_template is not None:
        for expert_id in range(adapter.num_experts):
            gate = load_tensor(model_path, weight_map, adapter.expert_gate_name(layer_id, expert_id)).to(device)
            up = load_tensor(model_path, weight_map, adapter.expert_up_name(layer_id, expert_id)).to(device)
            down = load_tensor(model_path, weight_map, adapter.expert_down_expert_name(layer_id, expert_id)).to(device)
            yield expert_id, gate, up, down
        return

    gate_up = load_tensor(model_path, weight_map, adapter.expert_gate_up_name(layer_id)).to(device)
    down = load_tensor(model_path, weight_map, adapter.expert_down_name(layer_id)).to(device)
    expected_gate_up = (adapter.num_experts, 2 * adapter.intermediate_size, gate_up.shape[-1])
    expected_down = (adapter.num_experts, down.shape[1], adapter.intermediate_size)
    if tuple(gate_up.shape) != expected_gate_up or tuple(down.shape) != expected_down:
        raise ValueError(
            f"Packed expert axes do not match adapter: gate_up={tuple(gate_up.shape)}, down={tuple(down.shape)}"
        )
    gate, up = gate_up.split(adapter.intermediate_size, dim=1)
    for expert_id in range(adapter.num_experts):
        yield expert_id, gate[expert_id], up[expert_id], down[expert_id]


def evidence_budget(n_eff: float, effective_rank: float, channel_count: int, config: NapsConfig) -> int:
    if n_eff < config.evidence_min or effective_rank < config.rank_min:
        return 0
    confidence = min(1.0, n_eff / config.evidence_saturation)
    confidence *= min(1.0, effective_rank / config.rank_saturation)
    maximum = round(config.replacement_fraction * channel_count)
    return int(maximum * confidence)


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def build_artifact_payload(
    orders: torch.Tensor,
    model_path: Path,
    retained_channels: int,
    block_size: int,
    method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if orders.ndim != 3:
        raise ValueError("orders must have shape [layers, experts, channels]")
    num_layers, num_experts, channel_count = map(int, orders.shape)
    if channel_count % block_size or retained_channels % block_size:
        raise ValueError("channel and retained widths must be block-aligned")
    block_count = channel_count // block_size
    retained_blocks = retained_channels // block_size
    block_scores = torch.zeros((num_experts, block_count), dtype=torch.float32)
    table: dict[int, dict[str, Any]] = {}
    for layer_id in range(num_layers):
        layer_orders = orders[layer_id].to(torch.long).cpu()
        ranked_position = torch.empty_like(layer_orders)
        ranked_position.scatter_(1, layer_orders, torch.arange(channel_count).expand_as(layer_orders))
        for expert_id in range(num_experts):
            block_scores[expert_id].scatter_add_(
                0,
                ranked_position[expert_id] // block_size,
                torch.ones(channel_count),
            )
        table[layer_id] = {
            "ranked_indices": layer_orders,
            "block_relative_scores": torch.ones((num_experts, block_count)),
            "block_coverage_scores": torch.full((num_experts, block_count), 1.0 / block_count),
            "block_sizes": torch.full((block_count,), block_size, dtype=torch.long),
            "intermediate_size": channel_count,
        }
    widths = torch.full((num_layers, num_experts), retained_blocks, dtype=torch.long)
    metadata = {
        "backbone": "stable_concat_aimer",
        "selection": "native_route_local_subset",
        "retained_channels": retained_channels,
    }
    channel = {
        "schema_version": 1,
        "purpose": f"{method}_channel_ranking",
        "model_path": str(model_path),
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": block_size,
        "table": table,
        "naps": metadata,
    }
    profile = {
        "schema_version": 1,
        "method": method,
        "mode": "stable_aimer_native_route_local_selection",
        "model_path": str(model_path),
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": list(range(num_layers)),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "num_blocks": block_count,
        "channel_block_size": block_size,
        "intermediate_size": channel_count,
        "allocation_scope": "per_expert_fixed",
        "total_blocks": int(widths.sum().item()),
        "maximum_blocks": int(widths.numel() * block_count),
        "target_pruning_ratio": 1.0 - retained_channels / channel_count,
        "actual_structural_pruning_ratio": 1.0 - retained_blocks / block_count,
        "profile_widths": widths,
        "naps": metadata,
    }
    return channel, profile


def build_layer(
    *,
    model_path: Path,
    weight_map: dict[str, str],
    adapter: PurePseudoModelAdapter,
    layer_id: int,
    retained_channels: int,
    config: NapsConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, dict[str, Any]], list[dict[str, Any]]]:
    router = load_tensor(model_path, weight_map, adapter.router_name(layer_id)).to(device)
    norm_name, norm = load_norm(model_path, weight_map, adapter, layer_id)
    probes = rms_norm_rows(router, norm.to(device), float(adapter.text_config["rms_norm_eps"]))
    logits, selected_experts, selected_weights = native_route(probes, router, adapter.router_top_k)
    top_plus_one = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :adapter.router_top_k + 1]
    sorted_logits = logits.gather(1, top_plus_one)
    margins = sorted_logits[:, adapter.router_top_k - 1] - sorted_logits[:, adapter.router_top_k]
    orders = []
    merge_plans: dict[int, dict[str, Any]] = {}
    diagnostics = []

    for expert_id, gate, up, down in iter_expert_weights(
        model_path, weight_map, adapter, layer_id, device
    ):
        zero_mask = effective_zero_mask(gate, up, down, config.effective_zero_threshold)
        prune_budget = adapter.intermediate_size - retained_channels
        zero_count = int(zero_mask.sum().item())
        forced_zero_retained = max(0, zero_count - prune_budget)
        aimer_scores = stable_concat_score(gate, up, down, config)
        aimer_order = torch.argsort(aimer_scores, descending=True, stable=True)
        routed_rows, routed_slots = torch.where(selected_experts == expert_id)
        routed_probes = probes.index_select(0, routed_rows)
        routed_weights = selected_weights[routed_rows, routed_slots]
        n_eff, effective_rank = effective_evidence(routed_probes, routed_weights)
        budget = evidence_budget(n_eff, effective_rank, adapter.intermediate_size, config)
        responses = swiglu_response(routed_probes, gate, up)
        order, mask_diagnostics = select_mask(
            aimer_order,
            aimer_scores,
            responses,
            down,
            routed_weights,
            zero_mask,
            retained_channels,
            budget,
            config,
        )
        retained = order[:retained_channels]
        displaced = mask_diagnostics.get("displaced", torch.empty(0, dtype=torch.long, device=device))
        plan = build_one_to_one_merge_plan(responses, down, retained, displaced, routed_weights, config)
        validated_plan, _ = validate_merge_plan(responses, down, retained, routed_weights, plan, config)
        merge_plans[expert_id] = validated_plan
        orders.append(order.cpu())
        diagnostics.append(
            {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "routed_probe_count": int(routed_rows.numel()),
                "effective_sample_size": n_eff,
                "effective_rank": effective_rank,
                "evidence_budget": budget,
                "effective_zero_count": zero_count,
                "forced_zero_retained": forced_zero_retained,
                "self_routed": bool((selected_experts[expert_id] == expert_id).any().item()),
                "topk_margin": float(margins[expert_id].item()),
                "mask": json_ready(mask_diagnostics),
                "merge": json_ready(validated_plan),
                "norm_tensor": norm_name,
            }
        )
        del gate, up, down, responses
    return torch.stack(orders), merge_plans, diagnostics


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    retained_channels = int(args.retained_channels)
    block_size = int(args.channel_block_size)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    if retained_channels % block_size or not 0 < retained_channels < adapter.intermediate_size:
        raise ValueError("retained_channels must be block-aligned and smaller than the source width")

    config = NapsConfig()
    device = torch.device(args.device)
    layer_orders = []
    merge_layers = {}
    diagnostics = []
    for layer_id in range(adapter.num_layers):
        orders, merge_plans, records = build_layer(
            model_path=model_path,
            weight_map=weight_map,
            adapter=adapter,
            layer_id=layer_id,
            retained_channels=retained_channels,
            config=config,
            device=device,
        )
        layer_orders.append(orders)
        merge_layers[layer_id] = merge_plans
        diagnostics.extend(records)
        print(f"Built NAPS layer {layer_id + 1}/{adapter.num_layers}", flush=True)

    all_orders = torch.stack(layer_orders)
    channel, profile = build_artifact_payload(
        all_orders, model_path, retained_channels, block_size, "naps_mask"
    )
    channel["naps"].update({"version": 1, "model_family": adapter.model_family, "config": vars(config)})
    profile["mode"] = "stable_aimer_native_route_local_selection"
    profile["naps"] = channel["naps"]
    profile_path = output_dir / "profile.pt"
    cache_path = output_dir / "rankings.pt"
    merge_path = output_dir / "merge_plan.pt"
    diagnostics_path = output_dir / "diagnostics.json"
    torch.save(channel, cache_path)
    profile["cache_provenance"] = {"channel": {"sha256": file_sha256(cache_path), "role": "naps_mask"}}
    torch.save(profile, profile_path)
    torch.save(
        {
            "schema_version": 1,
            "method": "naps_bounded_merge",
            "model_family": adapter.model_family,
            "retained_channels": retained_channels,
            "ranking_cache_sha256": file_sha256(cache_path),
            "layers": merge_layers,
        },
        merge_path,
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_family": adapter.model_family,
        "num_layers": adapter.num_layers,
        "num_experts": adapter.num_experts,
        "source_width": adapter.intermediate_size,
        "retained_channels": retained_channels,
        "router_top_k": adapter.router_top_k,
        "source_config_sha256": file_sha256(model_path / "config.json"),
        "ranking_cache_sha256": file_sha256(cache_path),
        "profile_sha256": file_sha256(profile_path),
        "merge_plan_sha256": file_sha256(merge_path),
        "records": diagnostics,
    }
    diagnostics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())