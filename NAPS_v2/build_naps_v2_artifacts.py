from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open

from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import (
    NapsV2Config,
    build_probe_sets,
    compensate_expert,
    effective_zero_mask,
    native_route,
    output_coverage,
    output_for_set,
    select_v2_mask,
    stable_concat_score,
    swiglu_response,
    weighted_output_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build independent NAPS-v2 artifacts.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retained-channels", type=int, required=True)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload["weight_map"].items()}


def load_tensor(model_path: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(f"Missing checkpoint tensor: {name}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def rms_norm_rows(rows: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = rows.float().square().mean(dim=-1, keepdim=True)
    return rows.float() * torch.rsqrt(variance + float(eps)) * norm_weight.float().unsqueeze(0)


def load_norm(model_path: Path, weight_map: dict[str, str], adapter: PurePseudoModelAdapter,
              layer_id: int) -> tuple[str, torch.Tensor]:
    for name in adapter.norm_names(layer_id):
        if name in weight_map:
            return name, load_tensor(model_path, weight_map, name)
    raise KeyError(f"No pre-MoE RMSNorm tensor found for layer {layer_id}")


def build_router_probe_route(
    model_path: Path,
    weight_map: dict[str, str],
    adapter: PurePseudoModelAdapter,
    layer_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    router = load_tensor(model_path, weight_map, adapter.router_name(layer_id)).to(device)
    if adapter.model_family != "gemma4":
        norm_name, norm = load_norm(model_path, weight_map, adapter, layer_id)
        route_probes = rms_norm_rows(router, norm.to(device), float(adapter.text_config["rms_norm_eps"]))
        logits, selected, weights = native_route(route_probes, router, adapter.router_top_k)
        return router, route_probes, route_probes, selected, weights, logits

    router_scale = load_tensor(model_path, weight_map, adapter.router_scale_name(layer_id)).to(device)
    per_expert_scale = load_tensor(
        model_path, weight_map, adapter.router_per_expert_scale_name(layer_id)
    ).to(device)
    route_probes = rms_norm_rows(
        router,
        torch.ones(router.shape[1], dtype=router.dtype, device=device),
        float(adapter.text_config["rms_norm_eps"]),
    )
    router_input = route_probes * router_scale.float() * (router.shape[1] ** -0.5)
    logits = router_input.float() @ router.float().transpose(0, 1)
    probabilities = torch.softmax(logits, dim=-1)
    top_probabilities, selected = torch.topk(probabilities, k=adapter.router_top_k, dim=-1)
    top_probabilities = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True)
    weights = top_probabilities * per_expert_scale.float()[selected]
    expert_norm = load_tensor(model_path, weight_map, adapter.expert_input_norm_name(layer_id)).to(device)
    expert_probes = rms_norm_rows(
        router,
        expert_norm,
        float(adapter.text_config["rms_norm_eps"]),
    )
    return router, route_probes, expert_probes, selected, weights, logits


def iter_expert_weights(
    model_path: Path, weight_map: dict[str, str], adapter: PurePseudoModelAdapter, layer_id: int, device: torch.device
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if adapter.expert_gate_template is not None:
        for expert_id in range(adapter.num_experts):
            yield (
                expert_id,
                load_tensor(model_path, weight_map, adapter.expert_gate_name(layer_id, expert_id)).to(device),
                load_tensor(model_path, weight_map, adapter.expert_up_name(layer_id, expert_id)).to(device),
                load_tensor(model_path, weight_map, adapter.expert_down_expert_name(layer_id, expert_id)).to(device),
            )
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


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def build_layer(
    *, model_path: Path, weight_map: dict[str, str], adapter: PurePseudoModelAdapter, layer_id: int,
    retained_channels: int, config: NapsV2Config, device: torch.device
) -> tuple[torch.Tensor, dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    router, route_probes, expert_probes, selected_experts, selected_weights, logits = build_router_probe_route(
        model_path, weight_map, adapter, layer_id, device
    )
    norm_name = "gemma4_router_norm_construction" if adapter.model_family == "gemma4" else "layer_norm"
    top_plus_one = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :adapter.router_top_k + 1]
    sorted_logits = logits.gather(1, top_plus_one)
    margins = sorted_logits[:, adapter.router_top_k
                            - 1] - sorted_logits[:, adapter.router_top_k
                                                 ] if adapter.router_top_k < adapter.num_experts else torch.full(
                                                     (route_probes.shape[0], ), float("inf"), device=device
                                                 )
    orders: list[torch.Tensor] = []
    compensation_layers: dict[int, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    audit = {
        "probes": expert_probes.detach().cpu(),
        "expert_probes": expert_probes.detach().cpu(),
        "selected_experts": selected_experts.detach().cpu(),
        "selected_weights": selected_weights.detach().cpu()
    }
    if adapter.model_family == "gemma4":
        audit["route_probes"] = route_probes.detach().cpu()

    for expert_id, gate, up, down in iter_expert_weights(model_path, weight_map, adapter, layer_id, device):
        probe_sets = build_probe_sets(expert_probes, selected_experts, selected_weights, expert_id)
        activation = str(adapter.text_config.get("hidden_activation", adapter.text_config.get("hidden_act", "silu")))
        coverage_responses = swiglu_response(
            probe_sets["coverage_probes"], gate, up, activation=activation
        )
        zero_mask = effective_zero_mask(gate, up, down, config.effective_zero_threshold)
        aimer_scores = stable_concat_score(gate, up, down, config)
        aimer_order = torch.argsort(aimer_scores, descending=True, stable=True)
        order, mask_info = select_v2_mask(
            aimer_order, aimer_scores, coverage_responses, zero_mask, retained_channels,
            int(probe_sets["native_rows"].numel()), config
        )
        retained = order[:retained_channels]
        coverage_before = output_coverage(coverage_responses, down, aimer_order[:retained_channels], zero_mask)
        coverage_after = output_coverage(coverage_responses, down, retained, zero_mask)
        compensation_down, compensation_info = compensate_expert(
            coverage_responses,
            down,
            retained,
            zero_mask,
            coverage_after["channel_output_mass"],
            int(probe_sets["native_rows"].numel()),
            config,
            native_responses=swiglu_response(
                probe_sets["native_probes"], gate, up, activation=activation
            ),
            native_weights=probe_sets["native_weights"],
        )
        compensation_layers[expert_id] = compensation_info
        orders.append(order.cpu())
        record = {
            "layer_id": layer_id,
            "expert_id": expert_id,
            "source_width": int(gate.shape[0]),
            "retained_width": retained_channels,
            "effective_zero_count": int(zero_mask.sum().item()),
            "forced_zero_retained": max(0,
                                        int(zero_mask.sum().item()) - (int(gate.shape[0]) - retained_channels)),
            "native_probe_count": int(probe_sets["native_rows"].numel()),
            "self_naturally_routed": probe_sets["self_naturally_routed"],
            "self_native_rank": int((torch.argsort(logits[int(expert_id)], descending=True, stable=True) == expert_id
                                     ).nonzero()[0].item() + 1),
            "anchor_added": probe_sets["anchor_added"],
            "coverage_probe_count": int(probe_sets["coverage_rows"].numel()),
            "native_topk_margin": float(margins[int(expert_id)].item()),
            "mask": mask_info,
            "coverage_before_swap": {
                key: value
                for key, value in coverage_before.items()
                if key != "channel_output_mass"
            },
            "coverage_after_swap": {
                key: value
                for key, value in coverage_after.items()
                if key != "channel_output_mass"
            },
            "compensation": compensation_info,
            "mask_uniform_loss": weighted_output_loss(
                coverage_responses.float() @ down.float().transpose(0, 1),
                output_for_set(coverage_responses, down, retained),
                None,
            ),
            "norm_tensor": norm_name,
        }
        diagnostics.append(json_ready(record))

    return torch.stack(orders), compensation_layers, diagnostics, audit


def build_artifact_payload(
    orders: torch.Tensor, model_path: Path, retained_channels: int, block_size: int, metadata: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    layers, experts, channels = map(int, orders.shape)
    if channels % block_size or retained_channels % block_size:
        raise ValueError("source and retained widths must be block-aligned")
    blocks = channels // block_size
    table = {}
    for layer_id in range(layers):
        ranked = orders[layer_id].to(torch.long).cpu()
        positions = torch.empty_like(ranked)
        positions.scatter_(1, ranked, torch.arange(channels).expand_as(ranked))
        table[layer_id] = {
            "ranked_indices": ranked,
            "block_relative_scores": torch.ones(experts, blocks),
            "block_coverage_scores": torch.full((experts, blocks), 1.0 / blocks),
            "block_sizes": torch.full((blocks, ), block_size, dtype=torch.long),
            "intermediate_size": channels
        }
    widths = torch.full((layers, experts), retained_channels // block_size, dtype=torch.long)
    channel = {
        "schema_version": 2,
        "purpose": "naps_v2_channel_ranking",
        "model_path": str(model_path),
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": block_size,
        "table": table,
        "naps": metadata
    }
    profile = {
        "schema_version": 2,
        "method": "naps_v2",
        "mode": "dynamic_pp_swap",
        "model_path": str(model_path),
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": list(range(layers)),
        "num_layers": layers,
        "num_experts": experts,
        "num_blocks": blocks,
        "channel_block_size": block_size,
        "intermediate_size": channels,
        "allocation_scope": "per_expert_fixed",
        "total_blocks": int(widths.sum().item()),
        "maximum_blocks": int(widths.numel() * blocks),
        "target_pruning_ratio": 1.0 - retained_channels / channels,
        "actual_structural_pruning_ratio": 1.0 - retained_channels / channels,
        "profile_widths": widths,
        "naps": metadata
    }
    return channel, profile


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    if not 0 < args.retained_channels < adapter.intermediate_size or args.retained_channels % args.channel_block_size:
        raise ValueError("retained-channels must be positive, smaller than source width, and block-aligned")
    config = NapsV2Config()
    device = torch.device(args.device)
    layer_orders = []
    compensation = {}
    records = []
    audits = {}
    metadata = {
        "version": 2,
        "model_family": adapter.model_family,
        "backbone": "stable_concat_aimer",
        "selection": "dynamic_3_to_8_percent_pp_swap",
        "compensation": "full_pruned_set_sparse_ridge",
        "retained_channels": args.retained_channels,
        "config": vars(config)
    }
    for layer_id in range(adapter.num_layers):
        orders, layer_comp, layer_records, audit = build_layer(
            model_path=model_path,
            weight_map=weight_map,
            adapter=adapter,
            layer_id=layer_id,
            retained_channels=args.retained_channels,
            config=config,
            device=device
        )
        layer_orders.append(orders)
        compensation[layer_id] = layer_comp
        records.extend(layer_records)
        audits[layer_id] = audit
        print(f"Built NAPS-v2 layer {layer_id + 1}/{adapter.num_layers}", flush=True)
    rankings, profile = build_artifact_payload(
        torch.stack(layer_orders), model_path, args.retained_channels, args.channel_block_size, metadata
    )
    rankings_path = output_dir / "rankings.pt"
    profile_path = output_dir / "profile.pt"
    compensation_path = output_dir / "compensation_plan.pt"
    audit_path = output_dir / "routing_audit.pt"
    diagnostics_path = output_dir / "diagnostics.json"
    torch.save(rankings, rankings_path)
    profile["cache_provenance"] = {"channel_sha256": file_sha256(rankings_path)}
    torch.save(profile, profile_path)
    torch.save({
        "schema_version": 2,
        "method": "naps_v2_expertcomp",
        "retained_channels": args.retained_channels,
        "layers": compensation
    }, compensation_path)
    torch.save({"schema_version": 2, "layers": audits}, audit_path)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": 2,
        "model_path": str(model_path),
        "model_family": adapter.model_family,
        "num_layers": adapter.num_layers,
        "num_experts": adapter.num_experts,
        "source_width": adapter.intermediate_size,
        "retained_channels": args.retained_channels,
        "router_top_k": adapter.router_top_k,
        "ranking_cache_sha256": file_sha256(rankings_path),
        "records": records
    }
    diagnostics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
