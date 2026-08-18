from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_naps_v2_artifacts import (
    build_router_probe_route,
    file_sha256,
    iter_expert_weights,
    load_norm,
    load_tensor,
    load_weight_map,
    rms_norm_rows,
)
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import (
    NapsV2Config,
    build_probe_sets,
    effective_zero_mask,
    native_route,
    select_v2_mask,
    stable_concat_score,
    swiglu_response,
)


WIDTH_PROFILES: dict[str, tuple[int, int, int]] = {
    "qwen3": (256, 384, 512),
    "qwen3.6": (192, 256, 320),
    "gemma4": (224, 352, 480),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a minimal NAPS-v2 heterogeneous expert-AIMER profile."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def expert_aimer_score(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Compute the original weight-only AIMER score for one expert."""

    if gate.ndim != 2 or up.shape != gate.shape or down.ndim != 2 or down.shape[1] != gate.shape[0]:
        raise ValueError("gate, up, and down tensors are not channel-aligned")
    tensors = (gate.float(), up.float(), down.float())
    numel = sum(tensor.numel() for tensor in tensors)
    l2_sq = sum(tensor.square().sum() for tensor in tensors)
    if numel <= 0 or float(l2_sq.item()) <= 0.0:
        return torch.zeros((), dtype=torch.float32, device=gate.device)
    absolute_mean = sum(tensor.abs().sum() for tensor in tensors) / numel
    root_mean_square = torch.sqrt(l2_sq / numel)
    return absolute_mean / root_mean_square


def assign_expert_widths(
    expert_scores: torch.Tensor,
    widths: tuple[int, int, int],
) -> torch.Tensor:
    """Assign smaller widths to experts with higher, more removable AIMER scores."""

    if expert_scores.ndim != 1 or expert_scores.numel() < 4:
        raise ValueError("expert_scores must contain at least four experts")
    small_width, medium_width, large_width = (int(value) for value in widths)
    if not small_width < medium_width < large_width:
        raise ValueError("widths must be strictly increasing")
    if medium_width - small_width != large_width - medium_width:
        raise ValueError("widths must be symmetric around the medium width")
    expert_count = int(expert_scores.numel())
    quarter = expert_count // 4
    if quarter <= 0 or expert_count % 4:
        raise ValueError("expert count must be divisible by four")
    order = torch.argsort(expert_scores.float(), descending=True, stable=True)
    assigned = torch.full_like(order, medium_width, dtype=torch.long)
    assigned[order[:quarter]] = small_width
    assigned[order[-quarter:]] = large_width
    return assigned


def assign_expert_widths_adaptive(
    expert_scores: torch.Tensor,
    widths: tuple[int, int, int],
) -> torch.Tensor:
    """Assign widths using a balanced minimum-variance partition of AIMER scores."""

    if expert_scores.ndim != 1 or expert_scores.numel() < 4:
        raise ValueError("expert_scores must contain at least four experts")
    small_width, medium_width, large_width = (int(value) for value in widths)
    if not small_width < medium_width < large_width:
        raise ValueError("widths must be strictly increasing")
    if medium_width - small_width != large_width - medium_width:
        raise ValueError("widths must be symmetric around the medium width")
    expert_count = int(expert_scores.numel())
    order = torch.argsort(expert_scores.float(), descending=True, stable=True)
    sorted_scores = expert_scores.float().index_select(0, order)

    best_tail_count = 1
    best_loss = torch.tensor(float("inf"), device=sorted_scores.device)
    for tail_count in range(1, expert_count // 2):
        groups = (
            sorted_scores[:tail_count],
            sorted_scores[tail_count:-tail_count],
            sorted_scores[-tail_count:],
        )
        loss = sum((group - group.mean()).square().sum() for group in groups)
        if bool(loss < best_loss):
            best_loss = loss
            best_tail_count = tail_count

    assigned = torch.full_like(order, medium_width, dtype=torch.long)
    assigned[order[:best_tail_count]] = small_width
    assigned[order[-best_tail_count:]] = large_width
    return assigned


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def build_layer(
    *,
    model_path: Path,
    weight_map: dict[str, str],
    adapter: PurePseudoModelAdapter,
    layer_id: int,
    widths: tuple[int, int, int],
    config: NapsV2Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    router, route_probes, expert_probes, selected_experts, selected_weights, logits = build_router_probe_route(
        model_path, weight_map, adapter, layer_id, device
    )
    norm_name = "gemma4_router_norm_construction" if adapter.model_family == "gemma4" else "layer_norm"
    width_orders: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    widths_by_expert: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []

    expert_payloads = []
    for expert_id, gate, up, down in iter_expert_weights(model_path, weight_map, adapter, layer_id, device):
        probe_sets = build_probe_sets(expert_probes, selected_experts, selected_weights, expert_id)
        activation = str(adapter.text_config.get("hidden_activation", adapter.text_config.get("hidden_act", "silu")))
        coverage_responses = swiglu_response(
            probe_sets["coverage_probes"], gate, up, activation=activation
        )
        zero_mask = effective_zero_mask(gate, up, down, config.effective_zero_threshold)
        aimer_scores = stable_concat_score(gate, up, down, config)
        expert_score = expert_aimer_score(gate, up, down)
        scores.append(expert_score)
        expert_payloads.append((expert_id, gate, up, down, probe_sets, coverage_responses, zero_mask, aimer_scores))

    expert_scores = torch.stack(scores)
    assigned_widths = assign_expert_widths(expert_scores, widths)
    for expert_id, gate, up, down, probe_sets, coverage_responses, zero_mask, aimer_scores in expert_payloads:
        aimer_order = torch.argsort(aimer_scores, descending=True, stable=True)
        orders_for_expert = []
        for width in widths:
            order, _ = select_v2_mask(
                aimer_order,
                aimer_scores,
                coverage_responses,
                zero_mask,
                width,
                int(probe_sets["native_rows"].numel()),
                config,
            )
            orders_for_expert.append(order.cpu())
        width_orders.append(torch.stack(orders_for_expert))
        assigned_width = int(assigned_widths[expert_id].item())
        diagnostics.append(
            {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "expert_aimer_score": float(expert_scores[expert_id].item()),
                "assigned_width": assigned_width,
                "aimer_removability_rank": int(
                    (torch.argsort(expert_scores, descending=True, stable=True) == expert_id).nonzero()[0].item() + 1
                ),
                "native_probe_count": int(probe_sets["native_rows"].numel()),
                "effective_zero_count": int(zero_mask.sum().item()),
                "anchor_added": bool(probe_sets["anchor_added"]),
                "norm_tensor": norm_name,
            }
        )

    audit = {
        "probes": expert_probes.detach().cpu(),
        "selected_experts": selected_experts.detach().cpu(),
        "selected_weights": selected_weights.detach().cpu(),
    }
    audit["route_probes"] = route_probes.detach().cpu()
    audit["expert_probes"] = expert_probes.detach().cpu()
    return (
        torch.stack(width_orders),
        expert_scores.cpu(),
        assigned_widths.cpu(),
        diagnostics,
        audit,
    )


def build_payload(
    *,
    model_path: Path,
    adapter: PurePseudoModelAdapter,
    widths: tuple[int, int, int],
    orders_by_layer: list[torch.Tensor],
    scores_by_layer: list[torch.Tensor],
    profile_widths: torch.Tensor,
    block_size: int,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_width = adapter.intermediate_size
    padded_width = max(widths)
    if source_width % block_size or padded_width % block_size:
        raise ValueError("source and padded widths must be block-aligned")
    table = {}
    for layer_id, (orders, scores) in enumerate(zip(orders_by_layer, scores_by_layer)):
        table[layer_id] = {
            "ranked_indices_by_width": orders,
            "width_options": torch.tensor(widths, dtype=torch.long),
            "expert_aimer_scores": scores,
            "intermediate_size": source_width,
        }
    channel = {
        "schema_version": 3,
        "purpose": "naps_v2_heterogeneous_expert_aimer_mask",
        "model_path": str(model_path),
        "test_metrics_used": False,
        "channel_block_size": block_size,
        "source_intermediate_size": source_width,
        "padded_intermediate_size": padded_width,
        "table": table,
        "naps": metadata,
    }
    profile = {
        "schema_version": 3,
        "method": "naps_v2_heterogeneous_mask_padded",
        "mode": "expert_aimer_quartiles_padded_homogeneous",
        "model_path": str(model_path),
        "profile_construction": "calibration_free",
        "test_metrics_used_for_profile": False,
        "layer_ids": list(range(adapter.num_layers)),
        "num_layers": adapter.num_layers,
        "num_experts": adapter.num_experts,
        "source_intermediate_size": source_width,
        "intermediate_size": padded_width,
        "padded_intermediate_size": padded_width,
        "num_blocks": padded_width // block_size,
        "channel_block_size": block_size,
        "width_options": torch.tensor(widths, dtype=torch.long),
        "profile_widths": profile_widths // block_size,
        "allocation_scope": "per_layer_expert_aimer_quartiles",
        "total_blocks": int((profile_widths // block_size).sum().item()),
        "maximum_blocks": int(profile_widths.numel() * padded_width // block_size),
        "padding_is_structural_zero": True,
        "naps": metadata,
    }
    return channel, profile


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    widths = WIDTH_PROFILES.get(adapter.model_family)
    if widths is None:
        raise ValueError(f"No first-version heterogeneous width profile for {adapter.model_family}")
    block_size = int(args.channel_block_size)
    if any(width <= 0 or width >= adapter.intermediate_size or width % block_size for width in widths):
        raise ValueError("heterogeneous widths must be positive, smaller than source width, and block-aligned")
    device = torch.device(args.device)
    config = NapsV2Config()
    orders_by_layer = []
    scores_by_layer = []
    assigned_by_layer = []
    records = []
    audits = {}
    metadata = {
        "version": 3,
        "model_family": adapter.model_family,
        "backbone": "stable_concat_aimer",
        "selection": "dynamic_pp_swap_per_candidate_width",
        "expert_importance": "original_weight_only_aimer",
        "width_assignment": "per_layer_high_aimer_small_low_aimer_large",
        "width_options": widths,
        "padding": "zero_to_layer_max_width",
        "retained_channels": widths[1],
    }
    for layer_id in range(adapter.num_layers):
        layer_orders, layer_scores, layer_widths, layer_records, audit = build_layer(
            model_path=model_path,
            weight_map=weight_map,
            adapter=adapter,
            layer_id=layer_id,
            widths=widths,
            config=config,
            device=device,
        )
        orders_by_layer.append(layer_orders)
        scores_by_layer.append(layer_scores)
        assigned_by_layer.append(layer_widths)
        records.extend(layer_records)
        audits[layer_id] = audit
        print(f"Built NAPS-v2 heterogeneous layer {layer_id + 1}/{adapter.num_layers}", flush=True)
    profile_widths = torch.stack(assigned_by_layer)
    rankings, profile = build_payload(
        model_path=model_path,
        adapter=adapter,
        widths=widths,
        orders_by_layer=orders_by_layer,
        scores_by_layer=scores_by_layer,
        profile_widths=profile_widths,
        block_size=block_size,
        metadata=metadata,
    )
    rankings_path = output_dir / "rankings.pt"
    profile_path = output_dir / "profile.pt"
    audit_path = output_dir / "routing_audit.pt"
    diagnostics_path = output_dir / "diagnostics.json"
    torch.save(rankings, rankings_path)
    profile["cache_provenance"] = {"channel_sha256": file_sha256(rankings_path)}
    torch.save(profile, profile_path)
    torch.save({"schema_version": 3, "layers": audits}, audit_path)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": 3,
        "model_path": str(model_path),
        "model_family": adapter.model_family,
        "num_layers": adapter.num_layers,
        "num_experts": adapter.num_experts,
        "source_width": adapter.intermediate_size,
        "padded_width": max(widths),
        "width_options": widths,
        "profile_widths": profile_widths.tolist(),
        "ranking_cache_sha256": file_sha256(rankings_path),
        "records": records,
    }
    diagnostics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
