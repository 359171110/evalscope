from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Mapping

import torch


def build_reap_profile_payload(
    *,
    observer_data: Mapping[int, Mapping[str, object]],
    model_path: str,
    calibration_payload: Mapping[str, object],
    calibration_file_sha256: str,
    channel_file_sha256: str,
    official_reap_commit: str,
    num_blocks: int,
    experts_to_prune_per_layer: int,
    top_k: int,
    renormalize_router_weights: bool,
) -> dict:
    """Convert official REAP saliency into a whole-expert runtime profile."""

    if not renormalize_router_weights:
        raise ValueError("official REAP protocol requires renormalize_router_weights=true.")
    if calibration_payload.get("split") != "train":
        raise ValueError("REAP profile calibration must use the train split.")
    if calibration_payload.get("frozen_before_profile") is not True:
        raise ValueError("REAP calibration artifact must be frozen before profile construction.")
    if calibration_payload.get("test_metrics_used") is not False:
        raise ValueError("REAP calibration artifact must not use test metrics.")
    blocks = int(num_blocks)
    prune_count = int(experts_to_prune_per_layer)
    routed_top_k = int(top_k)
    if blocks <= 0 or prune_count < 0 or routed_top_k <= 0:
        raise ValueError("REAP topology parameters are invalid.")
    layer_ids = sorted(int(layer_id) for layer_id in observer_data)
    if not layer_ids:
        raise ValueError("official REAP observer data contains no MoE layers.")
    saliency_rows = []
    retained_rows = []
    widths_rows = []
    expert_count = None
    for layer_id in layer_ids:
        values = observer_data[layer_id].get("reap")
        if not isinstance(values, torch.Tensor) or values.ndim != 1:
            raise ValueError(f"layer {layer_id} has no one-dimensional official REAP saliency.")
        values = values.detach().float().cpu()
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"layer {layer_id} REAP saliency must be finite.")
        if expert_count is None:
            expert_count = int(values.numel())
        elif expert_count != int(values.numel()):
            raise ValueError("REAP observer layers must have a uniform expert count.")
        if prune_count > expert_count - routed_top_k:
            raise ValueError("REAP pruning must retain at least top_k experts per layer.")
        retained = torch.ones(expert_count, dtype=torch.bool)
        if prune_count:
            experts_to_prune = torch.topk(values, prune_count, largest=False).indices
            retained[experts_to_prune] = False
        widths = retained.to(torch.long) * blocks
        saliency_rows.append(values)
        retained_rows.append(retained)
        widths_rows.append(widths)
    saliency = torch.stack(saliency_rows)
    retained_mask = torch.stack(retained_rows)
    widths = torch.stack(widths_rows)
    retained_by_layer = retained_mask.sum(dim=1).tolist()
    actual_blocks_by_layer = widths.sum(dim=1).tolist()
    maximum_blocks = int(widths.numel()) * blocks
    total_blocks = int(widths.sum().item())
    return {
        "schema_version": 1,
        "method": "official_reap",
        "mode": "official_reap_whole_expert",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "dataset": calibration_payload.get("dataset"),
        "calibration_split": "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": len(layer_ids),
        "num_experts": int(expert_count),
        "num_blocks": blocks,
        "top_k": routed_top_k,
        "allocation_scope": "per_layer",
        "experts_to_prune_per_layer": prune_count,
        "retained_experts_by_layer": retained_by_layer,
        "target_blocks_by_layer": actual_blocks_by_layer,
        "actual_blocks_by_layer": actual_blocks_by_layer,
        "total_blocks": total_blocks,
        "maximum_blocks": maximum_blocks,
        "target_pruning_ratio": 1.0 - total_blocks / maximum_blocks,
        "actual_structural_pruning_ratio": 1.0 - total_blocks / maximum_blocks,
        "retained_expert_mask": retained_mask,
        "profile_widths": widths,
        "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
        "official_reap_saliency": saliency,
        "observer": {
            "metric": "reap",
            "formula": "mean_active_token(router_weight * expert_output_l2)",
            "conditional_on_expert_active": True,
            "renormalize_router_weights": True,
            "official_reap_commit": official_reap_commit,
        },
        "cache_provenance": {
            "channel": {
                "sha256": channel_file_sha256,
                "role": "runtime_topology_only",
            },
            "calibration": {
                "sha256": calibration_file_sha256,
                "protocol_name": calibration_payload.get("protocol_name"),
                "input_ids_sha256": calibration_payload.get("input_ids_sha256"),
                "sequence_length": calibration_payload.get("sequence_length"),
                "calibration_sequences": calibration_payload.get("calibration_sequences"),
                "calibration_tokens": calibration_payload.get("calibration_tokens"),
                "split": calibration_payload.get("split"),
            }
        },
    }