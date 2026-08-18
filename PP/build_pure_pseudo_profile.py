from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

from src.channel_runtime import _build_layer_channel_table_from_raw_scores, channel_table_to_payload
from WICK.build_wick_profile import file_sha256, pseudo_protection_importance, rms_norm_rows, router_gram_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Pure-Pseudo channel profile from router-derived probes.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float, required=True)
    parser.add_argument("--router-neighbors", type=int, default=8)
    parser.add_argument("--top-q", type=int, default=4)
    parser.add_argument("--probe-signs", choices=("positive", "positive-negative"), default="positive")
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def pure_pseudo_priority(pseudo_scores: torch.Tensor) -> torch.Tensor:
    """Return the channel priority determined only by pseudo-probe scores."""

    if pseudo_scores.ndim != 1:
        raise ValueError("pseudo_scores must be a one-dimensional tensor.")
    if not torch.isfinite(pseudo_scores).all():
        raise ValueError("pseudo_scores must contain only finite values.")
    return pseudo_scores.float()


def expand_probe_signs(probes: torch.Tensor, probe_signs: str) -> torch.Tensor:
    """Return the requested signed probe set without changing positive-only behavior."""

    if probe_signs == "positive":
        return probes
    if probe_signs == "positive-negative":
        return torch.cat((probes, -probes), dim=0)
    raise ValueError("probe_signs must be 'positive' or 'positive-negative'.")


def build_pure_pseudo_artifacts(
    *,
    model_path: Path,
    priorities_by_layer: dict[int, torch.Tensor],
    target_pruning_ratio: float,
    router_neighbors: int,
    top_q: int,
    probe_signs: str,
    block_size: int,
    checkpoint_identity: dict[str, object],
) -> tuple[dict, dict]:
    if not priorities_by_layer:
        raise ValueError("Pure-Pseudo priorities must be non-empty.")
    if not 0.0 <= float(target_pruning_ratio) <= 1.0:
        raise ValueError("target_pruning_ratio must be in [0, 1].")
    layer_ids = sorted(priorities_by_layer)
    shapes = {tuple(priorities_by_layer[layer_id].shape) for layer_id in layer_ids}
    if len(shapes) != 1:
        raise ValueError("all Pure-Pseudo priority layers must have the same shape.")
    num_experts, intermediate_size = next(iter(shapes))
    tables = {
        layer_id: _build_layer_channel_table_from_raw_scores(priorities_by_layer[layer_id], int(block_size))
        for layer_id in layer_ids
    }
    num_blocks = int(tables[layer_ids[0]].block_sizes.numel())
    retained_blocks = int(round(num_blocks * (1.0 - float(target_pruning_ratio))))
    widths = torch.full((len(layer_ids), num_experts), retained_blocks, dtype=torch.long)
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    pseudo_metadata = {
        "ranking_criterion": "top_q_mean_absolute_swiglu_response_times_down_l2",
        "probe_source": "rmsnorm_router_self_plus_cosine_topk",
        "router_neighbors": int(router_neighbors),
        "top_q": int(top_q),
        "probe_signs": probe_signs,
        "checkpoint_identity": checkpoint_identity,
    }
    channel_payload = {
        "schema_version": 1,
        "purpose": "pure_pseudo_channel_ranking",
        "model_path": str(model_path),
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": int(block_size),
        "table": channel_table_to_payload(tables),
        "pure_pseudo": pseudo_metadata,
    }
    profile = {
        "schema_version": 1,
        "method": "pure_pseudo",
        "mode": "router_gram_pure_pseudo_channel_ranking",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": len(layer_ids),
        "num_experts": int(num_experts),
        "num_blocks": num_blocks,
        "channel_block_size": int(block_size),
        "intermediate_size": int(intermediate_size),
        "allocation_scope": "per_expert_fixed",
        "target_blocks_by_layer": widths.sum(dim=1).tolist(),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "total_blocks": total_blocks,
        "maximum_blocks": maximum_blocks,
        "target_pruning_ratio": float(target_pruning_ratio),
        "actual_structural_pruning_ratio": 1.0 - total_blocks / maximum_blocks,
        "retained_expert_mask": None,
        "profile_widths": widths,
        "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
        "pure_pseudo": pseudo_metadata,
    }
    return channel_payload, profile


def _load_model_config(model_path: Path) -> dict:
    payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if payload.get("model_type") != "qwen3_moe":
        raise ValueError("Pure-Pseudo currently supports Qwen3 MoE checkpoints only.")
    return payload


def _load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json is missing weight_map.")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def _load_tensor(model_path: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(f"Missing checkpoint tensor: {name}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _load_first_tensor(model_path: Path, weight_map: dict[str, str], names: list[str]) -> torch.Tensor:
    for name in names:
        if name in weight_map:
            return _load_tensor(model_path, weight_map, name)
    raise KeyError(f"Missing checkpoint tensor; tried: {names}")


def collect_pure_pseudo_priorities(
    *,
    model_path: Path,
    config: dict,
    router_neighbors: int,
    top_q: int,
    probe_signs: str,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    weight_map = _load_weight_map(model_path)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    priorities: dict[int, torch.Tensor] = {}
    for layer_id in range(num_layers):
        layer_prefix = f"model.layers.{layer_id}"
        router = _load_tensor(model_path, weight_map, f"{layer_prefix}.mlp.gate.weight").to(device=device)
        norm_weight = _load_first_tensor(
            model_path,
            weight_map,
            [
                f"{layer_prefix}.post_attention_layernorm.weight",
                f"{layer_prefix}.pre_feedforward_layernorm.weight",
                f"{layer_prefix}.input_layernorm.weight",
            ],
        ).to(device=device)
        neighbor_ids = router_gram_neighbors(router, router_neighbors)
        normalized_router = rms_norm_rows(router, norm_weight, float(config["rms_norm_eps"]))
        layer_priorities = []
        for expert_id in range(num_experts):
            expert_prefix = f"{layer_prefix}.mlp.experts.{expert_id}"
            gate = _load_tensor(model_path, weight_map, f"{expert_prefix}.gate_proj.weight").to(device=device)
            up = _load_tensor(model_path, weight_map, f"{expert_prefix}.up_proj.weight").to(device=device)
            down = _load_tensor(model_path, weight_map, f"{expert_prefix}.down_proj.weight").to(device=device)
            probes = normalized_router.index_select(0, neighbor_ids[expert_id])
            probes = expand_probe_signs(probes, probe_signs)
            pseudo_scores = pseudo_protection_importance(probes, gate, up, down, min(top_q, probes.shape[0]))
            layer_priorities.append(pure_pseudo_priority(pseudo_scores).cpu())
            del gate, up, down
        priorities[layer_id] = torch.stack(layer_priorities)
        print(f"Scored Pure-Pseudo layer {layer_id + 1}/{num_layers}", flush=True)
    return priorities


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    config = _load_model_config(model_path)
    priorities = collect_pure_pseudo_priorities(
        model_path=model_path,
        config=config,
        router_neighbors=int(args.router_neighbors),
        top_q=int(args.top_q),
        probe_signs=args.probe_signs,
        device=torch.device(args.device),
    )
    index_path = model_path / "model.safetensors.index.json"
    channel, profile = build_pure_pseudo_artifacts(
        model_path=model_path,
        priorities_by_layer=priorities,
        target_pruning_ratio=float(args.target_pruning_ratio),
        router_neighbors=int(args.router_neighbors),
        top_q=int(args.top_q),
        probe_signs=args.probe_signs,
        block_size=int(args.channel_block_size),
        checkpoint_identity={
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(index_path),
        },
    )
    args.output_channel_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel, args.output_channel_cache)
    profile["cache_provenance"] = {
        "channel": {"sha256": file_sha256(args.output_channel_cache), "role": "pure_pseudo_channel_ranking"}
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output_profile)
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["width_histogram"] = {
        str(int(width)): int(count) for width, count in zip(*torch.unique(profile["profile_widths"], return_counts=True))
    }
    args.output_profile.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output_channel_cache.resolve())
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())