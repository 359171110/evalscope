from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from src.channel_runtime import _build_layer_channel_table_from_raw_scores, channel_table_to_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a WICK Gram-guided pseudo-protected channel profile.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float, required=True)
    parser.add_argument("--protection-ratio", type=float, default=0.10)
    parser.add_argument("--router-neighbors", type=int, default=8)
    parser.add_argument("--top-q", type=int, default=4)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_path_importance(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Return the geometric mean of aligned gate, up, and down path norms."""

    if gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("gate/up weights must have the same two-dimensional shape.")
    if down_weight.ndim != 2 or int(down_weight.shape[1]) != int(gate_weight.shape[0]):
        raise ValueError("down weight columns must align with gate/up output channels.")
    gate_norm = torch.linalg.vector_norm(gate_weight.float(), dim=1)
    up_norm = torch.linalg.vector_norm(up_weight.float(), dim=1)
    down_norm = torch.linalg.vector_norm(down_weight.float(), dim=0)
    return (gate_norm * up_norm * down_norm).clamp_min(eps).pow(1.0 / 3.0)


def router_gram_neighbors(router_weight: torch.Tensor, neighbor_count: int, eps: float = 1.0e-12) -> torch.Tensor:
    """Return self plus the most cosine-similar router rows for every expert."""

    if router_weight.ndim != 2:
        raise ValueError("router_weight must have shape [experts, hidden_size].")
    num_experts = int(router_weight.shape[0])
    neighbors = int(neighbor_count)
    if not 0 <= neighbors < num_experts:
        raise ValueError("neighbor_count must be in [0, num_experts).")
    normalized = F.normalize(router_weight.float(), p=2, dim=1, eps=eps)
    gram = normalized @ normalized.transpose(0, 1)
    gram.fill_diagonal_(-torch.inf)
    adjacent = torch.topk(gram, k=neighbors, dim=1, largest=True, sorted=True).indices
    self_ids = torch.arange(num_experts, device=router_weight.device).unsqueeze(1)
    return torch.cat((self_ids, adjacent), dim=1)


def rms_norm_rows(rows: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    if rows.ndim != 2 or norm_weight.ndim != 1 or int(rows.shape[1]) != int(norm_weight.numel()):
        raise ValueError("rows and norm_weight have incompatible RMSNorm shapes.")
    variance = rows.float().square().mean(dim=-1, keepdim=True)
    return rows.float() * torch.rsqrt(variance + float(eps)) * norm_weight.float().unsqueeze(0)


def pseudo_protection_importance(
    probes: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    top_q: int,
    use_down_proj_norm: bool = True,
) -> torch.Tensor:
    """Score channel output contributions on the strongest selected pseudo probes."""

    probe_count = int(probes.shape[0])
    selected = int(top_q)
    if not 1 <= selected <= probe_count:
        raise ValueError("top_q must be in [1, number of probes].")
    gate_hidden = F.linear(probes.float(), gate_weight.float())
    up_hidden = F.linear(probes.float(), up_weight.float())
    activation = F.silu(gate_hidden) * up_hidden
    contribution = activation.abs()
    if use_down_proj_norm:
        down_norm = torch.linalg.vector_norm(down_weight.float(), dim=0)
        contribution = contribution * down_norm.unsqueeze(0)
    return torch.topk(contribution, k=selected, dim=0, largest=True, sorted=False).values.mean(dim=0)


def combine_wick_priority(
    weight_scores: torch.Tensor,
    pseudo_scores: torch.Tensor,
    *,
    protection_ratio: float,
    retained_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place pseudo-protected channels before the weight-only retention ranking."""

    if weight_scores.ndim != 1 or pseudo_scores.shape != weight_scores.shape:
        raise ValueError("weight_scores and pseudo_scores must be aligned one-dimensional tensors.")
    ratio = float(protection_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("protection_ratio must be in [0, 1].")
    channel_count = int(weight_scores.numel())
    retained = int(retained_channels)
    if not 0 <= retained <= channel_count:
        raise ValueError("retained_channels must be in [0, channel_count].")
    protected_count = int(round(ratio * channel_count))
    if protected_count > retained:
        raise ValueError("protected channels cannot exceed retained channels.")

    protected = torch.zeros(channel_count, dtype=torch.bool, device=weight_scores.device)
    if protected_count:
        protected_indices = torch.topk(pseudo_scores.float(), k=protected_count, largest=True, sorted=True).indices
        protected[protected_indices] = True
        protected_order = protected_indices[torch.argsort(pseudo_scores[protected_indices], descending=True, stable=True)]
    else:
        protected_order = torch.empty(0, dtype=torch.long, device=weight_scores.device)
    unprotected_indices = torch.nonzero(~protected, as_tuple=False).flatten()
    unprotected_order = unprotected_indices[
        torch.argsort(weight_scores[unprotected_indices], descending=True, stable=True)
    ]
    order = torch.cat((protected_order, unprotected_order))
    priority = torch.empty(channel_count, dtype=torch.float32, device=weight_scores.device)
    priority[order] = torch.arange(channel_count, 0, -1, dtype=torch.float32, device=weight_scores.device)
    return priority, protected


def build_wick_artifacts(
    *,
    model_path: Path,
    priorities_by_layer: dict[int, torch.Tensor],
    protected_by_layer: dict[int, torch.Tensor],
    target_pruning_ratio: float,
    protection_ratio: float,
    router_neighbors: int,
    top_q: int,
    block_size: int,
    checkpoint_identity: dict[str, object],
) -> tuple[dict, dict]:
    if not priorities_by_layer or set(priorities_by_layer) != set(protected_by_layer):
        raise ValueError("priority and protection layers must be non-empty and aligned.")
    if not 0.0 <= float(target_pruning_ratio) <= 1.0:
        raise ValueError("target_pruning_ratio must be in [0, 1].")
    layer_ids = sorted(priorities_by_layer)
    shapes = {tuple(priorities_by_layer[layer_id].shape) for layer_id in layer_ids}
    if len(shapes) != 1:
        raise ValueError("all WICK priority layers must have the same shape.")
    num_experts, intermediate_size = next(iter(shapes))
    if any(protected_by_layer[layer_id].shape != priorities_by_layer[layer_id].shape for layer_id in layer_ids):
        raise ValueError("protection masks must match WICK priority shapes.")
    tables = {
        layer_id: _build_layer_channel_table_from_raw_scores(priorities_by_layer[layer_id], int(block_size))
        for layer_id in layer_ids
    }
    num_blocks = int(tables[layer_ids[0]].block_sizes.numel())
    retained_blocks = int(round(num_blocks * (1.0 - float(target_pruning_ratio))))
    widths = torch.full((len(layer_ids), num_experts), retained_blocks, dtype=torch.long)
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    protected_counts = torch.stack(
        [protected_by_layer[layer_id].to(torch.long).sum(dim=1).cpu() for layer_id in layer_ids]
    )
    channel_payload = {
        "schema_version": 1,
        "purpose": "wick_gram_protected_channel_ranking",
        "model_path": str(model_path),
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": int(block_size),
        "table": channel_table_to_payload(tables),
        "wick": {
            "protected_counts": protected_counts,
            "protection_ratio": float(protection_ratio),
            "router_neighbors": int(router_neighbors),
            "top_q": int(top_q),
        },
    }
    profile = {
        "schema_version": 1,
        "method": "wick_gram_protect",
        "mode": "weight_path_router_gram_pseudo_protection",
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
        "wick": {
            "weight_criterion": "geometric_mean_l2_gate_up_down",
            "probe_source": "rmsnorm_router_self_plus_cosine_topk",
            "protection_criterion": "top_q_mean_absolute_swiglu_response_times_down_l2",
            "protection_ratio": float(protection_ratio),
            "router_neighbors": int(router_neighbors),
            "top_q": int(top_q),
            "checkpoint_identity": checkpoint_identity,
        },
    }
    return channel_payload, profile


def _load_model_config(model_path: Path) -> dict:
    payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if payload.get("model_type") != "qwen3_moe":
        raise ValueError("WICK currently supports Qwen3 MoE checkpoints only.")
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


def collect_wick_priorities(
    *,
    model_path: Path,
    config: dict,
    protection_ratio: float,
    router_neighbors: int,
    top_q: int,
    target_pruning_ratio: float,
    block_size: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    weight_map = _load_weight_map(model_path)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    intermediate_size = int(config["moe_intermediate_size"])
    retained_blocks = int(round(math.ceil(intermediate_size / block_size) * (1.0 - target_pruning_ratio)))
    retained_channels = min(intermediate_size, retained_blocks * int(block_size))
    priorities: dict[int, torch.Tensor] = {}
    protected_masks: dict[int, torch.Tensor] = {}
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
        layer_protected = []
        for expert_id in range(num_experts):
            expert_prefix = f"{layer_prefix}.mlp.experts.{expert_id}"
            gate = _load_tensor(model_path, weight_map, f"{expert_prefix}.gate_proj.weight").to(device=device)
            up = _load_tensor(model_path, weight_map, f"{expert_prefix}.up_proj.weight").to(device=device)
            down = _load_tensor(model_path, weight_map, f"{expert_prefix}.down_proj.weight").to(device=device)
            probes = normalized_router.index_select(0, neighbor_ids[expert_id])
            weight_scores = weight_path_importance(gate, up, down)
            pseudo_scores = pseudo_protection_importance(probes, gate, up, down, min(top_q, probes.shape[0]))
            priority, protected = combine_wick_priority(
                weight_scores,
                pseudo_scores,
                protection_ratio=protection_ratio,
                retained_channels=retained_channels,
            )
            layer_priorities.append(priority.cpu())
            layer_protected.append(protected.cpu())
            del gate, up, down
        priorities[layer_id] = torch.stack(layer_priorities)
        protected_masks[layer_id] = torch.stack(layer_protected)
        print(f"Scored WICK layer {layer_id + 1}/{num_layers}", flush=True)
    return priorities, protected_masks


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    config = _load_model_config(model_path)
    device = torch.device(args.device)
    priorities, protected = collect_wick_priorities(
        model_path=model_path,
        config=config,
        protection_ratio=float(args.protection_ratio),
        router_neighbors=int(args.router_neighbors),
        top_q=int(args.top_q),
        target_pruning_ratio=float(args.target_pruning_ratio),
        block_size=int(args.channel_block_size),
        device=device,
    )
    index_path = model_path / "model.safetensors.index.json"
    channel, profile = build_wick_artifacts(
        model_path=model_path,
        priorities_by_layer=priorities,
        protected_by_layer=protected,
        target_pruning_ratio=float(args.target_pruning_ratio),
        protection_ratio=float(args.protection_ratio),
        router_neighbors=int(args.router_neighbors),
        top_q=int(args.top_q),
        block_size=int(args.channel_block_size),
        checkpoint_identity={
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(index_path),
        },
    )
    args.output_channel_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel, args.output_channel_cache)
    profile["cache_provenance"] = {
        "channel": {"sha256": file_sha256(args.output_channel_cache), "role": "wick_channel_ranking"}
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output_profile)
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["width_histogram"] = {
        str(int(width)): int(count) for width, count in zip(*torch.unique(profile["profile_widths"], return_counts=True))
    }
    args.output_profile.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output_channel_cache.resolve())
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())