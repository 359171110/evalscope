from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from PP.build_functional_backbone import swiglu_probe_responses
from PP.build_protected_rankings import build_protected_artifacts, cache_orders
from WICK.build_wick_profile import file_sha256, rms_norm_rows, router_gram_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PP-seeded response-coverage channel ranking.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--importance-cache", type=Path, required=True)
    parser.add_argument("--pseudo-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--importance-name", required=True)
    parser.add_argument("--retained-blocks", type=int, required=True)
    parser.add_argument("--protection-ratio", type=float, default=0.10)
    parser.add_argument("--candidate-multiplier", type=float, default=2.0)
    parser.add_argument("--router-neighbors", type=int, default=8)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def response_coverage_order(
    responses: torch.Tensor,
    importance_order: torch.Tensor,
    pseudo_order: torch.Tensor,
    *,
    retained_channels: int,
    protected_channels: int,
    candidate_multiplier: float = 2.0,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if responses.ndim != 2:
        raise ValueError("responses must have shape [probes, channels].")
    channel_count = int(responses.shape[1])
    if importance_order.shape != (channel_count,) or pseudo_order.shape != (channel_count,):
        raise ValueError("importance and pseudo orders must contain every channel exactly once.")
    retained = int(retained_channels)
    protected = int(protected_channels)
    if not 0 <= protected <= retained <= channel_count:
        raise ValueError("channel budgets must satisfy 0 <= protected <= retained <= channel_count.")
    candidate_count = min(channel_count, max(retained, int(round(float(candidate_multiplier) * retained))))
    normalized = F.normalize(responses.float(), p=2, dim=0, eps=eps)
    selected = pseudo_order[:protected].to(torch.long).tolist()
    selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=responses.device)
    if selected:
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=responses.device)
        selected_mask[selected_tensor] = True
    candidate_mask = torch.zeros(channel_count, dtype=torch.bool, device=responses.device)
    candidate_mask[importance_order[:candidate_count].to(device=responses.device)] = True
    candidate_mask[pseudo_order[:protected].to(device=responses.device)] = True
    novelty = torch.ones(channel_count, dtype=torch.float32, device=responses.device)
    if selected:
        similarity = normalized[:, selected_tensor].transpose(0, 1) @ normalized
        novelty = 1.0 - similarity.abs().amax(dim=0)
    novelty[~candidate_mask] = -torch.inf
    novelty[selected_mask] = -torch.inf
    importance_rank = torch.empty(channel_count, dtype=torch.long, device=responses.device)
    importance_rank[importance_order.to(device=responses.device)] = torch.arange(channel_count, device=responses.device)
    while len(selected) < retained:
        maximum = novelty.max()
        tied = torch.nonzero(novelty == maximum, as_tuple=False).flatten()
        chosen = tied[importance_rank[tied].argmin()]
        chosen_id = int(chosen.item())
        selected.append(chosen_id)
        selected_mask[chosen] = True
        similarity = (normalized[:, chosen].unsqueeze(0) @ normalized).squeeze(0).abs()
        novelty = torch.minimum(novelty, 1.0 - similarity)
        novelty[chosen] = -torch.inf
    selected_tensor = torch.tensor(selected, dtype=torch.long, device=importance_order.device)
    remaining = importance_order[~selected_mask.to(device=importance_order.device)[importance_order]]
    return torch.cat((selected_tensor, remaining))


def output_direction_coverage_order(
    responses: torch.Tensor,
    down_weight: torch.Tensor,
    importance_order: torch.Tensor,
    pseudo_order: torch.Tensor,
    *,
    retained_channels: int,
    protected_channels: int,
    candidate_multiplier: float = 2.0,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if down_weight.ndim != 2 or int(down_weight.shape[1]) != int(responses.shape[1]):
        raise ValueError("down weight columns must align with response channels.")
    channel_count = int(responses.shape[1])
    retained = int(retained_channels)
    protected = int(protected_channels)
    if not 0 <= protected <= retained <= channel_count:
        raise ValueError("channel budgets must satisfy 0 <= protected <= retained <= channel_count.")
    candidate_count = min(channel_count, max(retained, int(round(float(candidate_multiplier) * retained))))
    normalized_response = F.normalize(responses.float(), p=2, dim=0, eps=eps)
    normalized_down = F.normalize(down_weight.float(), p=2, dim=0, eps=eps)
    similarity = (normalized_response.transpose(0, 1) @ normalized_response).abs()
    similarity.mul_((normalized_down.transpose(0, 1) @ normalized_down).abs())
    selected = pseudo_order[:protected].to(torch.long).tolist()
    selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=similarity.device)
    if selected:
        selected_mask[torch.tensor(selected, dtype=torch.long, device=similarity.device)] = True
    candidate_mask = torch.zeros(channel_count, dtype=torch.bool, device=similarity.device)
    candidate_mask[importance_order[:candidate_count].to(device=similarity.device)] = True
    candidate_mask[pseudo_order[:protected].to(device=similarity.device)] = True
    maximum_similarity = torch.zeros(channel_count, dtype=torch.float32, device=similarity.device)
    if selected:
        maximum_similarity = similarity[selected].amax(dim=0)
    novelty = 1.0 - maximum_similarity
    novelty[~candidate_mask] = -torch.inf
    novelty[selected_mask] = -torch.inf
    importance_rank = torch.empty(channel_count, dtype=torch.long, device=similarity.device)
    importance_rank[importance_order.to(device=similarity.device)] = torch.arange(
        channel_count, device=similarity.device
    )
    while len(selected) < retained:
        maximum = novelty.max()
        tied = torch.nonzero(novelty == maximum, as_tuple=False).flatten()
        chosen = tied[importance_rank[tied].argmin()]
        selected.append(int(chosen.item()))
        selected_mask[chosen] = True
        maximum_similarity = torch.maximum(maximum_similarity, similarity[chosen])
        novelty = 1.0 - maximum_similarity
        novelty[~candidate_mask] = -torch.inf
        novelty[selected_mask] = -torch.inf
    selected_tensor = torch.tensor(selected, dtype=torch.long, device=importance_order.device)
    remaining = importance_order[~selected_mask.to(device=importance_order.device)[importance_order]]
    return torch.cat((selected_tensor, remaining))


def _load_model_config(model_path: Path) -> dict:
    payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if payload.get("model_type") != "qwen3_moe":
        raise ValueError("Response coverage currently supports Qwen3 MoE checkpoints only.")
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


def build_coverage_orders(
    *,
    model_path: Path,
    config: dict,
    importance_orders: torch.Tensor,
    pseudo_orders: torch.Tensor,
    retained_channels: int,
    protected_channels: int,
    candidate_multiplier: float,
    router_neighbors: int,
    device: torch.device,
) -> torch.Tensor:
    weight_map = _load_weight_map(model_path)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    if importance_orders.shape[:2] != (num_layers, num_experts) or pseudo_orders.shape != importance_orders.shape:
        raise ValueError("importance and pseudo caches must match the model layer/expert dimensions.")
    orders_by_layer = []
    for layer_id in range(num_layers):
        layer_prefix = f"model.layers.{layer_id}"
        router = _load_tensor(model_path, weight_map, f"{layer_prefix}.mlp.gate.weight").to(device=device)
        norm_weight = _load_tensor(
            model_path, weight_map, f"{layer_prefix}.post_attention_layernorm.weight"
        ).to(device=device)
        neighbor_ids = router_gram_neighbors(router, router_neighbors)
        normalized_router = rms_norm_rows(router, norm_weight, float(config["rms_norm_eps"]))
        layer_orders = []
        for expert_id in range(num_experts):
            expert_prefix = f"{layer_prefix}.mlp.experts.{expert_id}"
            gate = _load_tensor(model_path, weight_map, f"{expert_prefix}.gate_proj.weight").to(device=device)
            up = _load_tensor(model_path, weight_map, f"{expert_prefix}.up_proj.weight").to(device=device)
            probes = normalized_router.index_select(0, neighbor_ids[expert_id])
            responses = swiglu_probe_responses(probes, gate, up)
            order = response_coverage_order(
                responses,
                importance_orders[layer_id, expert_id].to(device=device),
                pseudo_orders[layer_id, expert_id].to(device=device),
                retained_channels=retained_channels,
                protected_channels=protected_channels,
                candidate_multiplier=candidate_multiplier,
            )
            layer_orders.append(order.cpu())
            del gate, up
        orders_by_layer.append(torch.stack(layer_orders))
        print(f"Selected response coverage layer {layer_id + 1}/{num_layers}", flush=True)
    return torch.stack(orders_by_layer)


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    importance_path = args.importance_cache.expanduser().resolve()
    pseudo_path = args.pseudo_cache.expanduser().resolve()
    importance_payload = torch.load(importance_path, map_location="cpu", weights_only=True)
    pseudo_payload = torch.load(pseudo_path, map_location="cpu", weights_only=True)
    importance_orders = cache_orders(importance_payload)
    pseudo_orders = cache_orders(pseudo_payload)
    block_size = int(args.channel_block_size)
    retained_channels = int(args.retained_blocks) * block_size
    protected_channels = int(round(int(importance_orders.shape[-1]) * float(args.protection_ratio)))
    orders = build_coverage_orders(
        model_path=model_path,
        config=_load_model_config(model_path),
        importance_orders=importance_orders,
        pseudo_orders=pseudo_orders,
        retained_channels=retained_channels,
        protected_channels=protected_channels,
        candidate_multiplier=float(args.candidate_multiplier),
        router_neighbors=int(args.router_neighbors),
        device=torch.device(args.device),
    )
    channel, profile = build_protected_artifacts(
        model_path=model_path,
        orders=orders,
        method=args.method,
        backbone=f"{args.importance_name}-response-coverage",
        retained_blocks=int(args.retained_blocks),
        protection_ratio=float(args.protection_ratio),
        block_size=block_size,
        backbone_cache_sha256=file_sha256(importance_path),
        pseudo_cache_sha256=file_sha256(pseudo_path),
    )
    profile["response_coverage"] = {
        "importance_name": args.importance_name,
        "candidate_multiplier": float(args.candidate_multiplier),
        "similarity": "absolute_cosine_swiglu_response",
        "router_neighbors": int(args.router_neighbors),
    }
    channel["response_coverage"] = profile["response_coverage"]
    args.output_channel_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel, args.output_channel_cache)
    profile["cache_provenance"] = {
        "channel": {"sha256": file_sha256(args.output_channel_cache), "role": args.method}
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output_profile)
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["width_histogram"] = {
        str(int(width)): int(count) for width, count in zip(*torch.unique(profile["profile_widths"], return_counts=True))
    }
    args.output_profile.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_channel_cache.resolve())
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())