from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from PP.build_protected_rankings import build_protected_artifacts, cache_orders
from WICK.build_wick_profile import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AIMER-screened, PP-seeded BFC channel ranking.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--aimer-cache", type=Path, required=True)
    parser.add_argument("--pseudo-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--retained-blocks", type=int, required=True)
    parser.add_argument("--selection-mode", choices=("global", "local"), default="global")
    parser.add_argument("--global-bfc-cache", type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument("--protection-ratio", type=float, default=0.10)
    parser.add_argument("--candidate-extra-ratio", type=float, default=0.5)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def candidate_channel_count(
    channel_count: int,
    retained_channels: int,
    protected_channels: int,
    candidate_extra_ratio: float,
) -> int:
    remaining_budget = int(retained_channels) - int(protected_channels)
    pruned_channels = int(channel_count) - int(retained_channels)
    candidate_count = remaining_budget + int(round(float(candidate_extra_ratio) * pruned_channels))
    return min(int(channel_count) - int(protected_channels), max(remaining_budget, candidate_count))


def bilinear_functional_similarity(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if gate_weight.ndim != 2 or gate_weight.shape != up_weight.shape:
        raise ValueError("gate and up weights must have aligned [channels, hidden_size] shapes.")
    gate = gate_weight.float()
    up = up_weight.float()
    gate_gram = gate @ gate.transpose(0, 1)
    up_gram = up @ up.transpose(0, 1)
    cross_gram = gate @ up.transpose(0, 1)
    kernel = gate_gram.mul_(up_gram).add_(cross_gram * cross_gram.transpose(0, 1))
    diagonal = kernel.diagonal().clamp_min(0.0)
    denominator = torch.sqrt(diagonal.unsqueeze(1) * diagonal.unsqueeze(0)).add_(float(eps))
    return (kernel / denominator).clamp_(min=0.0, max=1.0)


def bilinear_functional_coverage_order(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    aimer_order: torch.Tensor,
    pseudo_order: torch.Tensor,
    *,
    retained_channels: int,
    protected_channels: int,
    candidate_extra_ratio: float = 0.5,
) -> torch.Tensor:
    if gate_weight.ndim != 2 or gate_weight.shape != up_weight.shape:
        raise ValueError("gate and up weights must have aligned [channels, hidden_size] shapes.")
    channel_count = int(gate_weight.shape[0])
    if aimer_order.shape != (channel_count,) or pseudo_order.shape != (channel_count,):
        raise ValueError("AIMER and pseudo orders must contain every channel exactly once.")
    retained = int(retained_channels)
    protected = int(protected_channels)
    if not 0 <= protected <= retained <= channel_count:
        raise ValueError("channel budgets must satisfy 0 <= protected <= retained <= channel_count.")

    protected_ids = pseudo_order[:protected].to(device=gate_weight.device, dtype=torch.long)
    protected_mask = torch.zeros(channel_count, dtype=torch.bool, device=gate_weight.device)
    protected_mask[protected_ids] = True
    aimer_device_order = aimer_order.to(device=gate_weight.device, dtype=torch.long)
    non_protected_aimer = aimer_device_order[~protected_mask[aimer_device_order]]
    candidate_count = candidate_channel_count(
        channel_count,
        retained,
        protected,
        candidate_extra_ratio,
    )
    candidate_ids = non_protected_aimer[:candidate_count]
    vertex_ids = torch.cat((protected_ids, candidate_ids))
    similarity = bilinear_functional_similarity(
        gate_weight.index_select(0, vertex_ids),
        up_weight.index_select(0, vertex_ids),
    )

    selected_vertices = list(range(protected))
    selected_vertex_mask = torch.zeros(len(vertex_ids), dtype=torch.bool, device=gate_weight.device)
    selected_vertex_mask[:protected] = True
    maximum_similarity = torch.zeros(len(vertex_ids), dtype=torch.float32, device=gate_weight.device)
    if protected:
        maximum_similarity = similarity[:protected].amax(dim=0)
    novelty = 1.0 - maximum_similarity
    novelty[:protected] = -torch.inf
    aimer_rank = torch.arange(candidate_count, device=gate_weight.device)
    while len(selected_vertices) < retained:
        candidate_novelty = novelty[protected:]
        maximum = candidate_novelty.max()
        tied_candidates = torch.nonzero(candidate_novelty == maximum, as_tuple=False).flatten()
        chosen_candidate = tied_candidates[aimer_rank[tied_candidates].argmin()]
        chosen_vertex = chosen_candidate + protected
        selected_vertices.append(int(chosen_vertex.item()))
        selected_vertex_mask[chosen_vertex] = True
        maximum_similarity = torch.maximum(maximum_similarity, similarity[chosen_vertex])
        novelty = 1.0 - maximum_similarity
        novelty[selected_vertex_mask] = -torch.inf

    selected_ids = vertex_ids[torch.tensor(selected_vertices, dtype=torch.long, device=gate_weight.device)]
    selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=gate_weight.device)
    selected_mask[selected_ids] = True
    remaining = aimer_device_order[~selected_mask[aimer_device_order]]
    return torch.cat((selected_ids, remaining)).to(device=aimer_order.device)


def local_bilinear_functional_coverage_order(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    aimer_order: torch.Tensor,
    pseudo_order: torch.Tensor,
    *,
    retained_channels: int,
    protected_channels: int,
    boundary_channels: int,
) -> torch.Tensor:
    if gate_weight.ndim != 2 or gate_weight.shape != up_weight.shape:
        raise ValueError("gate and up weights must have aligned [channels, hidden_size] shapes.")
    channel_count = int(gate_weight.shape[0])
    if aimer_order.shape != (channel_count,) or pseudo_order.shape != (channel_count,):
        raise ValueError("AIMER and pseudo orders must contain every channel exactly once.")
    retained = int(retained_channels)
    protected = int(protected_channels)
    boundary = int(boundary_channels)
    remaining_budget = retained - protected
    if not 0 <= protected <= retained <= channel_count:
        raise ValueError("channel budgets must satisfy 0 <= protected <= retained <= channel_count.")
    if not 0 <= boundary <= remaining_budget:
        raise ValueError("boundary channels must fit within the non-protected retained budget.")

    protected_ids = pseudo_order[:protected].to(device=gate_weight.device, dtype=torch.long)
    protected_mask = torch.zeros(channel_count, dtype=torch.bool, device=gate_weight.device)
    protected_mask[protected_ids] = True
    aimer_device_order = aimer_order.to(device=gate_weight.device, dtype=torch.long)
    non_protected_aimer = aimer_device_order[~protected_mask[aimer_device_order]]
    if remaining_budget + boundary > len(non_protected_aimer):
        raise ValueError("boundary band extends beyond the non-protected AIMER order.")

    frozen_aimer_count = remaining_budget - boundary
    frozen_ids = torch.cat((protected_ids, non_protected_aimer[:frozen_aimer_count]))
    boundary_ids = non_protected_aimer[frozen_aimer_count : remaining_budget + boundary]
    vertex_ids = torch.cat((frozen_ids, boundary_ids))
    similarity = bilinear_functional_similarity(
        gate_weight.index_select(0, vertex_ids),
        up_weight.index_select(0, vertex_ids),
    )

    frozen_count = len(frozen_ids)
    selected_vertices = list(range(frozen_count))
    selected_vertex_mask = torch.zeros(len(vertex_ids), dtype=torch.bool, device=gate_weight.device)
    selected_vertex_mask[:frozen_count] = True
    maximum_similarity = similarity[:frozen_count].amax(dim=0)
    novelty = 1.0 - maximum_similarity
    novelty[:frozen_count] = -torch.inf
    boundary_rank = torch.arange(len(boundary_ids), device=gate_weight.device)
    while len(selected_vertices) < retained:
        boundary_novelty = novelty[frozen_count:]
        maximum = boundary_novelty.max()
        tied_candidates = torch.nonzero(boundary_novelty == maximum, as_tuple=False).flatten()
        chosen_boundary = tied_candidates[boundary_rank[tied_candidates].argmin()]
        chosen_vertex = chosen_boundary + frozen_count
        selected_vertices.append(int(chosen_vertex.item()))
        selected_vertex_mask[chosen_vertex] = True
        maximum_similarity = torch.maximum(maximum_similarity, similarity[chosen_vertex])
        novelty = 1.0 - maximum_similarity
        novelty[selected_vertex_mask] = -torch.inf

    selected_ids = vertex_ids[torch.tensor(selected_vertices, dtype=torch.long, device=gate_weight.device)]
    selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=gate_weight.device)
    selected_mask[selected_ids] = True
    remaining = aimer_device_order[~selected_mask[aimer_device_order]]
    return torch.cat((selected_ids, remaining)).to(device=aimer_order.device)


def protected_aimer_order(
    aimer_order: torch.Tensor,
    pseudo_order: torch.Tensor,
    *,
    protected_channels: int,
) -> torch.Tensor:
    channel_count = int(aimer_order.shape[0])
    protected_ids = pseudo_order[:protected_channels].to(dtype=torch.long)
    protected_mask = torch.zeros(channel_count, dtype=torch.bool, device=aimer_order.device)
    protected_mask[protected_ids.to(device=aimer_order.device)] = True
    non_protected = aimer_order[~protected_mask[aimer_order]]
    return torch.cat((protected_ids.to(device=aimer_order.device), non_protected))


def _mean_pairwise_similarity(similarity: torch.Tensor) -> float:
    count = int(similarity.shape[0])
    if count < 2:
        return 0.0
    indices = torch.triu_indices(count, count, offset=1, device=similarity.device)
    return float(similarity[indices[0], indices[1]].mean().item())


def selection_diagnostics(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    aimer_order: torch.Tensor,
    baseline_order: torch.Tensor,
    global_order: torch.Tensor,
    local_order: torch.Tensor,
    *,
    retained_channels: int,
) -> dict[str, float]:
    retained = int(retained_channels)
    baseline_ids = baseline_order[:retained].to(device=gate_weight.device, dtype=torch.long)
    global_ids = global_order[:retained].to(device=gate_weight.device, dtype=torch.long)
    local_ids = local_order[:retained].to(device=gate_weight.device, dtype=torch.long)
    channel_count = int(gate_weight.shape[0])
    aimer_rank = torch.empty(channel_count, dtype=torch.long, device=gate_weight.device)
    aimer_rank[aimer_order.to(device=gate_weight.device, dtype=torch.long)] = torch.arange(
        1, channel_count + 1, device=gate_weight.device
    )
    baseline_mask = torch.zeros(channel_count, dtype=torch.bool, device=gate_weight.device)
    baseline_mask[baseline_ids] = True

    union_ids = torch.unique(torch.cat((baseline_ids, global_ids, local_ids)), sorted=True)
    union_similarity = bilinear_functional_similarity(
        gate_weight.index_select(0, union_ids),
        up_weight.index_select(0, union_ids),
    )
    union_position = torch.empty(channel_count, dtype=torch.long, device=gate_weight.device)
    union_position[union_ids] = torch.arange(len(union_ids), device=gate_weight.device)

    diagnostics = {}
    for name, selected_ids in (("aimer", baseline_ids), ("global_bfc", global_ids), ("local_bfc", local_ids)):
        selected_similarity = union_similarity.index_select(0, union_position[selected_ids]).index_select(
            1, union_position[selected_ids]
        )
        diagnostics[f"{name}_overlap_with_aimer"] = float(baseline_mask[selected_ids].float().mean().item())
        diagnostics[f"{name}_mean_aimer_rank"] = float(aimer_rank[selected_ids].float().mean().item())
        diagnostics[f"{name}_mean_pairwise_redundancy"] = _mean_pairwise_similarity(selected_similarity)
    return diagnostics


def summarize_diagnostics(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not records:
        return {}
    summary = {}
    for key in records[0]:
        values = torch.tensor([record[key] for record in records], dtype=torch.float64)
        summary[key] = {
            "mean": float(values.mean().item()),
            "p10": float(torch.quantile(values, 0.10).item()),
            "median": float(values.median().item()),
            "p90": float(torch.quantile(values, 0.90).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }
    return summary


def _load_model_config(model_path: Path) -> dict:
    payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if payload.get("model_type") != "qwen3_moe":
        raise ValueError("Bilinear functional coverage currently supports Qwen3 MoE checkpoints only.")
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


def build_bfc_orders(
    *,
    model_path: Path,
    config: dict,
    aimer_orders: torch.Tensor,
    pseudo_orders: torch.Tensor,
    retained_channels: int,
    protected_channels: int,
    candidate_extra_ratio: float,
    selection_mode: str,
    global_bfc_orders: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    weight_map = _load_weight_map(model_path)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    if aimer_orders.shape[:2] != (num_layers, num_experts) or pseudo_orders.shape != aimer_orders.shape:
        raise ValueError("AIMER and pseudo caches must match the model layer/expert dimensions.")
    if selection_mode == "local" and (global_bfc_orders is None or global_bfc_orders.shape != aimer_orders.shape):
        raise ValueError("Local-BFC diagnostics require an aligned Global-BFC cache.")
    orders_by_layer = []
    diagnostics = []
    for layer_id in range(num_layers):
        layer_orders = []
        for expert_id in range(num_experts):
            expert_prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
            gate = _load_tensor(model_path, weight_map, f"{expert_prefix}.gate_proj.weight").to(device=device)
            up = _load_tensor(model_path, weight_map, f"{expert_prefix}.up_proj.weight").to(device=device)
            aimer_order = aimer_orders[layer_id, expert_id].to(device=device)
            pseudo_order = pseudo_orders[layer_id, expert_id].to(device=device)
            if selection_mode == "local":
                order = local_bilinear_functional_coverage_order(
                    gate,
                    up,
                    aimer_order,
                    pseudo_order,
                    retained_channels=retained_channels,
                    protected_channels=protected_channels,
                    boundary_channels=protected_channels,
                )
                baseline_order = protected_aimer_order(
                    aimer_order,
                    pseudo_order,
                    protected_channels=protected_channels,
                )
                diagnostics.append(
                    selection_diagnostics(
                        gate,
                        up,
                        aimer_order,
                        baseline_order,
                        global_bfc_orders[layer_id, expert_id].to(device=device),
                        order,
                        retained_channels=retained_channels,
                    )
                )
            else:
                order = bilinear_functional_coverage_order(
                    gate,
                    up,
                    aimer_order,
                    pseudo_order,
                    retained_channels=retained_channels,
                    protected_channels=protected_channels,
                    candidate_extra_ratio=candidate_extra_ratio,
                )
            layer_orders.append(order.cpu())
            del gate, up
        orders_by_layer.append(torch.stack(layer_orders))
        print(f"Selected BFC layer {layer_id + 1}/{num_layers}", flush=True)
    return torch.stack(orders_by_layer), diagnostics


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    aimer_path = args.aimer_cache.expanduser().resolve()
    pseudo_path = args.pseudo_cache.expanduser().resolve()
    aimer_orders = cache_orders(torch.load(aimer_path, map_location="cpu", weights_only=True))
    pseudo_orders = cache_orders(torch.load(pseudo_path, map_location="cpu", weights_only=True))
    global_bfc_orders = None
    if args.global_bfc_cache is not None:
        global_bfc_orders = cache_orders(
            torch.load(args.global_bfc_cache.expanduser().resolve(), map_location="cpu", weights_only=True)
        )
    block_size = int(args.channel_block_size)
    retained_channels = int(args.retained_blocks) * block_size
    protected_channels = int(round(int(aimer_orders.shape[-1]) * float(args.protection_ratio)))
    orders, diagnostic_records = build_bfc_orders(
        model_path=model_path,
        config=_load_model_config(model_path),
        aimer_orders=aimer_orders,
        pseudo_orders=pseudo_orders,
        retained_channels=retained_channels,
        protected_channels=protected_channels,
        candidate_extra_ratio=float(args.candidate_extra_ratio),
        selection_mode=args.selection_mode,
        global_bfc_orders=global_bfc_orders,
        device=torch.device(args.device),
    )
    channel, profile = build_protected_artifacts(
        model_path=model_path,
        orders=orders,
        method=args.method,
        backbone=f"aimer-{args.selection_mode}-bilinear-functional-coverage",
        retained_blocks=int(args.retained_blocks),
        protection_ratio=float(args.protection_ratio),
        block_size=block_size,
        backbone_cache_sha256=file_sha256(aimer_path),
        pseudo_cache_sha256=file_sha256(pseudo_path),
    )
    bfc_metadata = {
        "selection_mode": args.selection_mode,
        "kernel": "bilinear_sigma_identity",
        "similarity": "positive_normalized_bilinear_covariance",
        "uses_down_proj": False,
    }
    if args.selection_mode == "local":
        remaining_budget = retained_channels - protected_channels
        bfc_metadata.update(
            {
                "boundary_channels": protected_channels,
                "boundary_pool_channels": 2 * protected_channels,
                "frozen_aimer_channels": remaining_budget - protected_channels,
                "diagnostics": summarize_diagnostics(diagnostic_records),
            }
        )
    else:
        bfc_metadata.update(
            {
                "aimer_candidate_count": candidate_channel_count(
                    int(aimer_orders.shape[-1]),
                    retained_channels,
                    protected_channels,
                    float(args.candidate_extra_ratio),
                ),
                "candidate_extra_ratio": float(args.candidate_extra_ratio),
            }
        )
    profile["bilinear_functional_coverage"] = bfc_metadata
    channel["bilinear_functional_coverage"] = bfc_metadata
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
    if args.diagnostics_output is not None:
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_output.write_text(
            json.dumps(
                {
                    "method": args.method,
                    "retained_channels": retained_channels,
                    "protected_channels": protected_channels,
                    "expert_count": len(diagnostic_records),
                    "summary": summarize_diagnostics(diagnostic_records),
                    "per_expert": diagnostic_records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(args.output_channel_cache.resolve())
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())