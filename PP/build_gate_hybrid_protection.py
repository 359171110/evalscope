from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from PP.build_protected_rankings import build_protected_artifacts, cache_orders
from WICK.build_wick_profile import file_sha256, rms_norm_rows, router_gram_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gate and Hybrid protection rankings with AIMER fill.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--aimer-cache", type=Path, required=True)
    parser.add_argument("--pseudo-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--method", choices=("GateGA", "Hybrid"), required=True)
    parser.add_argument("--retained-blocks", type=int, required=True)
    parser.add_argument("--router-neighbors", type=int, default=8)
    parser.add_argument("--top-q", type=int, default=4)
    parser.add_argument("--protection-ratio", type=float, default=0.10)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def gate_accessibility(
    probes: torch.Tensor,
    gate_weight: torch.Tensor,
    top_q: int,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Score gate rows by positive cosine affinity to the strongest local probes."""

    if probes.ndim != 2 or gate_weight.ndim != 2 or int(probes.shape[1]) != int(gate_weight.shape[1]):
        raise ValueError("probes and gate weight must align as [probe, hidden_size] and [channel, hidden_size].")
    if not 1 <= int(top_q) <= int(probes.shape[0]):
        raise ValueError("top_q must be in [1, number of probes].")
    if not torch.isfinite(probes).all() or not torch.isfinite(gate_weight).all():
        raise ValueError("probes and gate weight must contain only finite values.")
    normalized_probes = F.normalize(probes.float(), p=2, dim=1, eps=eps)
    normalized_gate = F.normalize(gate_weight.float(), p=2, dim=1, eps=eps)
    affinity = (normalized_probes @ normalized_gate.transpose(0, 1)).clamp_min(0.0)
    return torch.topk(affinity, k=int(top_q), dim=0, largest=True, sorted=False).values.mean(dim=0)


def select_protection_sets(
    pp_order: torch.Tensor,
    ga_scores: torch.Tensor,
    *,
    method: str,
    total_protected: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the requested protected set and overlap diagnostics for one expert."""

    if pp_order.ndim != 1 or ga_scores.shape != pp_order.shape:
        raise ValueError("PP order and GA scores must be aligned one-dimensional tensors.")
    channel_count = int(pp_order.numel())
    total = int(total_protected)
    if not 0 < total <= channel_count:
        raise ValueError("total_protected must be in [1, channel_count].")
    if not torch.equal(torch.sort(pp_order.to(torch.long)).values.cpu(), torch.arange(channel_count)):
        raise ValueError("PP order must be a permutation of all channel indices.")
    pp10_count = int(round(channel_count * 0.10))
    if total != pp10_count:
        raise ValueError("total_protected must equal round(0.10 * channel_count).")
    half_count = int(round(channel_count * 0.05))
    ga_order = torch.argsort(ga_scores.float(), descending=True, stable=True)
    if method == "GateGA":
        protected = ga_order[:total]
    elif method == "Hybrid":
        pp_count = half_count
        ga_count = total - pp_count
        pp_protected = pp_order[:pp_count].to(device=ga_scores.device)
        remaining_mask = torch.ones(channel_count, dtype=torch.bool, device=ga_scores.device)
        remaining_mask[pp_protected] = False
        ga_protected = ga_order[remaining_mask[ga_order]][:ga_count]
        protected = torch.cat((pp_protected, ga_protected))
    else:
        raise ValueError("method must be GateGA or Hybrid.")

    pp10 = pp_order[:pp10_count].to(device=ga_scores.device)
    ga10 = ga_order[:pp10_count]
    overlap = torch.isin(pp10, ga10).sum().item() / pp10_count
    diagnostics = {
        "pp10_ga10_overlap": float(overlap),
        "protected_channels": float(protected.numel()),
        "pp5_channels": float(half_count if method == "Hybrid" else 0),
        "ga5_channels": float(total - half_count if method == "Hybrid" else total),
    }
    return protected, diagnostics


def build_aimer_filled_order(aimer_order: torch.Tensor, protected: torch.Tensor) -> torch.Tensor:
    if aimer_order.ndim != 1 or protected.ndim != 1:
        raise ValueError("AIMER order and protected channels must be one-dimensional.")
    channel_count = int(aimer_order.numel())
    if not torch.equal(torch.sort(aimer_order.to(torch.long)).values.cpu(), torch.arange(channel_count)):
        raise ValueError("AIMER order must be a permutation of all channel indices.")
    if protected.numel() > channel_count or protected.unique().numel() != protected.numel():
        raise ValueError("protected channels must be unique and fit within the channel count.")
    protected_mask = torch.zeros(channel_count, dtype=torch.bool, device=aimer_order.device)
    protected_mask[protected.to(device=aimer_order.device)] = True
    return torch.cat((protected.to(device=aimer_order.device), aimer_order[~protected_mask[aimer_order]]))


def _load_model_config(model_path: Path) -> dict:
    payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if payload.get("model_type") != "qwen3_moe":
        raise ValueError("Gate and Hybrid currently support Qwen3 MoE checkpoints only.")
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


def collect_orders(
    *,
    model_path: Path,
    config: dict,
    aimer_orders: torch.Tensor,
    pp_orders: torch.Tensor,
    method: str,
    router_neighbors: int,
    top_q: int,
    total_protected: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    weight_map = _load_weight_map(model_path)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    expected_shape = (num_layers, num_experts, int(config["moe_intermediate_size"]))
    if tuple(aimer_orders.shape) != expected_shape or tuple(pp_orders.shape) != expected_shape:
        raise ValueError("AIMER and PP cache dimensions must match the model dimensions.")
    orders_by_layer = []
    diagnostics = []
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
        layer_orders = []
        for expert_id in range(num_experts):
            expert_prefix = f"{layer_prefix}.mlp.experts.{expert_id}"
            gate = _load_tensor(model_path, weight_map, f"{expert_prefix}.gate_proj.weight").to(device=device)
            probes = normalized_router.index_select(0, neighbor_ids[expert_id])
            ga_scores = gate_accessibility(probes, gate, min(top_q, probes.shape[0]))
            protected, record = select_protection_sets(
                pp_orders[layer_id, expert_id], ga_scores, method=method, total_protected=total_protected
            )
            order = build_aimer_filled_order(aimer_orders[layer_id, expert_id], protected)
            record.update({"layer_id": float(layer_id), "expert_id": float(expert_id)})
            diagnostics.append(record)
            layer_orders.append(order.cpu())
        orders_by_layer.append(torch.stack(layer_orders))
        print(f"Scored {method} layer {layer_id + 1}/{num_layers}", flush=True)
    return torch.stack(orders_by_layer), diagnostics


def summarize_diagnostics(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("diagnostic records must be non-empty.")
    summary = {}
    for key in ("pp10_ga10_overlap", "protected_channels", "pp5_channels", "ga5_channels"):
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


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    aimer_path = args.aimer_cache.expanduser().resolve()
    pseudo_path = args.pseudo_cache.expanduser().resolve()
    config = _load_model_config(model_path)
    aimer_orders = cache_orders(torch.load(aimer_path, map_location="cpu", weights_only=True))
    pp_orders = cache_orders(torch.load(pseudo_path, map_location="cpu", weights_only=True))
    channel_count = int(aimer_orders.shape[-1])
    total_protected = int(round(channel_count * float(args.protection_ratio)))
    orders, records = collect_orders(
        model_path=model_path,
        config=config,
        aimer_orders=aimer_orders,
        pp_orders=pp_orders,
        method=args.method,
        router_neighbors=int(args.router_neighbors),
        top_q=int(args.top_q),
        total_protected=total_protected,
        device=torch.device(args.device),
    )
    channel, profile = build_protected_artifacts(
        model_path=model_path,
        orders=orders,
        method=args.method.lower(),
        backbone=f"{args.method.lower()}_protection",
        retained_blocks=int(args.retained_blocks),
        protection_ratio=float(args.protection_ratio),
        block_size=int(args.channel_block_size),
        backbone_cache_sha256=file_sha256(aimer_path),
        pseudo_cache_sha256=file_sha256(pseudo_path),
    )
    metadata = {
        "method": args.method,
        "ranking_criterion": "positive_gate_probe_cosine_top_q_mean",
        "router_neighbors": int(args.router_neighbors),
        "top_q": int(args.top_q),
        "probe_source": "rmsnorm_router_self_plus_cosine_topk",
        "protection_ratio": float(args.protection_ratio),
        "protected_channels": total_protected,
        "pp5_channels": int(round(channel_count * 0.05)) if args.method == "Hybrid" else 0,
        "ga5_channels": total_protected - int(round(channel_count * 0.05)) if args.method == "Hybrid" else total_protected,
        "integer_split_policy": "PP=round(0.05D), GA=round(0.10D)-PP" if args.method == "Hybrid" else None,
        "overlap_diagnostic": "|Top-round(0.10D)(PP) intersect Top-round(0.10D)(GA)| / round(0.10D)",
        "aimer_cache_sha256": file_sha256(aimer_path),
        "pseudo_cache_sha256": file_sha256(pseudo_path),
        "diagnostics": summarize_diagnostics(records),
    }
    metadata["protection_source"] = "gate_accessibility" if args.method == "GateGA" else "frozen_pp_then_gate_accessibility"
    channel["pseudo_protection"] = metadata
    profile["pseudo_protection"] = metadata
    channel["gate_hybrid"] = metadata
    profile["gate_hybrid"] = metadata
    args.output_channel_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel, args.output_channel_cache)
    profile["cache_provenance"] = {
        "channel": {"sha256": file_sha256(args.output_channel_cache), "role": args.method.lower()}
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
    args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_output.write_text(
        json.dumps(
            {
                "method": args.method,
                "expert_count": len(records),
                "summary": summarize_diagnostics(records),
                "per_expert": records,
            },
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