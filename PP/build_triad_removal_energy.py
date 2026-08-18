from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from PP.build_protected_rankings import build_protected_artifacts, cache_orders
from WICK.build_wick_profile import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AIMER-anchored Triad Removal Energy boundary ranking.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--aimer-cache", type=Path, required=True)
    parser.add_argument("--pseudo-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--retained-blocks", type=int, required=True)
    parser.add_argument("--protection-ratio", type=float, default=0.10)
    parser.add_argument("--boundary-ratio", type=float, default=0.05)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def triad_removal_energy(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    if gate_weight.ndim != 2 or gate_weight.shape != up_weight.shape:
        raise ValueError("gate and up weights must have aligned [channels, hidden_size] shapes.")
    if down_weight.ndim != 2 or down_weight.shape != gate_weight.transpose(0, 1).shape:
        raise ValueError("down weight must have shape [hidden_size, channels].")

    gate = gate_weight.float()
    up = up_weight.float()
    down = down_weight.float()
    gate_norm_sq = gate.square().sum(dim=1)
    up_norm_sq = up.square().sum(dim=1)
    gate_up_inner = (gate * up).sum(dim=1)
    down_norm_sq = down.square().sum(dim=0)
    return down_norm_sq * (gate_norm_sq * up_norm_sq + gate_up_inner.square())


def triad_boundary_order(
    aimer_order: torch.Tensor,
    pseudo_order: torch.Tensor,
    energy: torch.Tensor,
    *,
    retained_channels: int,
    protected_channels: int,
    boundary_channels: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if aimer_order.ndim != 1 or pseudo_order.shape != aimer_order.shape or energy.shape != aimer_order.shape:
        raise ValueError("AIMER order, pseudo order, and energy must align on the channel dimension.")
    channel_count = int(aimer_order.shape[0])
    retained = int(retained_channels)
    protected = int(protected_channels)
    boundary = int(boundary_channels)
    remaining_budget = retained - protected
    if not 0 <= protected <= retained <= channel_count:
        raise ValueError("channel budgets must satisfy 0 <= protected <= retained <= channel_count.")
    if not 0 < boundary <= remaining_budget:
        raise ValueError("boundary channels must be positive and fit within the non-protected retained budget.")

    device = energy.device
    protected_ids = pseudo_order[:protected].to(device=device, dtype=torch.long)
    protected_mask = torch.zeros(channel_count, dtype=torch.bool, device=device)
    protected_mask[protected_ids] = True
    aimer_device_order = aimer_order.to(device=device, dtype=torch.long)
    non_protected_aimer = aimer_device_order[~protected_mask[aimer_device_order]]
    if remaining_budget + boundary > len(non_protected_aimer):
        raise ValueError("boundary band extends beyond the non-protected AIMER order.")

    frozen_count = remaining_budget - boundary
    frozen_ids = torch.cat((protected_ids, non_protected_aimer[:frozen_count]))
    boundary_ids = non_protected_aimer[frozen_count : remaining_budget + boundary]
    boundary_energy = energy.index_select(0, boundary_ids)
    selected_positions = torch.argsort(boundary_energy, descending=True, stable=True)[:boundary]
    selected_boundary = boundary_ids.index_select(0, selected_positions)
    selected_ids = torch.cat((frozen_ids, selected_boundary))
    selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=device)
    selected_mask[selected_ids] = True
    remaining = aimer_device_order[~selected_mask[aimer_device_order]]
    order = torch.cat((selected_ids, remaining)).to(device=aimer_order.device)

    baseline_boundary = boundary_ids[:boundary]
    baseline_boundary_mask = torch.zeros(channel_count, dtype=torch.bool, device=device)
    baseline_boundary_mask[baseline_boundary] = True
    retained_from_baseline_boundary = baseline_boundary_mask[selected_boundary].sum()
    diagnostics = {
        "overlap_with_aimer": float((retained - boundary + retained_from_baseline_boundary).item() / retained),
        "boundary_overlap": float(retained_from_baseline_boundary.item() / boundary),
        "replacements": float(boundary - retained_from_baseline_boundary.item()),
        "baseline_boundary_energy_mean": float(energy.index_select(0, baseline_boundary).mean().item()),
        "selected_boundary_energy_mean": float(energy.index_select(0, selected_boundary).mean().item()),
        "boundary_pool_energy_mean": float(boundary_energy.mean().item()),
    }
    return order, diagnostics


def summarize_diagnostics(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
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
        raise ValueError("Triad Removal Energy currently supports Qwen3 MoE checkpoints only.")
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


def build_triad_orders(
    *,
    model_path: Path,
    config: dict,
    aimer_orders: torch.Tensor,
    pseudo_orders: torch.Tensor,
    retained_channels: int,
    protected_channels: int,
    boundary_channels: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    weight_map = _load_weight_map(model_path)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    if aimer_orders.shape[:2] != (num_layers, num_experts) or pseudo_orders.shape != aimer_orders.shape:
        raise ValueError("AIMER and pseudo caches must match the model layer/expert dimensions.")

    orders_by_layer = []
    diagnostics = []
    for layer_id in range(num_layers):
        layer_orders = []
        for expert_id in range(num_experts):
            expert_prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
            gate = _load_tensor(model_path, weight_map, f"{expert_prefix}.gate_proj.weight").to(device=device)
            up = _load_tensor(model_path, weight_map, f"{expert_prefix}.up_proj.weight").to(device=device)
            down = _load_tensor(model_path, weight_map, f"{expert_prefix}.down_proj.weight").to(device=device)
            energy = triad_removal_energy(gate, up, down)
            order, record = triad_boundary_order(
                aimer_orders[layer_id, expert_id],
                pseudo_orders[layer_id, expert_id],
                energy,
                retained_channels=retained_channels,
                protected_channels=protected_channels,
                boundary_channels=boundary_channels,
            )
            record.update({"layer_id": layer_id, "expert_id": expert_id})
            diagnostics.append(record)
            layer_orders.append(order.cpu())
            del gate, up, down, energy
        orders_by_layer.append(torch.stack(layer_orders))
        print(f"Selected TRE layer {layer_id + 1}/{num_layers}", flush=True)
    return torch.stack(orders_by_layer), diagnostics


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    aimer_path = args.aimer_cache.expanduser().resolve()
    pseudo_path = args.pseudo_cache.expanduser().resolve()
    aimer_orders = cache_orders(torch.load(aimer_path, map_location="cpu", weights_only=True))
    pseudo_orders = cache_orders(torch.load(pseudo_path, map_location="cpu", weights_only=True))
    channel_count = int(aimer_orders.shape[-1])
    block_size = int(args.channel_block_size)
    retained_channels = int(args.retained_blocks) * block_size
    protected_channels = int(round(channel_count * float(args.protection_ratio)))
    boundary_channels = int(round(channel_count * float(args.boundary_ratio)))
    orders, diagnostic_records = build_triad_orders(
        model_path=model_path,
        config=_load_model_config(model_path),
        aimer_orders=aimer_orders,
        pseudo_orders=pseudo_orders,
        retained_channels=retained_channels,
        protected_channels=protected_channels,
        boundary_channels=boundary_channels,
        device=torch.device(args.device),
    )
    channel, profile = build_protected_artifacts(
        model_path=model_path,
        orders=orders,
        method=args.method,
        backbone="aimer-triad-removal-energy-boundary",
        retained_blocks=int(args.retained_blocks),
        protection_ratio=float(args.protection_ratio),
        block_size=block_size,
        backbone_cache_sha256=file_sha256(aimer_path),
        pseudo_cache_sha256=file_sha256(pseudo_path),
    )
    metadata = {
        "formula": "down_norm_sq*(gate_norm_sq*up_norm_sq+gate_up_inner_sq)",
        "boundary_ratio": float(args.boundary_ratio),
        "boundary_channels": boundary_channels,
        "boundary_pool_channels": 2 * boundary_channels,
        "frozen_non_protected_aimer_channels": retained_channels - protected_channels - boundary_channels,
        "diagnostics": summarize_diagnostics(diagnostic_records),
    }
    channel["triad_removal_energy"] = metadata
    profile["triad_removal_energy"] = metadata
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
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_output.write_text(
        json.dumps(
            {
                "method": args.method,
                "retained_channels": retained_channels,
                "protected_channels": protected_channels,
                "boundary_channels": boundary_channels,
                "expert_count": len(diagnostic_records),
                "summary": summarize_diagnostics(diagnostic_records),
                "per_expert": diagnostic_records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_channel_cache.resolve())
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())