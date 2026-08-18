from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Original-only and Shape-only AIMER channels.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--original-rankings", type=Path, required=True)
    parser.add_argument("--shape-rankings", type=Path, required=True)
    parser.add_argument("--retained-channels", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_weight_map(model_path: Path) -> dict[str, str]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in index["weight_map"].items()}


def load_tensor(model_path: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(f"Missing checkpoint tensor: {name}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def aimer(weight: torch.Tensor, dim: int) -> torch.Tensor:
    weight = weight.float()
    return weight.square().mean(dim=dim).sqrt() / weight.abs().mean(dim=dim).clamp_min(1.0e-12)


def channel_metrics(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> dict[str, torch.Tensor]:
    gate = gate.float()
    up = up.float()
    down = down.float()
    gate_aimer = aimer(gate, 1)
    up_aimer = aimer(up, 1)
    down_aimer = aimer(down, 0)
    concentration = torch.stack((gate_aimer, up_aimer, down_aimer))
    gate_norm = torch.linalg.vector_norm(gate, dim=1).clamp_min(1.0e-12)
    up_norm = torch.linalg.vector_norm(up, dim=1).clamp_min(1.0e-12)
    down_norm = torch.linalg.vector_norm(down, dim=0).clamp_min(1.0e-12)
    log_norms = torch.stack((gate_norm.log(), up_norm.log(), down_norm.log()))
    gate_zero = gate.abs().sum(dim=1) == 0
    up_zero = up.abs().sum(dim=1) == 0
    down_zero = down.abs().sum(dim=0) == 0
    gate_max_abs = gate.abs().max(dim=1).values
    up_max_abs = up.abs().max(dim=1).values
    down_max_abs = down.abs().max(dim=0).values
    effective_zero = torch.stack((gate_max_abs, up_max_abs, down_max_abs)).max(dim=0).values < 1.0e-12
    return {
        "a_gate": gate_aimer,
        "a_up": up_aimer,
        "a_down": down_aimer,
        "a_std": concentration.std(dim=0, correction=0),
        "a_range": concentration.max(dim=0).values - concentration.min(dim=0).values,
        "log_u_over_g": up_norm.log() - gate_norm.log(),
        "log_d_over_g": down_norm.log() - gate_norm.log(),
        "log_d_over_u": down_norm.log() - up_norm.log(),
        "log_norm_std": log_norms.std(dim=0, correction=0),
        "log_norm_range": log_norms.max(dim=0).values - log_norms.min(dim=0).values,
        "gate_zero": gate_zero.float(),
        "up_zero": up_zero.float(),
        "down_zero": down_zero.float(),
        "all_zero": (gate_zero & up_zero & down_zero).float(),
        "effective_zero": effective_zero.float(),
        "max_projection_abs": torch.stack((gate_max_abs, up_max_abs, down_max_abs)).max(dim=0).values,
    }


def summarize(values: torch.Tensor) -> dict[str, float | int]:
    values = values.float()
    quantiles = torch.quantile(values, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99]))
    return {
        "n": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(correction=0)),
        "p1": float(quantiles[0]),
        "p5": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
    }


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    config_payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config = config_payload.get("text_config", config_payload)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    intermediate_size = int(config["moe_intermediate_size"])
    retained_channels = int(args.retained_channels)
    if not 0 < retained_channels < intermediate_size:
        raise ValueError("retained-channels must be between zero and the expert intermediate size.")

    original = torch.load(args.original_rankings, map_location="cpu", weights_only=True)
    shape = torch.load(args.shape_rankings, map_location="cpu", weights_only=True)
    weight_map = load_weight_map(model_path)
    pooled = {group: {} for group in ("original_only", "shape_only", "original_only_active", "shape_only_active")}
    paired_differences: dict[str, list[torch.Tensor]] = {}
    exchange_counts = []

    for layer_id in range(num_layers):
        original_order = original["table"][layer_id]["ranked_indices"].to(torch.long)
        shape_order = shape["table"][layer_id]["ranked_indices"].to(torch.long)
        if original_order.shape != (num_experts, intermediate_size) or shape_order.shape != original_order.shape:
            raise ValueError(f"Ranking shape mismatch at layer {layer_id}.")

        fused_prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
        fused_gate_up_name = f"{fused_prefix}.gate_up_proj"
        if fused_gate_up_name in weight_map:
            gate_up = load_tensor(model_path, weight_map, fused_gate_up_name)
            fused_down = load_tensor(model_path, weight_map, f"{fused_prefix}.down_proj")

        for expert_id in range(num_experts):
            if fused_gate_up_name in weight_map:
                gate, up = gate_up[expert_id].chunk(2, dim=0)
                down = fused_down[expert_id]
            else:
                prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
                gate = load_tensor(model_path, weight_map, f"{prefix}.gate_proj.weight")
                up = load_tensor(model_path, weight_map, f"{prefix}.up_proj.weight")
                down = load_tensor(model_path, weight_map, f"{prefix}.down_proj.weight")

            original_mask = torch.zeros(intermediate_size, dtype=torch.bool)
            shape_mask = torch.zeros(intermediate_size, dtype=torch.bool)
            original_mask[original_order[expert_id, :retained_channels]] = True
            shape_mask[shape_order[expert_id, :retained_channels]] = True
            original_only = original_mask & ~shape_mask
            shape_only = shape_mask & ~original_mask
            if int(original_only.sum()) != int(shape_only.sum()):
                raise ValueError("Original-only and Shape-only set sizes must match per expert.")
            exchange_counts.append(original_only.sum().reshape(1))
            metrics = channel_metrics(gate, up, down)
            active = metrics["effective_zero"] == 0
            for metric_name, metric_values in metrics.items():
                original_values = metric_values[original_only].cpu()
                shape_values = metric_values[shape_only].cpu()
                pooled["original_only"].setdefault(metric_name, []).append(original_values)
                pooled["shape_only"].setdefault(metric_name, []).append(shape_values)
                pooled["original_only_active"].setdefault(metric_name, []).append(
                    metric_values[original_only & active].cpu()
                )
                pooled["shape_only_active"].setdefault(metric_name, []).append(
                    metric_values[shape_only & active].cpu()
                )
                if original_values.numel():
                    difference = shape_values.mean() - original_values.mean()
                    paired_differences.setdefault(metric_name, []).append(difference.reshape(1))
        print(f"Analyzed layer {layer_id + 1}/{num_layers}", flush=True)

    output = {
        "model_path": str(model_path),
        "original_rankings": str(args.original_rankings.expanduser().resolve()),
        "shape_rankings": str(args.shape_rankings.expanduser().resolve()),
        "retained_channels": retained_channels,
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "exchange_count_per_expert": summarize(torch.cat(exchange_counts).float()),
        "groups": {},
        "paired_shape_minus_original": {},
    }
    for group_name, metrics in pooled.items():
        output["groups"][group_name] = {
            metric_name: summarize(torch.cat(parts)) for metric_name, parts in metrics.items()
        }
    for metric_name, differences in paired_differences.items():
        values = torch.cat(differences)
        summary = summarize(values)
        summary["fraction_positive"] = float((values > 0).float().mean())
        output["paired_shape_minus_original"][metric_name] = summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())