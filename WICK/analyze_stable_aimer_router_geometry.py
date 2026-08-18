from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


METRICS = ("activation_uniqueness", "functional_uniqueness", "activation_energy", "output_energy")
SCORES = ("stable", "shape")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AIMER against router-geometry functional probes.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--effective-zero-threshold", type=float, default=1.0e-12)
    return parser.parse_args()


def load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload["weight_map"].items()}


def load_tensor(model_path: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(f"Missing checkpoint tensor: {name}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    boundaries = torch.ones(values.numel(), dtype=torch.bool, device=values.device)
    boundaries[1:] = sorted_values[1:] != sorted_values[:-1]
    group_ids = boundaries.cumsum(dim=0) - 1
    positions = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    group_sums = torch.zeros(int(group_ids[-1]) + 1, device=values.device).scatter_add_(0, group_ids, positions)
    group_counts = torch.zeros_like(group_sums).scatter_add_(0, group_ids, torch.ones_like(positions))
    ranks = torch.empty_like(positions)
    ranks[order] = (group_sums / group_counts)[group_ids]
    return ranks


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2:
        return float("nan")
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = torch.linalg.vector_norm(left_rank) * torch.linalg.vector_norm(right_rank)
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left_rank, right_rank) / denominator)


def summarize(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor([value for value in values if value == value], dtype=torch.float32)
    quantiles = torch.quantile(tensor, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "n": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "p5": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "fraction_positive": float((tensor > 0).float().mean()),
        "fraction_above_0_2": float((tensor > 0.2).float().mean()),
        "fraction_above_0_4": float((tensor > 0.4).float().mean()),
    }


def summarize_integers(values: list[int]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float32)
    quantiles = torch.quantile(tensor, torch.tensor([0.05, 0.5, 0.95]))
    return {
        "n": len(values),
        "mean": float(tensor.mean()),
        "p5": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "min": int(tensor.min()),
        "max": int(tensor.max()),
    }


def aimer_scores(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projections = (gate, up, down.T)
    abs_mean = sum(weight.abs().sum(dim=1) for weight in projections)
    numel = sum(int(weight.shape[1]) for weight in projections)
    abs_mean = abs_mean / numel
    rms = (sum(weight.square().sum(dim=1) for weight in projections) / numel).sqrt()
    stable = (abs_mean / rms.clamp_min(1.0e-8)).clamp_min(1.0e-8).reciprocal()
    component_scores = torch.stack(
        [weight.square().mean(dim=1).sqrt() / weight.abs().mean(dim=1).clamp_min(1.0e-8) for weight in projections]
    )
    shape = component_scores.mean(dim=0)
    max_abs = torch.stack([weight.abs().max(dim=1).values for weight in projections]).max(dim=0).values
    active = max_abs >= threshold
    stable = stable.masked_fill(~active, -torch.inf)
    return stable, shape, active


def adaptive_neighbors(router_cosine: torch.Tensor, expert_id: int) -> tuple[torch.Tensor, int]:
    similarities = router_cosine[expert_id].clone()
    similarities[expert_id] = -torch.inf
    neighbors = torch.argsort(similarities, descending=True)[:-1]
    sorted_similarities = similarities[neighbors]
    gaps = sorted_similarities[:-1] - sorted_similarities[1:]
    neighbor_count = int(torch.argmax(gaps)) + 1
    probes = torch.cat(
        (torch.tensor([expert_id], device=router_cosine.device), neighbors[:neighbor_count])
    )
    return probes, neighbor_count


def channel_metrics(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    router_basis: torch.Tensor,
    router_cosine: torch.Tensor,
    expert_id: int,
) -> tuple[dict[str, torch.Tensor], int]:
    hidden = F.silu(router_basis @ gate.T) * (router_basis @ up.T)
    hidden_normalized = F.normalize(hidden.T, dim=1, eps=1.0e-12)
    activation_cosine = hidden_normalized @ hidden_normalized.T
    activation_cosine.fill_diagonal_(0)
    activation_uniqueness = 1.0 - activation_cosine.abs().max(dim=1).values

    down_normalized = F.normalize(down.T, dim=1, eps=1.0e-12)
    down_cosine = down_normalized @ down_normalized.T
    functional_cosine = activation_cosine * down_cosine
    functional_cosine.fill_diagonal_(0)
    functional_uniqueness = 1.0 - functional_cosine.abs().max(dim=1).values

    local_indices, neighbor_count = adaptive_neighbors(router_cosine, expert_id)
    local_hidden = hidden.index_select(0, local_indices)
    activation_energy = local_hidden.square().mean(dim=0)
    output_energy = activation_energy * down.square().sum(dim=0)
    return {
        "activation_uniqueness": activation_uniqueness,
        "functional_uniqueness": functional_uniqueness,
        "activation_energy": activation_energy,
        "output_energy": output_energy,
    }, neighbor_count


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    config_payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config = config_payload.get("text_config", config_payload)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    intermediate_size = int(config["moe_intermediate_size"])
    model_type = str(config["model_type"])
    threshold = float(args.effective_zero_threshold)
    device = torch.device(args.device)
    weight_map = load_weight_map(model_path)
    correlations = {score: {metric: [] for metric in METRICS} for score in SCORES}
    neighbor_counts: list[int] = []
    neighbor_counts_by_layer: dict[str, dict[str, float | int]] = {}
    active_counts: list[int] = []

    for layer_id in range(num_layers):
        if model_type == "qwen3_moe":
            router_name = f"model.layers.{layer_id}.mlp.gate.weight"
        else:
            router_name = f"model.language_model.layers.{layer_id}.mlp.gate.weight"
        router = load_tensor(model_path, weight_map, router_name).float().to(device)
        router_basis = F.normalize(router, dim=1, eps=1.0e-12)
        router_cosine = router_basis @ router_basis.T
        layer_neighbor_counts = []

        fused_prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
        fused_gate_up_name = f"{fused_prefix}.gate_up_proj"
        if fused_gate_up_name in weight_map:
            gate_up_all = load_tensor(model_path, weight_map, fused_gate_up_name)
            down_all = load_tensor(model_path, weight_map, f"{fused_prefix}.down_proj")

        for expert_id in range(num_experts):
            if fused_gate_up_name in weight_map:
                gate, up = gate_up_all[expert_id].chunk(2, dim=0)
                down = down_all[expert_id]
            else:
                prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
                gate = load_tensor(model_path, weight_map, f"{prefix}.gate_proj.weight")
                up = load_tensor(model_path, weight_map, f"{prefix}.up_proj.weight")
                down = load_tensor(model_path, weight_map, f"{prefix}.down_proj.weight")
            gate = gate.float().to(device)
            up = up.float().to(device)
            down = down.float().to(device)
            stable, shape, active = aimer_scores(gate, up, down, threshold)
            metrics, neighbor_count = channel_metrics(
                gate, up, down, router_basis, router_cosine, expert_id
            )
            active_counts.append(int(active.sum()))
            neighbor_counts.append(neighbor_count)
            layer_neighbor_counts.append(neighbor_count)
            for metric_name, metric_values in metrics.items():
                correlations["stable"][metric_name].append(spearman(stable[active], metric_values[active]))
                correlations["shape"][metric_name].append(spearman(shape[active], metric_values[active]))

        neighbor_counts_by_layer[str(layer_id)] = summarize_integers(layer_neighbor_counts)
        print(f"Analyzed layer {layer_id + 1}/{num_layers}", flush=True)

    output = {
        "model_path": str(model_path),
        "model_type": model_type,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "intermediate_size": intermediate_size,
        "effective_zero_threshold": threshold,
        "active_channels_per_expert": summarize_integers(active_counts),
        "adaptive_neighbor_count": summarize_integers(neighbor_counts),
        "adaptive_neighbor_count_by_layer": neighbor_counts_by_layer,
        "correlations": {
            score: {metric: summarize(values) for metric, values in metrics.items()}
            for score, metrics in correlations.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())