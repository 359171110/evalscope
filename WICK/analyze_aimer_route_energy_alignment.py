from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


ENERGY_METRICS = ("activation_energy", "output_energy")
ALIGNMENT_METRICS = ("gate", "up", "down", "gate_up", "shape")
SCORES = ("stable", "shape")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze route-consistent AIMER energy and scale-free alignment.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--effective-zero-threshold", type=float, default=1.0e-12)
    parser.add_argument("--permutation-seeds", type=int, nargs="+", default=(11, 23, 37))
    parser.add_argument("--isotropic-seed", type=int, default=53)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--max-experts", type=int)
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
    if tensor.numel() == 0:
        return {"n": 0}
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


def summarize_counts(values: list[int]) -> dict[str, float | int]:
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
        "fraction_zero": float((tensor == 0).float().mean()),
        "fraction_one": float((tensor == 1).float().mean()),
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


def native_selected_experts(router: torch.Tensor, probes: torch.Tensor, top_k: int) -> torch.Tensor:
    router_logits = F.linear(probes, router)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    return torch.topk(routing_weights, top_k, dim=-1).indices


def signed_permutation_basis(router_basis: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_size = int(router_basis.shape[1])
    permutation = torch.randperm(hidden_size, generator=generator).to(router_basis.device)
    signs = torch.randint(0, 2, (hidden_size,), generator=generator, dtype=torch.float32)
    signs = signs.mul_(2).sub_(1).to(router_basis.device)
    return router_basis.index_select(1, permutation) * signs


def isotropic_basis(num_probes: int, hidden_size: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    basis = torch.randn((num_probes, hidden_size), generator=generator)
    return F.normalize(basis, dim=1, eps=1.0e-12).to(device)


def alignment_metrics(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, torch.Tensor]:
    gate_normalized = F.normalize(gate, dim=1, eps=1.0e-12)
    up_normalized = F.normalize(up, dim=1, eps=1.0e-12)
    down_normalized = F.normalize(down.T, dim=1, eps=1.0e-12)
    gate_projection = basis @ gate_normalized.T
    up_projection = basis @ up_normalized.T
    down_projection = basis @ down_normalized.T
    gate_alignment = gate_projection.square().mean(dim=0)
    up_alignment = up_projection.square().mean(dim=0)
    down_alignment = down_projection.square().mean(dim=0)
    return {
        "gate": gate_alignment,
        "up": up_alignment,
        "down": down_alignment,
        "gate_up": (gate_projection.square() * up_projection.square()).mean(dim=0),
        "shape": (gate_alignment + up_alignment + down_alignment) / 3.0,
    }


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    config_payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config = config_payload.get("text_config", config_payload)
    configured_layers = int(config["num_hidden_layers"])
    configured_experts = int(config["num_experts"])
    num_layers = min(configured_layers, args.max_layers or configured_layers)
    num_experts = min(configured_experts, args.max_experts or configured_experts)
    intermediate_size = int(config["moe_intermediate_size"])
    top_k = int(config["num_experts_per_tok"])
    model_type = str(config["model_type"])
    threshold = float(args.effective_zero_threshold)
    device = torch.device(args.device)
    weight_map = load_weight_map(model_path)
    permutation_seeds = tuple(int(seed) for seed in args.permutation_seeds)

    route_correlations = {score: {metric: [] for metric in ENERGY_METRICS} for score in SCORES}
    alignment_correlations = {
        "real": {score: {metric: [] for metric in ALIGNMENT_METRICS} for score in SCORES},
        "isotropic": {score: {metric: [] for metric in ALIGNMENT_METRICS} for score in SCORES},
        "permutations": {
            str(seed): {score: {metric: [] for metric in ALIGNMENT_METRICS} for score in SCORES}
            for seed in permutation_seeds
        },
    }
    coverage_counts: list[int] = []
    coverage_by_layer: dict[str, dict[str, float | int]] = {}
    active_counts: list[int] = []

    for layer_id in range(num_layers):
        if model_type == "qwen3_moe":
            router_name = f"model.layers.{layer_id}.mlp.gate.weight"
        else:
            router_name = f"model.language_model.layers.{layer_id}.mlp.gate.weight"
        router = load_tensor(model_path, weight_map, router_name).float().to(device)
        router_basis = F.normalize(router, dim=1, eps=1.0e-12)
        selected_experts = native_selected_experts(router, router_basis, top_k)
        routed_probe_masks = [
            (selected_experts == expert_id).any(dim=1) for expert_id in range(num_experts)
        ]
        layer_coverage = [int(mask.sum()) for mask in routed_probe_masks]
        coverage_counts.extend(layer_coverage)
        coverage_by_layer[str(layer_id)] = summarize_counts(layer_coverage)
        control_bases = {
            "real": router_basis,
            "isotropic": isotropic_basis(
                configured_experts, int(router.shape[1]), args.isotropic_seed + layer_id, device
            ),
        }
        control_bases.update(
            {
                f"permutation:{seed}": signed_permutation_basis(router_basis, seed)
                for seed in permutation_seeds
            }
        )

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
            scores = {"stable": stable, "shape": shape}
            active_counts.append(int(active.sum()))

            route_mask = routed_probe_masks[expert_id]
            if bool(route_mask.any()):
                route_basis = router_basis[route_mask]
                hidden = F.silu(route_basis @ gate.T) * (route_basis @ up.T)
                activation_energy = hidden.square().mean(dim=0)
                energy_metrics = {
                    "activation_energy": activation_energy,
                    "output_energy": activation_energy * down.square().sum(dim=0),
                }
                for score_name, score_values in scores.items():
                    for metric_name, metric_values in energy_metrics.items():
                        route_correlations[score_name][metric_name].append(
                            spearman(score_values[active], metric_values[active])
                        )

            for control_name, basis in control_bases.items():
                metrics = alignment_metrics(gate, up, down, basis)
                if control_name.startswith("permutation:"):
                    control = alignment_correlations["permutations"][control_name.split(":", 1)[1]]
                else:
                    control = alignment_correlations[control_name]
                for score_name, score_values in scores.items():
                    for metric_name, metric_values in metrics.items():
                        control[score_name][metric_name].append(
                            spearman(score_values[active], metric_values[active])
                        )
        print(f"Analyzed layer {layer_id + 1}/{num_layers}", flush=True)

    summarized_alignments = {
        "real": {
            score: {metric: summarize(values) for metric, values in metrics.items()}
            for score, metrics in alignment_correlations["real"].items()
        },
        "isotropic": {
            score: {metric: summarize(values) for metric, values in metrics.items()}
            for score, metrics in alignment_correlations["isotropic"].items()
        },
        "permutations": {
            seed: {
                score: {metric: summarize(values) for metric, values in metrics.items()}
                for score, metrics in controls.items()
            }
            for seed, controls in alignment_correlations["permutations"].items()
        },
    }
    permutation_mean_summary = {}
    for score in SCORES:
        permutation_mean_summary[score] = {}
        for metric in ALIGNMENT_METRICS:
            seed_means = torch.tensor(
                [summarized_alignments["permutations"][str(seed)][score][metric]["mean"] for seed in permutation_seeds]
            )
            permutation_mean_summary[score][metric] = {
                "mean_across_seed_means": float(seed_means.mean()),
                "std_across_seed_means": float(seed_means.std(correction=0)),
                "seed_means": {str(seed): float(value) for seed, value in zip(permutation_seeds, seed_means)},
            }

    output = {
        "model_path": str(model_path),
        "model_type": model_type,
        "configured_num_layers": configured_layers,
        "configured_num_experts": configured_experts,
        "analyzed_num_layers": num_layers,
        "analyzed_num_experts": num_experts,
        "intermediate_size": intermediate_size,
        "router_top_k": top_k,
        "native_router": {
            "logits": "bias_free_linear_on_normalized_router_row_probes",
            "selection": "fp32_softmax_then_global_topk",
            "norm_topk_prob": bool(config.get("norm_topk_prob", True)),
            "correction_bias": False,
            "group_constraints": False,
        },
        "effective_zero_threshold": threshold,
        "permutation_seeds": list(permutation_seeds),
        "isotropic_seed": int(args.isotropic_seed),
        "active_channels_per_expert": summarize_counts(active_counts),
        "route_probe_coverage": summarize_counts(coverage_counts),
        "route_probe_coverage_by_layer": coverage_by_layer,
        "route_energy_correlations": {
            score: {metric: summarize(values) for metric, values in metrics.items()}
            for score, metrics in route_correlations.items()
        },
        "alignment_correlations": summarized_alignments,
        "permutation_control_seed_mean_summary": permutation_mean_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())