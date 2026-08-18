from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_naps_v2_artifacts import iter_expert_weights, load_weight_map
from NAPS_v2.build_naps_v2_heterogeneous import expert_aimer_score
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import NapsV2Config, build_probe_sets, stable_concat_score, swiglu_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Stable-AIMER channel scores with structural-probe output removal damage."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--expert-limit", type=int)
    parser.add_argument("--width", type=int, default=352)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.double().cpu()
    order = torch.argsort(values, stable=True)
    sorted_values = values.index_select(0, order)
    ranks = torch.empty_like(values)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: torch.Tensor, right: torch.Tensor, epsilon: float) -> float:
    left = average_ranks(left)
    right = average_ranks(right)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= epsilon:
        return 0.0
    return float((left @ right / denominator).item())


def top_overlap(left: torch.Tensor, right: torch.Tensor, count: int) -> float:
    left_ids = set(torch.topk(left, count).indices.tolist())
    right_ids = set(torch.topk(right, count).indices.tolist())
    return len(left_ids & right_ids) / count


def reconstruction_loss(
    responses: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    epsilon: float,
) -> float:
    full_output = responses.float() @ down.float().transpose(0, 1)
    retained_output = responses.float().index_select(1, retained) @ down.float().index_select(
        1, retained
    ).transpose(0, 1)
    return float(
        ((full_output - retained_output).square().sum() / full_output.square().sum().clamp_min(epsilon)).item()
    )


def summarize_expert(
    aimer_scores: torch.Tensor,
    damage: torch.Tensor,
    baseline_retained: torch.Tensor,
    actual_retained: torch.Tensor,
    epsilon: float,
) -> dict[str, float]:
    def selection_metrics(prefix: str, retained: torch.Tensor) -> dict[str, float]:
        retained_mask = torch.zeros_like(aimer_scores, dtype=torch.bool)
        retained_mask[retained] = True
        retained_damage = damage[retained_mask]
        pruned_damage = damage[~retained_mask]
        return {
            f"{prefix}_retained_damage_mean": float(retained_damage.mean().item()),
            f"{prefix}_pruned_damage_mean": float(pruned_damage.mean().item()),
            f"{prefix}_retained_to_pruned_damage_ratio": float(
                (retained_damage.mean() / pruned_damage.mean().clamp_min(epsilon)).item()
            ),
            f"{prefix}_retained_damage_fraction": float(
                (retained_damage.sum() / damage.sum().clamp_min(epsilon)).item()
            ),
        }

    width = int(baseline_retained.numel())
    return {
        "spearman_aimer_damage": spearman(aimer_scores, damage, epsilon),
        "top_width_overlap": top_overlap(aimer_scores, damage, width),
        **selection_metrics("baseline", baseline_retained),
        **selection_metrics("actual", actual_retained),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    if not 0 < args.width < adapter.intermediate_size:
        raise ValueError("width must be smaller than the source intermediate size")
    audit = torch.load(artifact_dir / "routing_audit.pt", map_location="cpu", weights_only=True)
    rankings = torch.load(artifact_dir / "rankings.pt", map_location="cpu", weights_only=True)
    config = NapsV2Config()
    activation = str(adapter.text_config.get("hidden_activation", adapter.text_config.get("hidden_act", "silu")))
    rows: list[dict[str, Any]] = []

    for layer_id in args.layers:
        layer_audit = audit["layers"][layer_id]
        layer_rankings = rankings["table"][layer_id]
        width_options = layer_rankings["width_options"].to(torch.long)
        width_positions = torch.where(width_options == args.width)[0]
        if width_positions.numel() != 1:
            raise ValueError(f"Width {args.width} is not uniquely present in layer {layer_id}")
        width_position = int(width_positions.item())
        probes = layer_audit["expert_probes"].to(device)
        selected_experts = layer_audit["selected_experts"].to(device)
        selected_weights = layer_audit["selected_weights"].to(device)
        for expert_id, gate, up, down in iter_expert_weights(
            model_path, weight_map, adapter, layer_id, device
        ):
            if args.expert_limit is not None and expert_id >= args.expert_limit:
                break
            probe_sets = build_probe_sets(probes, selected_experts, selected_weights, expert_id)
            responses = swiglu_response(probe_sets["coverage_probes"], gate, up, activation=activation)
            full_output = responses.float() @ down.float().transpose(0, 1)
            denominator = full_output.square().sum().clamp_min(args.epsilon)
            channel_output_energy = responses.float().square().sum(0) * down.float().square().sum(0)
            damage = channel_output_energy / denominator
            aimer_scores = stable_concat_score(gate, up, down, config)
            finite_scores = aimer_scores.masked_fill(~torch.isfinite(aimer_scores), aimer_scores[torch.isfinite(aimer_scores)].min())
            baseline_retained = torch.topk(finite_scores, args.width).indices
            actual_retained = layer_rankings["ranked_indices_by_width"][expert_id, width_position, :args.width].to(device)
            oracle_retained = torch.topk(damage, args.width).indices
            losses_by_width = {}
            for candidate_position, candidate_width in enumerate(width_options.tolist()):
                candidate_retained = layer_rankings["ranked_indices_by_width"][
                    expert_id, candidate_position, :candidate_width
                ].to(device)
                losses_by_width[candidate_width] = reconstruction_loss(
                    responses, down, candidate_retained, args.epsilon
                )
            rows.append({
                "layer_id": layer_id,
                "expert_id": expert_id,
                "expert_aimer_score": float(expert_aimer_score(gate, up, down).item()),
                "coverage_probe_count": int(responses.shape[0]),
                "native_probe_count": int(probe_sets["native_rows"].numel()),
                **summarize_expert(
                    finite_scores, damage, baseline_retained, actual_retained, args.epsilon
                ),
                "baseline_reconstruction_loss": reconstruction_loss(
                    responses, down, baseline_retained, args.epsilon
                ),
                "actual_reconstruction_loss": reconstruction_loss(
                    responses, down, actual_retained, args.epsilon
                ),
                "energy_oracle_reconstruction_loss": reconstruction_loss(
                    responses, down, oracle_retained, args.epsilon
                ),
                "small_reconstruction_loss": losses_by_width[int(width_options[0].item())],
                "medium_reconstruction_loss": losses_by_width[int(width_options[1].item())],
                "large_reconstruction_loss": losses_by_width[int(width_options[2].item())],
                "shrink_cost": losses_by_width[int(width_options[0].item())]
                - losses_by_width[int(width_options[1].item())],
                "expand_gain": losses_by_width[int(width_options[1].item())]
                - losses_by_width[int(width_options[2].item())],
            })
        print(f"Analyzed layer {layer_id}", flush=True)

    correlations = torch.tensor([row["spearman_aimer_damage"] for row in rows], dtype=torch.double)
    overlaps = torch.tensor([row["top_width_overlap"] for row in rows], dtype=torch.double)
    baseline_ratios = torch.tensor(
        [row["baseline_retained_to_pruned_damage_ratio"] for row in rows], dtype=torch.double
    )
    actual_ratios = torch.tensor(
        [row["actual_retained_to_pruned_damage_ratio"] for row in rows], dtype=torch.double
    )
    baseline_losses = torch.tensor([row["baseline_reconstruction_loss"] for row in rows], dtype=torch.double)
    actual_losses = torch.tensor([row["actual_reconstruction_loss"] for row in rows], dtype=torch.double)
    oracle_losses = torch.tensor(
        [row["energy_oracle_reconstruction_loss"] for row in rows], dtype=torch.double
    )
    expert_scores = torch.tensor([row["expert_aimer_score"] for row in rows], dtype=torch.double)
    shrink_costs = torch.tensor([row["shrink_cost"] for row in rows], dtype=torch.double)
    expand_gains = torch.tensor([row["expand_gain"] for row in rows], dtype=torch.double)
    summary = {
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
        "model_family": adapter.model_family,
        "layers": args.layers,
        "expert_count": len(rows),
        "width": args.width,
        "probe_source": "router-derived structural expert probes",
        "damage_definition": "single-channel output energy divided by full expert output energy",
        "spearman_mean": float(correlations.mean().item()),
        "spearman_median": float(correlations.median().item()),
        "top_width_overlap_mean": float(overlaps.mean().item()),
        "baseline_retained_to_pruned_damage_ratio_mean": float(baseline_ratios.mean().item()),
        "baseline_retained_to_pruned_damage_ratio_median": float(baseline_ratios.median().item()),
        "actual_retained_to_pruned_damage_ratio_mean": float(actual_ratios.mean().item()),
        "actual_retained_to_pruned_damage_ratio_median": float(actual_ratios.median().item()),
        "baseline_reconstruction_loss_mean": float(baseline_losses.mean().item()),
        "actual_reconstruction_loss_mean": float(actual_losses.mean().item()),
        "energy_oracle_reconstruction_loss_mean": float(oracle_losses.mean().item()),
        "actual_to_baseline_reconstruction_loss_ratio": float(
            (actual_losses.mean() / baseline_losses.mean().clamp_min(args.epsilon)).item()
        ),
        "expert_aimer_vs_shrink_cost_spearman": spearman(expert_scores, shrink_costs, args.epsilon),
        "expert_aimer_vs_expand_gain_spearman": spearman(expert_scores, expand_gains, args.epsilon),
    }
    write_csv(output_dir / "expert_proxy_damage.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())