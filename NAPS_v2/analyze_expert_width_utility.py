from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from NAPS_v2.build_naps_v2_artifacts import iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import NapsV2Config, dynamic_swap_fraction, effective_zero_mask, stable_concat_score


MODEL_WIDTHS = {
    "qwen3": (256, 384, 512),
    "qwen3.6": (192, 256, 320),
    "gemma4": (224, 352, 480),
}
SUMMARY_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Stable-AIMER marginal expert-width utility under a fixed global B6 budget."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_transfer_curve(path: Path, rows: list[dict[str, Any]], model_family: str) -> None:
    fractions = [100.0 * float(row["transfer_fraction"]) for row in rows]
    naive = [float(row["naive_cumulative_delta_coverage"]) for row in rows]
    exact = [float(row["exact_disjoint_cumulative_delta_coverage"]) for row in rows]
    width, height = 900, 540
    left, right, top, bottom = 90, 30, 60, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = min(fractions), max(fractions)
    y_min = min(0.0, min(naive), min(exact))
    y_max = max(0.0, max(naive), max(exact))
    y_padding = max((y_max - y_min) * 0.08, 1.0e-6)
    y_min -= y_padding
    y_max += y_padding

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        x = left + (x_value - x_min) / max(x_max - x_min, 1.0e-12) * plot_width
        y = top + (y_max - y_value) / max(y_max - y_min, 1.0e-12) * plot_height
        return x, y

    def polyline(values: list[float], color: str, stroke_width: float) -> str:
        points = " ".join(
            f"{x:.2f},{y:.2f}" for x, y in (point(fraction, value) for fraction, value in zip(fractions, values))
        )
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke_width}"/>'

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafaf9"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{model_family} intra-layer width transfer utility</text>',
    ]
    for index in range(6):
        fraction = x_min + index * (x_max - x_min) / 5
        x, _ = point(fraction, 0.0)
        elements.extend((
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#d6d3d1" stroke-width="1"/>',
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="13">{fraction:.0f}</text>',
        ))
    for index in range(6):
        value = y_min + index * (y_max - y_min) / 5
        _, y = point(0.0, value)
        elements.extend((
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#d6d3d1" stroke-width="1"/>',
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.3g}</text>',
        ))
    quarter_x, _ = point(25.0, 0.0)
    zero_y = point(0.0, 0.0)[1]
    elements.extend((
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_width}" y2="{zero_y:.2f}" stroke="#57534e" stroke-width="1"/>',
        f'<line x1="{quarter_x:.2f}" y1="{top}" x2="{quarter_x:.2f}" y2="{top + plot_height}" stroke="#9a3412" stroke-width="1.5" stroke-dasharray="7 5"/>',
        polyline(naive, "#0369a1", 2.2),
        polyline(exact, "#15803d", 3.0),
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="15">Small / Large experts per layer (%)</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" font-family="sans-serif" font-size="15" transform="rotate(-90 20 {height / 2})">Cumulative AIMER coverage delta</text>',
        '<line x1="555" y1="72" x2="590" y2="72" stroke="#0369a1" stroke-width="2.2"/><text x="598" y="77" font-family="sans-serif" font-size="13">Independent pairing upper bound</text>',
        '<line x1="555" y1="94" x2="590" y2="94" stroke="#15803d" stroke-width="3"/><text x="598" y="99" font-family="sans-serif" font-size="13">Exact disjoint allocation</text>',
        '<line x1="555" y1="116" x2="590" y2="116" stroke="#9a3412" stroke-width="1.5" stroke-dasharray="7 5"/><text x="598" y="121" font-family="sans-serif" font-size="13">25/50/25</text>',
        '</svg>',
    ))
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def quantile_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.double().cpu()
    points = torch.quantile(values, torch.tensor(SUMMARY_QUANTILES, dtype=torch.double))
    names = ("min", "p10", "p25", "median", "p75", "p90", "max")
    return {name: float(value.item()) for name, value in zip(names, points)}


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.double().cpu()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(left: torch.Tensor, right: torch.Tensor, epsilon: float) -> float:
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if denominator <= epsilon:
        return 0.0
    return float((left_centered @ right_centered / denominator).item())


def width_profile(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    widths: tuple[int, int, int],
    native_probe_count: int,
    epsilon: float,
) -> dict[str, float | int]:
    small_width, medium_width, large_width = widths
    config = NapsV2Config()
    raw_scores = stable_concat_score(gate, up, down, config)
    zero_mask = effective_zero_mask(gate, up, down, config.effective_zero_threshold)
    scores = raw_scores.masked_fill(~torch.isfinite(raw_scores), 0.0).clamp_min(0.0)
    sorted_scores = torch.sort(scores, descending=True, stable=True).values.double()
    cumulative = torch.cumsum(sorted_scores, dim=0)
    total = cumulative[-1].clamp_min(float(epsilon))
    coverage_small = cumulative[small_width - 1] / total
    coverage_medium = cumulative[medium_width - 1] / total
    coverage_large = cumulative[large_width - 1] / total
    shrink_block = sorted_scores[small_width:medium_width]
    expand_block = sorted_scores[medium_width:large_width]
    shrink_cost = coverage_medium - coverage_small
    expand_gain = coverage_large - coverage_medium
    shrink_tail_mean = shrink_block.mean()
    expand_tail_mean = expand_block.mean()
    cutoff_cliff = expand_tail_mean / shrink_tail_mean.clamp_min(float(epsilon))

    aimer_order = torch.argsort(raw_scores, descending=True, stable=True)
    baseline_keep = aimer_order[:medium_width]
    baseline_prune = aimer_order[medium_width:]
    active_keep = int((~zero_mask[baseline_keep]).sum().item())
    active_rescue = int((~zero_mask[baseline_prune]).sum().item())
    requested_swaps = round(dynamic_swap_fraction(native_probe_count, config) * gate.shape[0])
    actual_swaps = min(requested_swaps, active_keep, active_rescue)
    return {
        "coverage_small": float(coverage_small.item()),
        "coverage_medium": float(coverage_medium.item()),
        "coverage_large": float(coverage_large.item()),
        "shrink_cost": float(shrink_cost.item()),
        "expand_gain": float(expand_gain.item()),
        "shrink_tail_mean": float(shrink_tail_mean.item()),
        "expand_tail_mean": float(expand_tail_mean.item()),
        "cutoff_cliff": float(cutoff_cliff.item()),
        "native_probe_count": native_probe_count,
        "pp_rescue_count": actual_swaps,
        "pp_displaced_count": actual_swaps,
        "pp_aimer_conflict_rate": actual_swaps / medium_width,
        "effective_zero_count": int(zero_mask.sum().item()),
    }


def balanced_transfer_values(shrink_cost: torch.Tensor, expand_gain: torch.Tensor) -> torch.Tensor:
    expert_count = int(shrink_cost.numel())
    maximum_transfers = expert_count // 2
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.double)
    values = torch.full((maximum_transfers + 1, maximum_transfers + 1), negative_infinity)
    values[0, 0] = 0.0
    for expert_id in range(expert_count):
        updated = values.clone()
        updated[1:, :] = torch.maximum(updated[1:, :], values[:-1, :] - shrink_cost[expert_id])
        updated[:, 1:] = torch.maximum(updated[:, 1:], values[:, :-1] + expand_gain[expert_id])
        values = updated
    return torch.diagonal(values)


def balanced_assignment(
    shrink_cost: torch.Tensor,
    expand_gain: torch.Tensor,
    transfer_count: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    expert_count = int(shrink_cost.numel())
    negative_infinity = torch.tensor(float("-inf"), dtype=torch.double)
    values = torch.full((transfer_count + 1, transfer_count + 1), negative_infinity)
    values[0, 0] = 0.0
    decisions = torch.zeros(
        (expert_count, transfer_count + 1, transfer_count + 1), dtype=torch.int8
    )
    for expert_id in range(expert_count):
        updated = values.clone()
        donor_values = torch.full_like(values, negative_infinity)
        recipient_values = torch.full_like(values, negative_infinity)
        donor_values[1:, :] = values[:-1, :] - shrink_cost[expert_id]
        recipient_values[:, 1:] = values[:, :-1] + expand_gain[expert_id]
        donor_better = donor_values > updated
        updated = torch.where(donor_better, donor_values, updated)
        decisions[expert_id][donor_better] = 1
        recipient_better = recipient_values > updated
        updated = torch.where(recipient_better, recipient_values, updated)
        decisions[expert_id][recipient_better] = 2
        values = updated

    donors = []
    recipients = []
    donor_count = transfer_count
    recipient_count = transfer_count
    for expert_id in range(expert_count - 1, -1, -1):
        decision = int(decisions[expert_id, donor_count, recipient_count].item())
        if decision == 1:
            donors.append(expert_id)
            donor_count -= 1
        elif decision == 2:
            recipients.append(expert_id)
            recipient_count -= 1
    if donor_count or recipient_count:
        raise RuntimeError("Failed to reconstruct balanced width assignment")
    return torch.tensor(donors), torch.tensor(recipients), float(values[transfer_count, transfer_count].item())


def load_native_probe_counts(artifact_dir: Path, num_layers: int, num_experts: int) -> dict[tuple[int, int], int]:
    audit = torch.load(artifact_dir / "routing_audit.pt", map_location="cpu", weights_only=True)
    counts = {}
    for layer_id in range(num_layers):
        selected_experts = audit["layers"][layer_id]["selected_experts"].to(torch.long)
        layer_counts = torch.bincount(selected_experts.flatten(), minlength=num_experts)
        for expert_id in range(num_experts):
            counts[(layer_id, expert_id)] = int(layer_counts[expert_id].item())
    return counts


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    widths = MODEL_WIDTHS[adapter.model_family]
    small_width, medium_width, large_width = widths
    if medium_width != int(torch.load(artifact_dir / "profile.pt", map_location="cpu", weights_only=True)["naps"]["retained_channels"]):
        raise ValueError("Configured medium width does not match the fixed B6 artifact")
    native_counts = load_native_probe_counts(artifact_dir, adapter.num_layers, adapter.num_experts)
    layer_end = adapter.num_layers if args.layer_end is None else min(args.layer_end, adapter.num_layers)
    expert_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    layer_transfer_values: list[torch.Tensor] = []
    layer_naive_transfer_values: list[torch.Tensor] = []

    for layer_id in range(args.layer_start, layer_end):
        layer_experts = []
        for expert_id, gate, up, down in iter_expert_weights(model_path, weight_map, adapter, layer_id, device):
            row = {
                "model_family": adapter.model_family,
                "layer_id": layer_id,
                "expert_id": expert_id,
                **width_profile(
                    gate,
                    up,
                    down,
                    widths,
                    native_counts[(layer_id, expert_id)],
                    args.epsilon,
                ),
            }
            expert_rows.append(row)
            layer_experts.append(row)

        shrink_cost = torch.tensor([row["shrink_cost"] for row in layer_experts], dtype=torch.double)
        expand_gain = torch.tensor([row["expand_gain"] for row in layer_experts], dtype=torch.double)
        transfer_values = balanced_transfer_values(shrink_cost, expand_gain)
        layer_transfer_values.append(transfer_values)
        sorted_cost = torch.sort(shrink_cost).values
        sorted_gain = torch.sort(expand_gain, descending=True).values
        naive_transfer_values = torch.cat((
            torch.zeros(1, dtype=torch.double),
            torch.cumsum(sorted_gain[:adapter.num_experts // 2] - sorted_cost[:adapter.num_experts // 2], dim=0),
        ))
        layer_naive_transfer_values.append(naive_transfer_values)
        optimal_count = int(torch.argmax(transfer_values).item())
        quarter_count = adapter.num_experts // 4
        donors, recipients, quarter_utility = balanced_assignment(shrink_cost, expand_gain, quarter_count)
        assigned_widths = torch.full((adapter.num_experts,), medium_width, dtype=torch.long)
        assigned_widths[donors] = small_width
        assigned_widths[recipients] = large_width
        if int(assigned_widths.sum().item()) != adapter.num_experts * medium_width:
            raise RuntimeError("Layer width assignment does not preserve the B6 budget")
        donor_mask = torch.zeros(adapter.num_experts, dtype=torch.bool)
        recipient_mask = torch.zeros_like(donor_mask)
        donor_mask[donors] = True
        recipient_mask[recipients] = True
        for expert_id, row in enumerate(layer_experts):
            allocation = "small" if donor_mask[expert_id] else "large" if recipient_mask[expert_id] else "medium"
            assignment_rows.append({
                "model_family": adapter.model_family,
                "layer_id": layer_id,
                "expert_id": expert_id,
                "allocation": allocation,
                "assigned_width": int(assigned_widths[expert_id].item()),
                "shrink_cost": row["shrink_cost"],
                "expand_gain": row["expand_gain"],
            })
        layer_rows.append({
            "model_family": adapter.model_family,
            "layer_id": layer_id,
            "expert_count": adapter.num_experts,
            "optimal_transfer_count": optimal_count,
            "optimal_transfer_fraction": optimal_count / adapter.num_experts,
            "optimal_delta_coverage": float(transfer_values[optimal_count].item()),
            "quarter_transfer_count": quarter_count,
            "quarter_delta_coverage": quarter_utility,
            "quarter_mean_delta_coverage": quarter_utility / adapter.num_experts,
            **{f"shrink_cost_{key}": value for key, value in quantile_summary(shrink_cost).items()},
            **{f"expand_gain_{key}": value for key, value in quantile_summary(expand_gain).items()},
        })
        print(f"Analyzed layer {layer_id + 1}/{adapter.num_layers}", flush=True)

    metrics = (
        "shrink_cost",
        "expand_gain",
        "shrink_tail_mean",
        "expand_tail_mean",
        "cutoff_cliff",
    )
    distribution_rows = []
    for scope, selected_rows in (
        ("all", expert_rows),
        ("fully_active", [row for row in expert_rows if not int(row["effective_zero_count"])]),
    ):
        for metric in metrics:
            values = torch.tensor([row[metric] for row in selected_rows], dtype=torch.double)
            distribution_rows.append({
                "model_family": adapter.model_family,
                "scope": scope,
                "metric": metric,
                "expert_count": len(selected_rows),
                **quantile_summary(values),
            })

    pp_metrics = ("native_probe_count", "pp_rescue_count", "pp_displaced_count", "pp_aimer_conflict_rate")
    pp_correlation_rows = []
    for utility_metric in ("shrink_cost", "expand_gain"):
        utility = torch.tensor([row[utility_metric] for row in expert_rows], dtype=torch.double)
        for pp_metric in pp_metrics:
            pp_values = torch.tensor([row[pp_metric] for row in expert_rows], dtype=torch.double)
            pp_correlation_rows.append({
                "model_family": adapter.model_family,
                "utility_metric": utility_metric,
                "pp_metric": pp_metric,
                "spearman": spearman_correlation(utility, pp_values, args.epsilon),
            })

    transfer_curve = torch.stack(layer_transfer_values).sum(0)
    naive_transfer_curve = torch.stack(layer_naive_transfer_values).sum(0)
    transfer_rows = [{
        "model_family": adapter.model_family,
        "transfers_per_layer": transfer_count,
        "transfer_fraction": transfer_count / adapter.num_experts,
        "naive_pair_utility": 0.0 if not transfer_count else float(
            naive_transfer_curve[transfer_count].item() - naive_transfer_curve[transfer_count - 1].item()
        ),
        "naive_cumulative_delta_coverage": float(naive_transfer_curve[transfer_count].item()),
        "exact_disjoint_cumulative_delta_coverage": float(value.item()),
        "exact_disjoint_mean_delta_per_expert": float(value.item()) / len(expert_rows),
    } for transfer_count, value in enumerate(transfer_curve)]
    quarter_count = adapter.num_experts // 4
    quarter_delta = float(transfer_curve[quarter_count].item())
    homo_coverage = sum(float(row["coverage_medium"]) for row in expert_rows)
    summary = {
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
        "model_family": adapter.model_family,
        "source_width": adapter.intermediate_size,
        "widths": {"small": small_width, "medium": medium_width, "large": large_width},
        "allocation_scope": "intra_layer",
        "layer_start": args.layer_start,
        "layer_end": layer_end,
        "expert_count": len(expert_rows),
        "homo_coverage": homo_coverage,
        "quarter_hetero_coverage": homo_coverage + quarter_delta,
        "quarter_delta_coverage": quarter_delta,
        "quarter_relative_delta": quarter_delta / max(homo_coverage, args.epsilon),
        "distribution_rows": distribution_rows,
        "pp_correlation_rows": pp_correlation_rows,
    }
    write_csv(output_dir / "expert_summary.csv", expert_rows)
    write_csv(output_dir / "distribution_summary.csv", distribution_rows)
    write_csv(output_dir / "layer_summary.csv", layer_rows)
    write_csv(output_dir / "transfer_curve.csv", transfer_rows)
    write_csv(output_dir / "quarter_assignment.csv", assignment_rows)
    write_csv(output_dir / "pp_correlations.csv", pp_correlation_rows)
    plot_transfer_curve(output_dir / "transfer_curve.svg", transfer_rows, adapter.model_family)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())