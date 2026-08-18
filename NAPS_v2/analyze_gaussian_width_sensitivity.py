from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from NAPS_v2.analyze_expert_width_utility import (
    MODEL_WIDTHS,
    balanced_assignment,
    balanced_transfer_values,
    quantile_summary,
    write_csv,
)
from NAPS_v2.analyze_functional_subspace import functional_kernel_parts
from NAPS_v2.build_naps_v2_artifacts import iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import (
    NapsV2Config,
    build_probe_sets,
    effective_zero_mask,
    select_v2_mask,
    stable_concat_score,
    swiglu_response,
)


TRANSFER_FRACTIONS = (0.10, 0.20, 0.25, 0.30)
SCOPES = ("all", "fully_active", "has_effective_zero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Gaussian output-energy sensitivity to AIMER+PP expert width."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def trace_kernel_energy(down: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    down_gram = down.transpose(0, 1) @ down
    return (down_gram * kernel).sum()


def subset_energy(down: torch.Tensor, kernel: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    if not channels.numel():
        return torch.zeros((), dtype=kernel.dtype, device=kernel.device)
    channels = channels.to(device=kernel.device, dtype=torch.long)
    subset_down = down.float().index_select(1, channels)
    subset_kernel = kernel.index_select(0, channels).index_select(1, channels)
    return trace_kernel_energy(subset_down, subset_kernel)


def selected_sets(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    coverage_responses: torch.Tensor,
    native_probe_count: int,
    widths: tuple[int, int, int],
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    config = NapsV2Config()
    zero_mask = effective_zero_mask(gate, up, down, config.effective_zero_threshold)
    aimer_scores = stable_concat_score(gate, up, down, config)
    aimer_order = torch.argsort(aimer_scores, descending=True, stable=True)
    retained = {}
    for width in widths:
        order, _ = select_v2_mask(
            aimer_order,
            aimer_scores,
            coverage_responses,
            zero_mask,
            width,
            native_probe_count,
            config,
        )
        retained[width] = order[:width]
    return retained, zero_mask


def complement(channel_count: int, retained: torch.Tensor) -> torch.Tensor:
    retained_mask = torch.zeros(channel_count, dtype=torch.bool, device=retained.device)
    retained_mask[retained] = True
    return torch.where(~retained_mask)[0]


def set_difference(left: torch.Tensor, right: torch.Tensor, channel_count: int) -> torch.Tensor:
    right_mask = torch.zeros(channel_count, dtype=torch.bool, device=left.device)
    right_mask[right] = True
    return left[~right_mask[left]]


def functional_tail_entropy(
    kernel_diagonal: torch.Tensor,
    down: torch.Tensor,
    epsilon: float,
) -> float:
    channel_energy = kernel_diagonal * down.float().square().sum(0)
    total = channel_energy.sum()
    if total <= float(epsilon):
        return 0.0
    probabilities = channel_energy / total
    positive = probabilities > 0
    entropy = -(probabilities[positive] * torch.log(probabilities[positive])).sum()
    return float((entropy / math.log(probabilities.numel())).item())


def expert_sensitivity(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    coverage_responses: torch.Tensor,
    native_probe_count: int,
    widths: tuple[int, int, int],
    epsilon: float,
) -> dict[str, float | int]:
    small_width, medium_width, large_width = widths
    retained, zero_mask = selected_sets(
        gate, up, down, coverage_responses, native_probe_count, widths
    )
    kernel, kernel_diagonal, _ = functional_kernel_parts(gate, up, gate, up)
    full_energy = trace_kernel_energy(down.float(), kernel)
    denominator = full_energy.clamp_min(float(epsilon))
    dropped_energy = {}
    dropped_loss = {}
    for width in widths:
        pruned = complement(gate.shape[0], retained[width])
        dropped_energy[width] = subset_energy(down, kernel, pruned)
        dropped_loss[width] = dropped_energy[width] / denominator

    shrink_block = set_difference(retained[medium_width], retained[small_width], gate.shape[0])
    expand_block = set_difference(retained[large_width], retained[medium_width], gate.shape[0])
    shrink_block_energy = subset_energy(down, kernel, shrink_block)
    expand_block_energy = subset_energy(down, kernel, expand_block)
    shrink_cost = dropped_loss[small_width] - dropped_loss[medium_width]
    expand_gain = dropped_loss[medium_width] - dropped_loss[large_width]
    return {
        "full_energy": float(full_energy.item()),
        "drop_loss_small": float(dropped_loss[small_width].item()),
        "drop_loss_medium": float(dropped_loss[medium_width].item()),
        "drop_loss_large": float(dropped_loss[large_width].item()),
        "functional_shrink_cost": float(shrink_cost.item()),
        "functional_expand_gain": float(expand_gain.item()),
        "shrink_block_channels": int(shrink_block.numel()),
        "expand_block_channels": int(expand_block.numel()),
        "shrink_block_energy_ratio": float((shrink_block_energy / denominator).item()),
        "expand_block_energy_ratio": float((expand_block_energy / denominator).item()),
        "functional_tail_entropy": functional_tail_entropy(kernel_diagonal, down, epsilon),
        "effective_zero_count": int(zero_mask.sum().item()),
    }


def scope_rows(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "fully_active":
        return [row for row in rows if not int(row["effective_zero_count"])]
    if scope == "has_effective_zero":
        return [row for row in rows if int(row["effective_zero_count"])]
    return rows


def summarize_distributions(expert_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "drop_loss_small",
        "drop_loss_medium",
        "drop_loss_large",
        "functional_shrink_cost",
        "functional_expand_gain",
        "shrink_block_energy_ratio",
        "expand_block_energy_ratio",
        "functional_tail_entropy",
    )
    output = []
    for scope in SCOPES:
        selected = scope_rows(expert_rows, scope)
        for metric in metrics:
            values = torch.tensor([row[metric] for row in selected], dtype=torch.double)
            statistics = quantile_summary(values) if values.numel() else {
                name: 0.0 for name in ("min", "p10", "p25", "median", "p75", "p90", "max")
            }
            output.append({
                "model_family": expert_rows[0]["model_family"],
                "scope": scope,
                "metric": metric,
                "expert_count": len(selected),
                "negative_fraction": float((values < 0).double().mean().item()) if values.numel() else 0.0,
                "p90_minus_p10": float(
                    (torch.quantile(values, 0.9) - torch.quantile(values, 0.1)).item()
                ) if values.numel() else 0.0,
                **statistics,
            })
    return output


def layer_allocation(
    model_family: str,
    layer_id: int,
    layer_rows: list[dict[str, Any]],
    widths: tuple[int, int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor]:
    small_width, medium_width, large_width = widths
    allocation_widths = {"small": small_width, "medium": medium_width, "large": large_width}
    shrink_cost = torch.tensor(
        [row["functional_shrink_cost"] for row in layer_rows], dtype=torch.double
    )
    expand_gain = torch.tensor(
        [row["functional_expand_gain"] for row in layer_rows], dtype=torch.double
    )
    transfer_values = balanced_transfer_values(shrink_cost, expand_gain)
    quarter_count = len(layer_rows) // 4
    donors, recipients, quarter_utility = balanced_assignment(
        shrink_cost, expand_gain, quarter_count
    )
    donor_mask = torch.zeros(len(layer_rows), dtype=torch.bool)
    recipient_mask = torch.zeros_like(donor_mask)
    donor_mask[donors] = True
    recipient_mask[recipients] = True
    assignments = []
    for position, row in enumerate(layer_rows):
        allocation = "small" if donor_mask[position] else "large" if recipient_mask[position] else "medium"
        assigned_width = allocation_widths[allocation]
        assignments.append({
            "model_family": model_family,
            "layer_id": layer_id,
            "expert_id": row["expert_id"],
            "allocation": allocation,
            "assigned_width": assigned_width,
            "functional_shrink_cost": row["functional_shrink_cost"],
            "functional_expand_gain": row["functional_expand_gain"],
        })
    if sum(1 for row in assignments if row["allocation"] == "small") != quarter_count:
        raise RuntimeError("Invalid donor count")
    if sum(1 for row in assignments if row["allocation"] == "large") != quarter_count:
        raise RuntimeError("Invalid recipient count")
    if sum(int(row["assigned_width"]) for row in assignments) != len(layer_rows) * medium_width:
        raise RuntimeError("Layer width assignment does not preserve the B6 budget")
    summary = {
        "model_family": model_family,
        "layer_id": layer_id,
        "expert_count": len(layer_rows),
        "optimal_transfer_count": int(torch.argmax(transfer_values).item()),
        "optimal_transfer_fraction": int(torch.argmax(transfer_values).item()) / len(layer_rows),
        "optimal_delta_retained_energy": float(transfer_values.max().item()),
        "quarter_transfer_count": quarter_count,
        "quarter_delta_retained_energy": quarter_utility,
        "quarter_mean_delta_per_expert": quarter_utility / len(layer_rows),
        "medium_width_budget": len(layer_rows) * medium_width,
    }
    return summary, assignments, transfer_values


def scoped_transfer_points(
    model_family: str,
    expert_rows: list[dict[str, Any]],
    layer_start: int,
    layer_end: int,
) -> list[dict[str, Any]]:
    output = []
    for scope in SCOPES:
        scoped = scope_rows(expert_rows, scope)
        for fraction in TRANSFER_FRACTIONS:
            total_utility = 0.0
            total_experts = 0
            total_transfers = 0
            contributing_layers = 0
            for layer_id in range(layer_start, layer_end):
                layer_rows = [row for row in scoped if int(row["layer_id"]) == layer_id]
                if len(layer_rows) < 2:
                    continue
                shrink_cost = torch.tensor(
                    [row["functional_shrink_cost"] for row in layer_rows], dtype=torch.double
                )
                expand_gain = torch.tensor(
                    [row["functional_expand_gain"] for row in layer_rows], dtype=torch.double
                )
                transfer_count = min(round(fraction * len(layer_rows)), len(layer_rows) // 2)
                values = balanced_transfer_values(shrink_cost, expand_gain)
                total_utility += float(values[transfer_count].item())
                total_experts += len(layer_rows)
                total_transfers += transfer_count
                contributing_layers += 1
            if not total_experts:
                continue
            output.append({
                "model_family": model_family,
                "scope": scope,
                "requested_transfer_fraction": fraction,
                "actual_transfer_fraction": total_transfers / total_experts,
                "contributing_layers": contributing_layers,
                "expert_count": total_experts,
                "total_transfers": total_transfers,
                "cumulative_delta_retained_energy": total_utility,
                "mean_delta_per_expert": total_utility / total_experts,
            })
    return output


def plot_transfer_curve(path: Path, rows: list[dict[str, Any]], model_family: str) -> None:
    selected = [row for row in rows if row["scope"] == "all"]
    width, height = 820, 500
    left, right, top, bottom = 90, 30, 60, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    fractions = [100.0 * float(row["requested_transfer_fraction"]) for row in selected]
    values = [float(row["cumulative_delta_retained_energy"]) for row in selected]
    x_min, x_max = 0.0, max(fractions)
    y_min, y_max = min(0.0, min(values)), max(0.0, max(values))
    padding = max((y_max - y_min) * 0.08, 1.0e-6)
    y_min -= padding
    y_max += padding

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        x = left + (x_value - x_min) / max(x_max - x_min, 1.0e-12) * plot_width
        y = top + (y_max - y_value) / max(y_max - y_min, 1.0e-12) * plot_height
        return x, y

    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(lambda pair: point(*pair), zip(fractions, values)))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafaf9"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{model_family} Gaussian width transfer utility</text>',
    ]
    for fraction in (0, 10, 20, 25, 30):
        x, _ = point(float(fraction), 0.0)
        elements.extend((
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#d6d3d1"/>',
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="13">{fraction}</text>',
        ))
    for index in range(6):
        value = y_min + index * (y_max - y_min) / 5
        _, y = point(0.0, value)
        elements.extend((
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#d6d3d1"/>',
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.3g}</text>',
        ))
    elements.extend((
        f'<polyline points="{points}" fill="none" stroke="#15803d" stroke-width="3"/>',
        *(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#15803d"/>'
            for x, y in map(lambda pair: point(*pair), zip(fractions, values))
        ),
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="15">Small / Large experts per layer (%)</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" font-family="sans-serif" font-size="15" transform="rotate(-90 20 {height / 2})">Delta functional retained energy</text>',
        '</svg>',
    ))
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


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
    medium_width = widths[1]
    profile = torch.load(artifact_dir / "profile.pt", map_location="cpu", weights_only=True)
    if medium_width != int(profile["naps"]["retained_channels"]):
        raise ValueError("Configured medium width does not match the fixed B6 artifact")
    audit = torch.load(artifact_dir / "routing_audit.pt", map_location="cpu", weights_only=True)
    layer_end = adapter.num_layers if args.layer_end is None else min(args.layer_end, adapter.num_layers)
    expert_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    for layer_id in range(args.layer_start, layer_end):
        layer_audit = audit["layers"][layer_id]
        probes = layer_audit["probes"].to(device)
        selected_experts = layer_audit["selected_experts"].to(device)
        selected_weights = layer_audit["selected_weights"].to(device)
        current_layer_rows = []
        for expert_id, gate, up, down in iter_expert_weights(
            model_path, weight_map, adapter, layer_id, device
        ):
            probe_sets = build_probe_sets(
                probes, selected_experts, selected_weights, expert_id
            )
            coverage_responses = swiglu_response(probe_sets["coverage_probes"], gate, up)
            row = {
                "model_family": adapter.model_family,
                "layer_id": layer_id,
                "expert_id": expert_id,
                "native_probe_count": int(probe_sets["native_rows"].numel()),
                **expert_sensitivity(
                    gate,
                    up,
                    down,
                    coverage_responses,
                    int(probe_sets["native_rows"].numel()),
                    widths,
                    args.epsilon,
                ),
            }
            expert_rows.append(row)
            current_layer_rows.append(row)
        layer_summary, assignments, _ = layer_allocation(
            adapter.model_family, layer_id, current_layer_rows, widths
        )
        layer_rows.append(layer_summary)
        assignment_rows.extend(assignments)
        print(f"Analyzed layer {layer_id + 1}/{adapter.num_layers}", flush=True)

    distribution_rows = summarize_distributions(expert_rows)
    transfer_rows = scoped_transfer_points(
        adapter.model_family, expert_rows, args.layer_start, layer_end
    )
    quarter_all = next(
        row for row in transfer_rows
        if row["scope"] == "all" and row["requested_transfer_fraction"] == 0.25
    )
    summary = {
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
        "model_family": adapter.model_family,
        "source_width": adapter.intermediate_size,
        "widths": {"small": widths[0], "medium": widths[1], "large": widths[2]},
        "selection": "stable_aimer_plus_dynamic_pp_swap_recomputed_per_width",
        "allocation_scope": "intra_layer",
        "layer_start": args.layer_start,
        "layer_end": layer_end,
        "expert_count": len(expert_rows),
        "quarter_delta_retained_energy": quarter_all["cumulative_delta_retained_energy"],
        "quarter_mean_delta_per_expert": quarter_all["mean_delta_per_expert"],
        "distribution_rows": distribution_rows,
        "transfer_rows": transfer_rows,
    }
    write_csv(output_dir / "expert_summary.csv", expert_rows)
    write_csv(output_dir / "distribution_summary.csv", distribution_rows)
    write_csv(output_dir / "layer_summary.csv", layer_rows)
    write_csv(output_dir / "transfer_points.csv", transfer_rows)
    write_csv(output_dir / "quarter_assignment.csv", assignment_rows)
    plot_transfer_curve(output_dir / "transfer_curve.svg", transfer_rows, adapter.model_family)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())