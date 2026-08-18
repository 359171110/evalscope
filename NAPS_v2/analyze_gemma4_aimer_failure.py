from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.analyze_expert_width_utility import balanced_assignment, spearman_correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Gemma4 expert-level AIMER width assignment failure.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--utility-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_records(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(int(row["layer_id"]), int(row["expert_id"])): row for row in payload["records"]}


def load_utility_rows(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["layer_id"]), int(row["expert_id"])): {
            key: float(value)
            for key, value in row.items()
            if key not in {"model_family", "layer_id", "expert_id"} and value != ""
        }
        for row in rows
    }


def mean(values: torch.Tensor) -> float:
    return float(values.double().mean().item())


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    denominator = weights.double().sum().clamp_min(1.0)
    return float((values.double() * weights.double()).sum().div(denominator).item())


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    utility_csv = args.utility_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = load_records(artifact_dir / "diagnostics.json")
    utilities = load_utility_rows(utility_csv)
    if diagnostics.keys() != utilities.keys():
        raise ValueError("Diagnostics and utility rows do not cover the same experts")

    layer_ids = sorted({layer_id for layer_id, _ in diagnostics})
    layer_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    for layer_id in layer_ids:
        expert_ids = sorted(expert_id for candidate_layer, expert_id in diagnostics if candidate_layer == layer_id)
        records = [diagnostics[(layer_id, expert_id)] for expert_id in expert_ids]
        rows = [utilities[(layer_id, expert_id)] for expert_id in expert_ids]
        scores = torch.tensor([row["expert_aimer_score"] for row in records], dtype=torch.double)
        widths = torch.tensor([row["assigned_width"] for row in records], dtype=torch.long)
        native_counts = torch.tensor([row["native_probe_count"] for row in records], dtype=torch.double)
        shrink_cost = torch.tensor([row["shrink_cost"] for row in rows], dtype=torch.double)
        expand_gain = torch.tensor([row["expand_gain"] for row in rows], dtype=torch.double)
        medium_coverage = torch.tensor([row["coverage_medium"] for row in rows], dtype=torch.double)
        small_mask = widths == int(widths.min().item())
        large_mask = widths == int(widths.max().item())
        transfer_count = int(small_mask.sum().item())
        oracle_small, oracle_large, oracle_utility = balanced_assignment(shrink_cost, expand_gain, transfer_count)
        oracle_small_mask = torch.zeros_like(small_mask)
        oracle_large_mask = torch.zeros_like(large_mask)
        oracle_small_mask[oracle_small] = True
        oracle_large_mask[oracle_large] = True
        fixed_utility = float((expand_gain[large_mask].sum() - shrink_cost[small_mask].sum()).item())
        weighted_fixed_utility = float(
            (
                (expand_gain[large_mask] * native_counts[large_mask]).sum()
                - (shrink_cost[small_mask] * native_counts[small_mask]).sum()
            ).item()
        )
        normalized_weighted_fixed_utility = weighted_fixed_utility / float(native_counts.sum().clamp_min(1.0).item())
        layer_rows.append({
            "layer_id": layer_id,
            "aimer_vs_shrink_spearman": spearman_correlation(scores, shrink_cost, 1e-12),
            "aimer_vs_expand_spearman": spearman_correlation(scores, expand_gain, 1e-12),
            "fixed_utility": fixed_utility,
            "routing_weighted_fixed_utility": weighted_fixed_utility,
            "normalized_routing_weighted_fixed_utility": normalized_weighted_fixed_utility,
            "oracle_utility": oracle_utility,
            "oracle_gap": oracle_utility - fixed_utility,
            "small_oracle_overlap": mean((small_mask & oracle_small_mask).double()) / mean(small_mask.double()),
            "large_oracle_overlap": mean((large_mask & oracle_large_mask).double()) / mean(large_mask.double()),
            "assigned_small_shrink_cost": mean(shrink_cost[small_mask]),
            "other_shrink_cost": mean(shrink_cost[~small_mask]),
            "assigned_large_expand_gain": mean(expand_gain[large_mask]),
            "other_expand_gain": mean(expand_gain[~large_mask]),
            "medium_coverage_mean": mean(medium_coverage),
            "medium_coverage_routing_weighted": weighted_mean(medium_coverage, native_counts),
        })
        for index, expert_id in enumerate(expert_ids):
            expert_rows.append({
                "layer_id": layer_id,
                "expert_id": expert_id,
                "expert_aimer_score": float(scores[index].item()),
                "assigned_width": int(widths[index].item()),
                "native_probe_count": int(native_counts[index].item()),
                "shrink_cost": float(shrink_cost[index].item()),
                "expand_gain": float(expand_gain[index].item()),
                "coverage_medium": float(medium_coverage[index].item()),
                "oracle_allocation": (
                    "small" if oracle_small_mask[index] else "large" if oracle_large_mask[index] else "medium"
                ),
            })

    summary = {
        "layers": len(layer_rows),
        "experts": len(expert_rows),
        "mean_aimer_vs_shrink_spearman": sum(row["aimer_vs_shrink_spearman"] for row in layer_rows) / len(layer_rows),
        "mean_aimer_vs_expand_spearman": sum(row["aimer_vs_expand_spearman"] for row in layer_rows) / len(layer_rows),
        "total_fixed_utility": sum(row["fixed_utility"] for row in layer_rows),
        "total_routing_weighted_fixed_utility": sum(row["routing_weighted_fixed_utility"] for row in layer_rows),
        "mean_normalized_routing_weighted_fixed_utility": sum(
            row["normalized_routing_weighted_fixed_utility"] for row in layer_rows
        ) / len(layer_rows),
        "total_oracle_utility": sum(row["oracle_utility"] for row in layer_rows),
        "mean_small_oracle_overlap": sum(row["small_oracle_overlap"] for row in layer_rows) / len(layer_rows),
        "mean_large_oracle_overlap": sum(row["large_oracle_overlap"] for row in layer_rows) / len(layer_rows),
        "mean_medium_coverage": sum(row["medium_coverage_mean"] for row in layer_rows) / len(layer_rows),
        "mean_routing_weighted_medium_coverage": sum(
            row["medium_coverage_routing_weighted"] for row in layer_rows
        ) / len(layer_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("layer_summary.csv", layer_rows), ("expert_summary.csv", expert_rows)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())