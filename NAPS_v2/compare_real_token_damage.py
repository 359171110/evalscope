from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare real-token AIMER channel-damage diagnostics across model families."
    )
    parser.add_argument("--input", type=Path, action="append", required=True, help="Path to a diagnostic CSV")
    parser.add_argument("--label", action="append", required=True, help="Label paired with the preceding input")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quantiles(values: torch.Tensor) -> dict[str, float]:
    return {
        "p10": float(torch.quantile(values, 0.1).item()),
        "median": float(torch.quantile(values, 0.5).item()),
        "p90": float(torch.quantile(values, 0.9).item()),
        "mean": float(values.mean().item()),
    }


def summarize_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    metrics = {
        "spearman_aimer_damage": "aimer_damage_spearman",
        "top_width_overlap": "aimer_oracle_top_width_overlap",
        "baseline_reconstruction_loss": "pure_aimer_reconstruction_loss",
        "actual_reconstruction_loss": "actual_mask_reconstruction_loss",
        "activation_reconstruction_loss": "activation_reconstruction_loss",
        "energy_oracle_reconstruction_loss": "energy_oracle_reconstruction_loss",
    }
    summary: dict[str, Any] = {"expert_count": len(rows)}
    for source_key, output_key in metrics.items():
        values = torch.tensor([float(row[source_key]) for row in rows], dtype=torch.double)
        summary[output_key] = quantiles(values)

    actual = torch.tensor([float(row["actual_reconstruction_loss"]) for row in rows], dtype=torch.double)
    oracle = torch.tensor([float(row["energy_oracle_reconstruction_loss"]) for row in rows], dtype=torch.double)
    summary["ratio_of_mean_actual_to_oracle_loss"] = float(
        (actual.mean() / oracle.mean().clamp_min(1.0e-12)).item()
    )
    summary["mean_actual_minus_oracle_loss"] = float((actual - oracle).mean().item())

    return summary


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary = summarize_metrics(rows)
    by_layer: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer_id"])].append(row)
    summary["layers"] = {
        str(layer_id): summarize_metrics(layer_rows)
        for layer_id, layer_rows in sorted(by_layer.items())
    }
    return summary


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    if len(args.input) != len(args.label):
        raise ValueError("Each --input must have one matching --label")
    report = {
        label: summarize_rows(load_rows(path.expanduser().resolve()))
        for label, path in zip(args.label, args.input)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())