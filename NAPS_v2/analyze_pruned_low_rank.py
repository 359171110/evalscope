from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from NAPS_v2.analyze_weight_space_similarity import retained_table
from NAPS_v2.build_naps_v2_artifacts import iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter


ENERGY_TARGETS = (0.90, 0.95, 0.99)
FIXED_RANK_RATIOS = (0.0625, 0.125, 0.25, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze low-rank compressibility of fixed B6-pruned expert channel weights."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def pruned_channels(channel_count: int, retained: torch.Tensor, device: torch.device) -> torch.Tensor:
    retained = retained.to(device=device, dtype=torch.long)
    retained_mask = torch.zeros(channel_count, dtype=torch.bool, device=device)
    retained_mask[retained] = True
    return torch.where(~retained_mask)[0]


def singular_value_energy(matrix: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, bool]:
    matrix = matrix.double()
    total_energy = matrix.square().sum()
    if total_energy <= float(epsilon):
        return torch.ones(min(matrix.shape), dtype=matrix.dtype, device=matrix.device), True
    if matrix.shape[0] <= matrix.shape[1]:
        gram = matrix @ matrix.transpose(0, 1)
    else:
        gram = matrix.transpose(0, 1) @ matrix
    try:
        squared_singular_values = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    except torch._C._LinAlgError:
        squared_singular_values = torch.linalg.svdvals(matrix).square()
    cumulative = torch.cumsum(squared_singular_values, dim=0) / squared_singular_values.sum().clamp_min(epsilon)
    cumulative[-1] = 1.0
    return cumulative, False


def matrix_metrics(matrix: torch.Tensor, epsilon: float) -> dict[str, float | int]:
    cumulative_energy, is_zero = singular_value_energy(matrix, epsilon)
    rank_max = int(cumulative_energy.numel())
    metrics: dict[str, float | int] = {"rank_max": rank_max, "is_zero": int(is_zero)}
    for target in ENERGY_TARGETS:
        rank = 0 if is_zero else int(torch.searchsorted(cumulative_energy, target).item() + 1)
        metrics[f"r{round(100 * target)}"] = rank
        metrics[f"r{round(100 * target)}_ratio"] = rank / rank_max
    for ratio in FIXED_RANK_RATIOS:
        rank = min(rank_max, max(1, math.ceil(ratio * rank_max)))
        metrics[f"energy_at_{ratio:g}"] = float(cumulative_energy[rank - 1].item())
    return metrics


def quantile_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.float().cpu()
    points = torch.quantile(values, torch.tensor((0.5, 0.75, 0.9)))
    return {
        "median": float(points[0].item()),
        "p75": float(points[1].item()),
        "p90": float(points[2].item()),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
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
    cache = torch.load(artifact_dir / "rankings.pt", map_location="cpu", weights_only=True)
    retained_channels = int(cache["naps"]["retained_channels"])
    retained_by_expert = retained_table(cache, retained_channels)
    layer_end = adapter.num_layers if args.layer_end is None else min(args.layer_end, adapter.num_layers)
    expert_rows: list[dict[str, Any]] = []

    for layer_id in range(args.layer_start, layer_end):
        for expert_id, gate, up, down in iter_expert_weights(model_path, weight_map, adapter, layer_id, device):
            pruned = pruned_channels(
                gate.shape[0], retained_by_expert[(layer_id, expert_id)], gate.device
            )
            matrices = {
                "gate": gate.index_select(0, pruned),
                "up": up.index_select(0, pruned),
                "down": down.index_select(1, pruned),
            }
            for matrix_name, matrix in matrices.items():
                expert_rows.append({
                    "model_family": adapter.model_family,
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "matrix": matrix_name,
                    **matrix_metrics(matrix, args.epsilon),
                })
        print(f"Analyzed layer {layer_id + 1}/{adapter.num_layers}", flush=True)

    rank_rows = []
    fixed_rank_rows = []
    for matrix_name in ("gate", "up", "down"):
        all_selected = [row for row in expert_rows if row["matrix"] == matrix_name]
        zero_fraction = sum(int(row["is_zero"]) for row in all_selected) / max(len(all_selected), 1)
        for scope, selected in (
            ("all", all_selected),
            ("nonzero", [row for row in all_selected if not int(row["is_zero"])]),
        ):
            for target in ENERGY_TARGETS:
                metric = f"r{round(100 * target)}_ratio"
                values = torch.tensor([float(row[metric]) for row in selected])
                rank_rows.append({
                    "model_family": adapter.model_family,
                    "matrix": matrix_name,
                    "scope": scope,
                    "metric": metric,
                    "expert_count": len(selected),
                    "zero_matrix_fraction": zero_fraction,
                    **quantile_summary(values),
                })
            for ratio in FIXED_RANK_RATIOS:
                metric = f"energy_at_{ratio:g}"
                values = torch.tensor([float(row[metric]) for row in selected])
                fixed_rank_rows.append({
                    "model_family": adapter.model_family,
                    "matrix": matrix_name,
                    "scope": scope,
                    "rank_ratio": ratio,
                    "expert_count": len(selected),
                    "zero_matrix_fraction": zero_fraction,
                    **quantile_summary(values),
                })

    summary = {
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
        "model_family": adapter.model_family,
        "source_width": adapter.intermediate_size,
        "retained_channels": retained_channels,
        "pruned_channels": adapter.intermediate_size - retained_channels,
        "layer_start": args.layer_start,
        "layer_end": layer_end,
        "energy_targets": list(ENERGY_TARGETS),
        "fixed_rank_ratios": list(FIXED_RANK_RATIOS),
        "rank_rows": rank_rows,
        "fixed_rank_rows": fixed_rank_rows,
    }
    write_csv(output_dir / "expert_summary.csv", expert_rows)
    write_csv(output_dir / "rank_summary.csv", rank_rows)
    write_csv(output_dir / "fixed_rank_energy.csv", fixed_rank_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())