from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from NAPS_v2.analyze_functional_subspace import functional_kernel_parts
from NAPS_v2.analyze_weight_space_similarity import retained_table
from NAPS_v2.build_naps_v2_artifacts import iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import NapsV2Config, stable_concat_score


RIDGE_VALUES = (1.0e-4, 1.0e-3, 1.0e-2)
R2_THRESHOLDS = (0.1, 0.25, 0.5)
TARGETS = ("swap_displaced", "all_pruned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze aggregate pruned-output recoverability for a fixed NAPS-v2 mask."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def fixed_channel_sets(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
) -> dict[str, torch.Tensor]:
    retained = retained.to(device=gate.device, dtype=torch.long)
    retained_mask = torch.zeros(gate.shape[0], dtype=torch.bool, device=gate.device)
    retained_mask[retained] = True
    all_pruned = torch.where(~retained_mask)[0]
    aimer_order = torch.argsort(
        stable_concat_score(gate, up, down, NapsV2Config()), descending=True, stable=True
    )
    baseline_retained = aimer_order[:retained.numel()]
    swap_displaced = baseline_retained[~retained_mask[baseline_retained]]
    return {
        "retained": retained,
        "swap_displaced": swap_displaced,
        "all_pruned": all_pruned,
    }


def trace_kernel_energy(down: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    down_gram = down.transpose(0, 1) @ down
    return (down_gram * kernel).sum()


def output_recoverability(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    pruned: torch.Tensor,
    ridge: float,
    epsilon: float,
) -> dict[str, float | int]:
    if not pruned.numel():
        return {
            "target_channels": 0,
            "output_energy": 0.0,
            "residual": 0.0,
            "output_r2": 0.0,
            "delta_norm_sq": 0.0,
            "retained_down_norm_sq": 0.0,
            "delta_norm_ratio": 0.0,
        }
    gate_retained = gate.float().index_select(0, retained)
    up_retained = up.float().index_select(0, retained)
    gate_pruned = gate.float().index_select(0, pruned)
    up_pruned = up.float().index_select(0, pruned)
    down_retained = down.float().index_select(1, retained)
    down_pruned = down.float().index_select(1, pruned)
    kernel_rr, _, _ = functional_kernel_parts(
        gate_retained, up_retained, gate_retained, up_retained
    )
    kernel_rp, _, _ = functional_kernel_parts(
        gate_retained, up_retained, gate_pruned, up_pruned
    )
    kernel_pp, _, _ = functional_kernel_parts(
        gate_pruned, up_pruned, gate_pruned, up_pruned
    )
    diagonal = torch.diagonal(kernel_rr)
    right_hand_side = kernel_rp @ down_pruned.transpose(0, 1)
    valid = diagonal > float(epsilon)
    solution = torch.zeros_like(right_hand_side)
    if valid.any():
        scale = torch.sqrt(diagonal[valid])
        normalized_system = kernel_rr[valid][:, valid] / (scale[:, None] * scale[None, :])
        normalized_system = normalized_system + float(ridge) * torch.eye(
            scale.numel(), device=scale.device, dtype=scale.dtype
        )
        normalized_right_hand_side = right_hand_side[valid] / scale[:, None]
        normalized_solution = torch.linalg.solve(normalized_system, normalized_right_hand_side)
        solution[valid] = normalized_solution / scale[:, None]
    delta_down = solution.transpose(0, 1)

    output_energy = trace_kernel_energy(down_pruned, kernel_pp)
    reconstruction_energy = trace_kernel_energy(delta_down, kernel_rr)
    cross_energy = ((delta_down.transpose(0, 1) @ down_pruned) * kernel_rp).sum()
    residual = reconstruction_energy - 2.0 * cross_energy + output_energy
    denominator = output_energy.clamp_min(float(epsilon))
    delta_norm_sq = delta_down.square().sum()
    retained_norm_sq = down_retained.square().sum()
    return {
        "target_channels": int(pruned.numel()),
        "output_energy": float(output_energy.item()),
        "residual": float(residual.item()),
        "output_r2": float((1.0 - residual / denominator).item()),
        "delta_norm_sq": float(delta_norm_sq.item()),
        "retained_down_norm_sq": float(retained_norm_sq.item()),
        "delta_norm_ratio": float(torch.sqrt(delta_norm_sq / retained_norm_sq.clamp_min(float(epsilon))).item()),
    }


def quantile_summary(values: torch.Tensor, quantiles: tuple[float, ...]) -> dict[str, float]:
    values = values.float().cpu()
    if not values.numel():
        return {f"p{round(100 * quantile)}": 0.0 for quantile in quantiles}
    points = torch.quantile(values, torch.tensor(quantiles))
    return {f"p{round(100 * quantile)}": float(value.item()) for quantile, value in zip(quantiles, points)}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_energy = sum(float(row["output_energy"]) for row in rows)
    residual = sum(float(row["residual"]) for row in rows)
    delta_norm_sq = sum(float(row["delta_norm_sq"]) for row in rows)
    retained_norm_sq = sum(float(row["retained_down_norm_sq"]) for row in rows)
    valid_rows = [row for row in rows if float(row["output_energy"]) > 1.0e-12]
    r2 = torch.tensor([float(row["output_r2"]) for row in valid_rows])
    norm_ratio = torch.tensor([float(row["delta_norm_ratio"]) for row in valid_rows])
    summary: dict[str, Any] = {
        "expert_count": len(rows),
        "positive_energy_expert_count": len(valid_rows),
        "zero_energy_expert_fraction": 1.0 - len(valid_rows) / max(len(rows), 1),
        "target_channel_count": sum(int(row["target_channels"]) for row in rows),
        "output_energy": output_energy,
        "residual": residual,
        "output_r2": 1.0 - residual / max(output_energy, 1.0e-12),
        "delta_norm_ratio": (delta_norm_sq / max(retained_norm_sq, 1.0e-12)) ** 0.5,
        **{f"expert_r2_{key}": value for key, value in quantile_summary(r2, (0.25, 0.5, 0.75, 0.9)).items()},
        **{
            f"expert_delta_norm_ratio_{key}": value
            for key, value in quantile_summary(norm_ratio, (0.5, 0.9, 0.99)).items()
        },
    }
    summary.update({
        f"expert_r2_gt_{threshold}": float((r2 > threshold).float().mean().item()) if r2.numel() else 0.0
        for threshold in R2_THRESHOLDS
    })
    return summary


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
            channels = fixed_channel_sets(gate, up, down, retained_by_expert[(layer_id, expert_id)])
            for target in TARGETS:
                for ridge in RIDGE_VALUES:
                    expert_rows.append({
                        "model_family": adapter.model_family,
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "target": target,
                        "ridge": ridge,
                        **output_recoverability(
                            gate,
                            up,
                            down,
                            channels["retained"],
                            channels[target],
                            ridge,
                            args.epsilon,
                        ),
                    })
        print(f"Analyzed layer {layer_id + 1}/{adapter.num_layers}", flush=True)

    summary_rows = []
    for target in TARGETS:
        for ridge in RIDGE_VALUES:
            selected = [row for row in expert_rows if row["target"] == target and row["ridge"] == ridge]
            summary_rows.append({
                "model_family": adapter.model_family,
                "target": target,
                "ridge": ridge,
                **summarize_rows(selected),
            })
    summary = {
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
        "model_family": adapter.model_family,
        "source_width": adapter.intermediate_size,
        "retained_channels": retained_channels,
        "layer_start": args.layer_start,
        "layer_end": layer_end,
        "ridge_values": list(RIDGE_VALUES),
        "rows": summary_rows,
    }
    write_csv(output_dir / "expert_summary.csv", expert_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())