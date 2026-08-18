from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from NAPS_v2.analyze_weight_space_similarity import retained_table
from NAPS_v2.build_naps_v2_artifacts import iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter


SUBSPACE_SIZES = (1, 2, 4, 8, 16)
RIDGE_VALUES = (1.0e-4, 1.0e-3, 1.0e-2)
R2_THRESHOLDS = (0.1, 0.25, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze data-free functional subspace recoverability for a fixed NAPS-v2 mask."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def functional_kernel_parts(
    gate_left: torch.Tensor,
    up_left: torch.Tensor,
    gate_right: torch.Tensor,
    up_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_left = gate_left.float()
    up_left = up_left.float()
    gate_right = gate_right.float()
    up_right = up_right.float()
    left_gate_up = (gate_left * up_left).sum(1)
    right_gate_up = (gate_right * up_right).sum(1)
    kernel = (
        left_gate_up[:, None] * right_gate_up[None, :]
        + (gate_left @ gate_right.transpose(0, 1)) * (up_left @ up_right.transpose(0, 1))
        + (gate_left @ up_right.transpose(0, 1)) * (up_left @ gate_right.transpose(0, 1))
    )
    left_self = gate_left.square().sum(1) * up_left.square().sum(1) + 2.0 * left_gate_up.square()
    right_self = gate_right.square().sum(1) * up_right.square().sum(1) + 2.0 * right_gate_up.square()
    return kernel, left_self, right_self


def quantile_summary(values: torch.Tensor, quantiles: tuple[float, ...]) -> dict[str, float]:
    values = values.float().cpu()
    points = torch.quantile(values, torch.tensor(quantiles))
    return {f"p{round(100 * quantile)}": float(value.item()) for quantile, value in zip(quantiles, points)}


def recoverability_for_expert(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    epsilon: float,
) -> dict[tuple[int, float], dict[str, torch.Tensor]]:
    channel_count = gate.shape[0]
    retained = retained.to(device=gate.device, dtype=torch.long)
    retained_mask = torch.zeros(channel_count, dtype=torch.bool, device=gate.device)
    retained_mask[retained] = True
    pruned = torch.where(~retained_mask)[0]
    gate_retained = gate.index_select(0, retained)
    up_retained = up.index_select(0, retained)
    gate_pruned = gate.index_select(0, pruned)
    up_pruned = up.index_select(0, pruned)
    kernel_rr, retained_self, _ = functional_kernel_parts(
        gate_retained, up_retained, gate_retained, up_retained
    )
    kernel_pr, pruned_self, _ = functional_kernel_parts(
        gate_pruned, up_pruned, gate_retained, up_retained
    )
    denominator = torch.sqrt(
        pruned_self[:, None].clamp_min(epsilon) * retained_self[None, :].clamp_min(epsilon)
    )
    correlation = kernel_pr / denominator
    top_correlation, top_positions = torch.topk(
        correlation, k=min(max(SUBSPACE_SIZES), retained.numel()), dim=1, largest=True, sorted=True
    )
    energy = pruned_self * down.float().index_select(1, pruned).square().sum(0)
    results: dict[tuple[int, float], dict[str, torch.Tensor]] = {}

    for requested_size in SUBSPACE_SIZES:
        available = (top_correlation[:, :requested_size] > 0).sum(1)
        for ridge_value in RIDGE_VALUES:
            r2 = torch.zeros(pruned.numel(), dtype=torch.float32, device=gate.device)
            l1 = torch.zeros_like(r2)
            l2 = torch.zeros_like(r2)
            max_abs = torch.zeros_like(r2)
            negative_ratio = torch.zeros_like(r2)
            for actual_size in range(1, requested_size + 1):
                rows = torch.where(available == actual_size)[0]
                if not rows.numel():
                    continue
                candidates = top_positions.index_select(0, rows)[:, :actual_size]
                kernel_sj = kernel_pr.index_select(0, rows).gather(1, candidates)
                kernel_ss = kernel_rr[candidates[:, :, None], candidates[:, None, :]]
                diagonal = torch.diagonal(kernel_ss, dim1=1, dim2=2)
                system = kernel_ss + float(ridge_value) * torch.diag_embed(diagonal)
                try:
                    alpha = torch.linalg.solve(system, kernel_sj.unsqueeze(2)).squeeze(2)
                except RuntimeError:
                    alpha = torch.linalg.lstsq(system, kernel_sj.unsqueeze(2)).solution.squeeze(2)
                residual = (
                    pruned_self.index_select(0, rows)
                    - 2.0 * (alpha * kernel_sj).sum(1)
                    + torch.einsum("bi,bij,bj->b", alpha, kernel_ss, alpha)
                )
                explained = (1.0 - residual / pruned_self.index_select(0, rows).clamp_min(epsilon)).clamp(0.0, 1.0)
                r2[rows] = explained
                l1[rows] = alpha.abs().sum(1)
                l2[rows] = torch.linalg.vector_norm(alpha, dim=1)
                max_abs[rows] = alpha.abs().amax(1)
                negative_ratio[rows] = (alpha < 0).float().mean(1)
            results[(requested_size, ridge_value)] = {
                "pruned": pruned,
                "available": available,
                "r2": r2,
                "energy": energy,
                "l1": l1,
                "l2": l2,
                "max_abs": max_abs,
                "negative_ratio": negative_ratio,
            }
    return results


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_combination(
    model_family: str,
    subspace_size: int,
    ridge_value: float,
    chunks: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    r2 = torch.cat([chunk["r2"] for chunk in chunks])
    energy = torch.cat([chunk["energy"] for chunk in chunks])
    available = torch.cat([chunk["available"] for chunk in chunks]).float()
    row: dict[str, Any] = {
        "model_family": model_family,
        "subspace_size": subspace_size,
        "ridge": ridge_value,
        "count": int(r2.numel()),
        **quantile_summary(r2, (0.25, 0.5, 0.75, 0.9)),
        "mean_positive_candidates_used": float(available.mean().item()),
        "energy_weighted_r2": float((energy * r2).sum().div(energy.sum().clamp_min(1.0e-12)).item()),
    }
    row.update({f"r2_gt_{threshold}": float((r2 > threshold).float().mean().item()) for threshold in R2_THRESHOLDS})
    for key in ("l1", "l2", "max_abs", "negative_ratio"):
        values = torch.cat([chunk[key] for chunk in chunks])
        row.update({f"alpha_{key}_{name}": value for name, value in quantile_summary(values, (0.5, 0.9, 0.99)).items()})
    return row


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
    combinations: dict[tuple[int, float], list[dict[str, torch.Tensor]]] = {
        (size, ridge): [] for size in SUBSPACE_SIZES for ridge in RIDGE_VALUES
    }
    layer_rows: list[dict[str, Any]] = []

    for layer_id in range(adapter.num_layers):
        layer_chunks: dict[tuple[int, float], list[dict[str, torch.Tensor]]] = {
            key: [] for key in combinations
        }
        for expert_id, gate, up, down in iter_expert_weights(
            model_path, weight_map, adapter, layer_id, device
        ):
            results = recoverability_for_expert(
                gate, up, down, retained_by_expert[(layer_id, expert_id)], args.epsilon
            )
            for key, values in results.items():
                cpu_values = {name: value.detach().cpu() for name, value in values.items()}
                combinations[key].append(cpu_values)
                layer_chunks[key].append(cpu_values)
        for (size, ridge), chunks in layer_chunks.items():
            layer_rows.append({
                "layer_id": layer_id,
                **summarize_combination(adapter.model_family, size, ridge, chunks),
            })
        print(f"Analyzed layer {layer_id + 1}/{adapter.num_layers}", flush=True)

    summary_rows = [
        summarize_combination(adapter.model_family, size, ridge, chunks)
        for (size, ridge), chunks in combinations.items()
    ]
    summary = {
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
        "model_family": adapter.model_family,
        "source_width": adapter.intermediate_size,
        "retained_channels": retained_channels,
        "pruned_channels_per_expert": adapter.intermediate_size - retained_channels,
        "subspace_sizes": list(SUBSPACE_SIZES),
        "ridge_values": list(RIDGE_VALUES),
        "rows": summary_rows,
    }
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "layer_summary.csv", layer_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())