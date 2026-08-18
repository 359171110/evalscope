from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_naps_v2_artifacts import (
    file_sha256,
    iter_expert_weights,
    load_weight_map,
)
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import output_for_set, swiglu_response


@dataclass(frozen=True)
class ExpertLossRecord:
    layer_id: int
    expert_id: int
    holdout_token_count: int
    holdout_route_mass: float
    denominator: float
    candidate_residual: float
    baseline_residual: float

    @property
    def candidate_loss(self) -> float:
        return self.candidate_residual / max(self.denominator, 1.0e-12)

    @property
    def baseline_loss(self) -> float:
        return self.baseline_residual / max(self.denominator, 1.0e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two same-width channel rankings on held-out native routed-token responses."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidate-rankings", type=Path, required=True)
    parser.add_argument("--baseline-rankings", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-label", default="channel")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def layer_table(rankings: dict[str, Any], layer_id: int) -> dict[str, Any]:
    table = rankings["table"].get(layer_id, rankings["table"].get(str(layer_id)))
    if table is None:
        raise KeyError(f"Missing ranking table for layer {layer_id}")
    return table


def ranking_prefixes(rankings: dict[str, Any], layer_id: int, width: int) -> torch.Tensor:
    table = layer_table(rankings, layer_id)
    options = tuple(int(value) for value in table["width_options"].tolist())
    try:
        width_slot = options.index(int(width))
    except ValueError as error:
        raise ValueError(f"Width {width} is not present for layer {layer_id}: {options}") from error
    orders = table["ranked_indices_by_width"][:, width_slot].to(torch.long)
    if orders.ndim != 2 or width <= 0 or width > orders.shape[1]:
        raise ValueError(f"Layer {layer_id} ranking shape is incompatible with width {width}")
    expected = torch.arange(orders.shape[1]).expand_as(orders)
    if not torch.equal(torch.sort(orders, dim=1).values.cpu(), expected):
        raise ValueError(f"Layer {layer_id} rankings are not full channel permutations")
    return orders[:, :width]


def summarize_records(records: list[ExpertLossRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one held-out expert record is required")
    candidate_losses = torch.tensor([record.candidate_loss for record in records], dtype=torch.float64)
    baseline_losses = torch.tensor([record.baseline_loss for record in records], dtype=torch.float64)
    deltas = candidate_losses - baseline_losses
    candidate_global = sum(record.candidate_residual
                           for record in records) / max(sum(record.denominator for record in records), 1.0e-12)
    baseline_global = sum(record.baseline_residual
                          for record in records) / max(sum(record.denominator for record in records), 1.0e-12)
    summary: dict[str, Any] = {
        "compared_experts": len(records),
        "candidate_global_weighted_loss": candidate_global,
        "baseline_global_weighted_loss": baseline_global,
        "relative_global_loss_change": candidate_global / max(baseline_global, 1.0e-12) - 1.0,
        "candidate_mean_expert_loss": float(candidate_losses.mean()),
        "baseline_mean_expert_loss": float(baseline_losses.mean()),
        "candidate_median_expert_loss": float(candidate_losses.median()),
        "baseline_median_expert_loss": float(baseline_losses.median()),
        "candidate_expert_win_fraction": float((deltas < 0).double().mean()),
        "tie_fraction": float((deltas == 0).double().mean()),
        "mean_candidate_minus_baseline": float(deltas.mean()),
    }
    coverage_bins = (
        ("zero_to_7", 0, 8),
        ("8_to_31", 8, 32),
        ("32_to_127", 32, 128),
        ("128_plus", 128, None),
    )
    for label, lower, upper in coverage_bins:
        mask = torch.tensor([
            record.holdout_token_count >= lower and (upper is None or record.holdout_token_count < upper)
            for record in records
        ])
        if not bool(mask.any()):
            continue
        summary[f"{label}_experts"] = int(mask.sum())
        summary[f"{label}_candidate_mean_loss"] = float(candidate_losses[mask].mean())
        summary[f"{label}_baseline_mean_loss"] = float(baseline_losses[mask].mean())
        summary[f"{label}_candidate_win_fraction"] = float((deltas[mask] < 0).double().mean())
    return summary


def validate_inputs(
    model_path: Path,
    capture: dict[str, Any],
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> list[int]:
    if Path(capture["model_path"]).resolve() != model_path:
        raise ValueError("Capture and requested model paths do not match")
    for label, rankings in (("candidate", candidate), ("baseline", baseline)):
        ranking_model_path = rankings.get("model_path")
        if ranking_model_path is not None and Path(ranking_model_path).resolve() != model_path:
            raise ValueError(f"{label.capitalize()} rankings and requested model paths do not match")
    if "holdout" not in capture.get("splits", {}):
        raise ValueError("Capture does not contain a holdout split")
    layer_ids = [int(layer_id) for layer_id in capture["layers"]]
    if not layer_ids:
        raise ValueError("Capture does not contain any layers")
    return layer_ids


def compare_rankings(
    model_path: Path,
    capture: dict[str, Any],
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    width: int,
    device: torch.device,
) -> list[ExpertLossRecord]:
    layer_ids = validate_inputs(model_path, capture, candidate, baseline)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    adapter.channel_architecture.validate_width(width)
    records = []
    for layer_position, layer_id in enumerate(layer_ids, start=1):
        holdout_layers = capture["splits"]["holdout"]["layers"]
        holdout_layer = holdout_layers.get(layer_id, holdout_layers.get(str(layer_id)))
        if holdout_layer is None:
            raise KeyError(f"Capture is missing holdout layer {layer_id}")
        candidate_prefixes = ranking_prefixes(candidate, layer_id, width)
        baseline_prefixes = ranking_prefixes(baseline, layer_id, width)
        for expert_id, gate, up, down in iter_expert_weights(model_path, weight_map, adapter, layer_id, device):
            record = holdout_layer.get(expert_id, holdout_layer.get(str(expert_id)))
            if record is None:
                raise KeyError(f"Capture is missing holdout layer {layer_id} expert {expert_id}")
            inputs = record["inputs"].to(device)
            if inputs.shape[0] == 0:
                continue
            route_weights = record["route_weights"].to(device).float()
            responses = swiglu_response(
                inputs,
                gate,
                up,
                activation=adapter.channel_architecture.activation,
            )
            full_output = responses @ down.float().transpose(0, 1)
            factors = route_weights.square()
            denominator = float((full_output.square().sum(1) * factors).sum().item())
            residuals = []
            for prefixes in (candidate_prefixes, baseline_prefixes):
                retained_output = output_for_set(responses, down, prefixes[expert_id].to(device))
                residuals.append(float(((full_output - retained_output).square().sum(1) * factors).sum().item()))
            records.append(
                ExpertLossRecord(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    holdout_token_count=int(record["total_route_count"]),
                    holdout_route_mass=float(record["total_route_mass"]),
                    denominator=denominator,
                    candidate_residual=residuals[0],
                    baseline_residual=residuals[1],
                )
            )
        print(f"Compared holdout layer {layer_position}/{len(layer_ids)}", flush=True)
    return records


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    capture_path = args.capture.expanduser().resolve()
    candidate_path = args.candidate_rankings.expanduser().resolve()
    baseline_path = args.baseline_rankings.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    records = compare_rankings(
        model_path,
        capture,
        candidate,
        baseline,
        args.width,
        torch.device(args.device),
    )
    payload = {
        "schema_version": 1,
        "purpose": "same_width_channel_ranking_holdout_comparison",
        "model_path": str(model_path),
        "width": args.width,
        "candidate_label": args.candidate_label,
        "baseline_label": args.baseline_label,
        "capture_path": str(capture_path),
        "capture_sha256": file_sha256(capture_path),
        "candidate_rankings_path": str(candidate_path),
        "candidate_rankings_sha256": file_sha256(candidate_path),
        "baseline_rankings_path": str(baseline_path),
        "baseline_rankings_sha256": file_sha256(baseline_path),
        "summary": summarize_records(records),
        "records": [{
            **asdict(record),
            "candidate_loss": record.candidate_loss,
            "baseline_loss": record.baseline_loss,
        } for record in records],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(output_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
