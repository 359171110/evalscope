from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_naps_v2_artifacts import (
    file_sha256,
    iter_expert_weights,
    load_weight_map,
)
from NAPS_v2.compare_channel_holdout import ExpertLossRecord, layer_table, summarize_records
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import output_for_set, swiglu_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two variable-width CHANNEL profiles on held-out native routed-token responses."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def decode_profile_widths(
    profile: dict[str, Any],
    layer_count: int,
    num_experts: int,
    source_width: int,
) -> torch.Tensor:
    block_size = int(profile["channel_block_size"])
    widths = profile["profile_widths"].to(torch.long) * block_size
    if tuple(widths.shape) != (layer_count, num_experts):
        raise ValueError(
            f"Profile width shape is {tuple(widths.shape)}, expected {(layer_count, num_experts)}"
        )
    if block_size <= 0 or bool((widths <= 0).any()) or bool((widths > source_width).any()):
        raise ValueError("Profile widths must be positive and fit inside the source width")
    width_options = {int(value) for value in profile["width_options"].tolist()}
    if not set(widths.flatten().tolist()).issubset(width_options):
        raise ValueError("Profile contains a width that is absent from its width options")
    return widths


def ranking_prefix(
    rankings: dict[str, Any],
    layer_id: int,
    expert_id: int,
    width: int,
) -> torch.Tensor:
    table = layer_table(rankings, layer_id)
    order = table["ranked_indices"][expert_id].to(torch.long)
    if width <= 0 or width > order.numel():
        raise ValueError(f"Layer {layer_id} expert {expert_id} has invalid width {width}")
    expected = torch.arange(order.numel())
    if not torch.equal(torch.sort(order).values.cpu(), expected):
        raise ValueError(f"Layer {layer_id} expert {expert_id} ranking is not a full permutation")
    return order[:width]


def compare_profiles(
    model_path: Path,
    capture: dict[str, Any],
    rankings: dict[str, Any],
    candidate_profile: dict[str, Any],
    baseline_profile: dict[str, Any],
    device: torch.device,
) -> list[ExpertLossRecord]:
    if Path(capture["model_path"]).resolve() != model_path:
        raise ValueError("Capture and requested model paths do not match")
    if Path(rankings["model_path"]).resolve() != model_path:
        raise ValueError("Rankings and requested model paths do not match")
    if "holdout" not in capture.get("splits", {}):
        raise ValueError("Capture does not contain a holdout split")
    layer_ids = [int(layer_id) for layer_id in capture["layers"]]
    if layer_ids != list(range(len(layer_ids))):
        raise ValueError("Capture layer IDs must be contiguous and zero-based")

    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.channel_architecture
    candidate_widths = decode_profile_widths(
        candidate_profile,
        len(layer_ids),
        architecture.num_experts,
        architecture.source_intermediate_size,
    )
    baseline_widths = decode_profile_widths(
        baseline_profile,
        len(layer_ids),
        architecture.num_experts,
        architecture.source_intermediate_size,
    )
    records = []
    for layer_position, layer_id in enumerate(layer_ids, start=1):
        holdout_layers = capture["splits"]["holdout"]["layers"]
        holdout_layer = holdout_layers.get(layer_id, holdout_layers.get(str(layer_id)))
        if holdout_layer is None:
            raise KeyError(f"Capture is missing holdout layer {layer_id}")
        for expert_id, gate, up, down in iter_expert_weights(
            model_path, weight_map, adapter, layer_id, device
        ):
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
                activation=architecture.activation,
            )
            full_output = responses @ down.float().transpose(0, 1)
            factors = route_weights.square()
            denominator = float((full_output.square().sum(1) * factors).sum().item())
            residuals = []
            for widths in (candidate_widths, baseline_widths):
                width = int(widths[layer_id, expert_id].item())
                retained = ranking_prefix(rankings, layer_id, expert_id, width).to(device)
                retained_output = output_for_set(responses, down, retained)
                residuals.append(
                    float(((full_output - retained_output).square().sum(1) * factors).sum().item())
                )
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
        print(f"Compared held-out profile layer {layer_position}/{len(layer_ids)}", flush=True)
    return records


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    capture_path = args.capture.expanduser().resolve()
    rankings_path = args.rankings.expanduser().resolve()
    candidate_path = args.candidate_profile.expanduser().resolve()
    baseline_path = args.baseline_profile.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    rankings = torch.load(rankings_path, map_location="cpu", weights_only=True)
    candidate_profile = torch.load(candidate_path, map_location="cpu", weights_only=True)
    baseline_profile = torch.load(baseline_path, map_location="cpu", weights_only=True)
    records = compare_profiles(
        model_path,
        capture,
        rankings,
        candidate_profile,
        baseline_profile,
        torch.device(args.device),
    )
    payload = {
        "schema_version": 1,
        "purpose": "variable_width_channel_profile_holdout_comparison",
        "model_path": str(model_path),
        "candidate_label": args.candidate_label,
        "baseline_label": args.baseline_label,
        "capture_path": str(capture_path),
        "capture_sha256": file_sha256(capture_path),
        "rankings_path": str(rankings_path),
        "rankings_sha256": file_sha256(rankings_path),
        "candidate_profile_path": str(candidate_path),
        "candidate_profile_sha256": file_sha256(candidate_path),
        "baseline_profile_path": str(baseline_path),
        "baseline_profile_sha256": file_sha256(baseline_path),
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