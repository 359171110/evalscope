from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_naps_v2_artifacts import (
    file_sha256,
    iter_expert_weights,
    load_weight_map,
)
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import (
    NapsV2Config,
    build_probe_sets,
    effective_zero_mask,
    output_for_set,
    stable_concat_score,
    swiglu_response,
    weighted_output_loss,
)


@dataclass(frozen=True)
class ChannelScoreSelection:
    scores: torch.Tensor
    source: str
    coverage_confidence: float


def route_weighted_channel_utility(
    responses: torch.Tensor,
    down: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    if responses.ndim != 2 or down.ndim != 2 or responses.shape[1] != down.shape[1]:
        raise ValueError("Responses and down projection are not channel-aligned")
    if route_weights.ndim != 1 or route_weights.shape[0] != responses.shape[0]:
        raise ValueError("Route weights must have one value per response row")
    weighted_energy, down_energy = channel_utility_components(responses, down, route_weights)
    return weighted_energy * down_energy


def channel_utility_components(
    responses: torch.Tensor,
    down: torch.Tensor,
    route_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if responses.ndim != 2 or down.ndim != 2 or responses.shape[1] != down.shape[1]:
        raise ValueError("Responses and down projection are not channel-aligned")
    if route_weights.ndim != 1 or route_weights.shape[0] != responses.shape[0]:
        raise ValueError("Route weights must have one value per response row")
    weighted_energy = (responses.float() * route_weights.float().unsqueeze(1)).square().sum(0)
    down_energy = down.float().square().sum(0)
    return weighted_energy, down_energy


def normalized_utility(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("Channel utility must be one-dimensional")
    normalized = scores.float().clone()
    normalized[~torch.isfinite(normalized)] = 0.0
    scale = normalized.abs().mean().clamp_min(torch.finfo(normalized.dtype).eps)
    return normalized / scale


def select_channel_scores(
    calibrated: torch.Tensor | None,
    structural: torch.Tensor | None,
    aimer: torch.Tensor,
    fit_token_count: int,
    fit_route_mass: float,
    min_fit_tokens: int,
    min_fit_route_mass: float,
) -> ChannelScoreSelection:
    if aimer.ndim != 1:
        raise ValueError("Stable-AIMER scores must be one-dimensional")
    if min_fit_tokens <= 0 or min_fit_route_mass <= 0.0:
        raise ValueError("Coverage thresholds must be positive")
    for name, scores in (("calibrated", calibrated), ("structural", structural)):
        if scores is not None and scores.shape != aimer.shape:
            raise ValueError(f"{name.capitalize()} scores are not channel-aligned")
    coverage_confidence = min(
        1.0,
        max(0.0, float(fit_token_count) / min_fit_tokens),
        max(0.0, float(fit_route_mass) / min_fit_route_mass),
    )
    if calibrated is not None and coverage_confidence >= 1.0:
        return ChannelScoreSelection(calibrated.float(), "real_token_route_weighted", coverage_confidence)
    if structural is not None:
        if calibrated is not None and coverage_confidence > 0.0:
            scores = (
                coverage_confidence * normalized_utility(calibrated)
                + (1.0 - coverage_confidence) * normalized_utility(structural)
            )
            return ChannelScoreSelection(scores, "real_token_structural_shrinkage", coverage_confidence)
        return ChannelScoreSelection(structural.float(), "structural_probe_fallback", coverage_confidence)
    if calibrated is not None:
        return ChannelScoreSelection(
            calibrated.float(),
            "real_token_undercovered_no_structural",
            coverage_confidence,
        )
    return ChannelScoreSelection(aimer.float(), "stable_aimer_fallback", coverage_confidence)


def nested_order(
    scores: torch.Tensor,
    zero_mask: torch.Tensor,
    tie_break_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    if scores.ndim != 1 or zero_mask.shape != scores.shape:
        raise ValueError("Scores and zero mask must be one-dimensional and aligned")
    if tie_break_scores is not None and tie_break_scores.shape != scores.shape:
        raise ValueError("Tie-break scores must be channel-aligned")
    finite_scores = scores.float().clone()
    finite_scores[~torch.isfinite(finite_scores)] = -torch.inf
    finite_tie_break = torch.zeros_like(finite_scores)
    if tie_break_scores is not None:
        finite_tie_break = tie_break_scores.float().clone()
        finite_tie_break[~torch.isfinite(finite_tie_break)] = -torch.inf
    channel_ids = torch.arange(scores.numel(), device=scores.device)
    order = sorted(
        channel_ids.tolist(),
        key=lambda channel: (
            bool(zero_mask[channel].item()),
            -float(finite_scores[channel].item()),
            -float(finite_tie_break[channel].item()),
            int(channel),
        ),
    )
    return torch.tensor(order, dtype=torch.long, device=scores.device)


def nested_rankings_by_width(orders: torch.Tensor, widths: tuple[int, ...]) -> torch.Tensor:
    if orders.ndim != 2:
        raise ValueError("Expert channel orders must be two-dimensional")
    if not widths:
        raise ValueError("At least one candidate width is required")
    channels = orders.shape[1]
    if any(width <= 0 or width > channels for width in widths):
        raise ValueError("Candidate widths must fit inside the full channel order")
    expected = torch.arange(channels, device=orders.device).expand_as(orders)
    if not torch.equal(torch.sort(orders.to(torch.long), dim=1).values, expected):
        raise ValueError("Each expert order must be a full channel permutation")
    return orders.to(torch.long).unsqueeze(1).repeat(1, len(widths), 1)


def validate_response_energy_table(
    table: torch.Tensor | None,
    split: str,
    layer_count: int,
    num_experts: int,
    intermediate_size: int,
) -> None:
    if table is None:
        return
    expected_shape = (layer_count, num_experts, intermediate_size)
    if tuple(table.shape) != expected_shape:
        raise ValueError(
            f"Calibration {split} response-energy shape is {tuple(table.shape)}, expected {expected_shape}"
        )
    if not bool(torch.isfinite(table).all()):
        raise ValueError(f"Calibration {split} response energy contains non-finite values")
    if bool((table < 0).any()):
        raise ValueError(f"Calibration {split} response energy contains negative values")


def prefix_losses(
    inputs: torch.Tensor,
    route_weights: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    order: torch.Tensor,
    widths: tuple[int, ...],
    activation: str,
) -> list[float]:
    if inputs.shape[0] == 0:
        return [float("nan") for _ in widths]
    responses = swiglu_response(inputs, gate, up, activation=activation)
    full_output = responses @ down.float().transpose(0, 1)
    return [
        weighted_output_loss(
            full_output,
            output_for_set(responses, down, order[:width]),
            route_weights,
        )
        for width in widths
    ]


def structural_utility(
    model_path: Path,
    weight_map: dict[str, str],
    adapter: PurePseudoModelAdapter,
    layer_id: int,
    expert_id: int,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    structural_audit: dict[str, Any] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if structural_audit is None:
        return None
    layer_audit = structural_audit.get("layers", {}).get(layer_id)
    if layer_audit is None:
        layer_audit = structural_audit.get("layers", {}).get(str(layer_id))
    if layer_audit is None:
        return None
    required = ("expert_probes", "selected_experts", "selected_weights")
    if any(key not in layer_audit for key in required):
        return None
    probes = layer_audit["expert_probes"].to(device)
    selected_experts = layer_audit["selected_experts"].to(device)
    selected_weights = layer_audit["selected_weights"].to(device)
    probe_sets = build_probe_sets(probes, selected_experts, selected_weights, expert_id)
    if probe_sets["coverage_probes"].shape[0] == 0:
        return None
    responses = swiglu_response(
        probe_sets["coverage_probes"], gate, up,
        activation=adapter.channel_architecture.activation,
    )
    return responses.float().square().sum(0) * down.float().square().sum(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calibrated nested CHANNEL ranking artifacts.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--structural-artifact-dir", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", required=True)
    parser.add_argument("--min-fit-tokens", type=int, default=8)
    parser.add_argument("--min-fit-route-mass", type=float, default=0.5)
    parser.add_argument("--effective-zero-threshold", type=float, default=1.0e-12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    capture_path = args.capture.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.channel_architecture
    widths = tuple(sorted(set(int(width) for width in args.widths)))
    if not widths:
        raise ValueError("At least one candidate width is required")
    for width in widths:
        architecture.validate_width(width)
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    if int(capture.get("schema_version", -1)) not in {1, 2}:
        raise ValueError("Unsupported routed-token capture schema")
    if capture["model_family"] != adapter.model_family:
        raise ValueError("Capture and model family do not match")
    if Path(capture["model_path"]).resolve() != model_path:
        raise ValueError("Capture and requested model paths do not match")
    capture_architecture = capture["architecture"]
    if int(capture_architecture["source_intermediate_size"]) != architecture.source_intermediate_size:
        raise ValueError("Capture and checkpoint expert widths do not match")
    if int(capture_architecture["hidden_size"]) != architecture.hidden_size:
        raise ValueError("Capture and checkpoint hidden sizes do not match")
    model_provenance = capture.get("model_provenance")
    if model_provenance is not None:
        expected_model_hashes = {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        }
        if model_provenance != expected_model_hashes:
            raise ValueError("Capture checkpoint provenance does not match the requested model")
    structural_audit = None
    if args.structural_artifact_dir is not None:
        audit_path = args.structural_artifact_dir.expanduser().resolve() / "routing_audit.pt"
        if audit_path.is_file():
            structural_audit = torch.load(audit_path, map_location="cpu", weights_only=True)
    config = NapsV2Config(effective_zero_threshold=args.effective_zero_threshold)
    device = torch.device(args.device)
    split_data = capture["splits"]
    rows_by_split = {split: split_data[split]["layers"] for split in ("fit", "holdout")}
    layer_ids = [int(layer_id) for layer_id in capture["layers"]]
    layer_positions = {layer_id: position for position, layer_id in enumerate(layer_ids)}
    fit_energy_table = split_data["fit"].get("route_weighted_response_energy")
    holdout_energy_table = split_data["holdout"].get("route_weighted_response_energy")
    validate_response_energy_table(
        fit_energy_table,
        "fit",
        len(layer_ids),
        architecture.num_experts,
        architecture.source_intermediate_size,
    )
    validate_response_energy_table(
        holdout_energy_table,
        "holdout",
        len(layer_ids),
        architecture.num_experts,
        architecture.source_intermediate_size,
    )
    tables = {}
    layer_diagnostics = []
    for layer_id in layer_ids:
        layer_rows = []
        rankings = []
        channel_scores = []
        zero_masks = []
        fit_losses = []
        holdout_losses = []
        score_sources = []
        coverage_confidences = []
        fit_counts = []
        holdout_counts = []
        fit_captured_masses = []
        holdout_captured_masses = []
        fit_total_counts = []
        holdout_total_counts = []
        fit_total_masses = []
        holdout_total_masses = []
        score_statistic_scopes = []
        calibrated_scores = []
        route_weighted_response_energies = []
        holdout_route_weighted_response_energies = []
        down_channel_energies = []
        structural_scores = []
        aimer_scores = []
        for expert_id, gate, up, down in iter_expert_weights(
            model_path, weight_map, adapter, layer_id, device
        ):
            fit_record = rows_by_split["fit"].get(layer_id, rows_by_split["fit"].get(str(layer_id)))[
                expert_id
            ]
            holdout_record = rows_by_split["holdout"].get(layer_id, rows_by_split["holdout"].get(str(layer_id)))[
                expert_id
            ]
            fit_inputs = fit_record["inputs"].to(device)
            fit_weights = fit_record["route_weights"].to(device)
            holdout_inputs = holdout_record["inputs"].to(device)
            holdout_weights = holdout_record["route_weights"].to(device)
            zero_mask = effective_zero_mask(gate, up, down, config.effective_zero_threshold)
            aimer = stable_concat_score(gate, up, down, config)
            if fit_energy_table is not None:
                weighted_response_energy = fit_energy_table[layer_positions[layer_id], expert_id].to(device)
                calibrated = weighted_response_energy.float() * down.float().square().sum(0)
                statistic_scope = "all_routed_tokens"
            elif fit_inputs.shape[0] > 0:
                responses = swiglu_response(
                    fit_inputs, gate, up, activation=architecture.activation
                )
                weighted_response_energy, down_energy = channel_utility_components(
                    responses, down, fit_weights
                )
                calibrated = weighted_response_energy * down_energy
                statistic_scope = "bounded_captured_tokens"
            else:
                calibrated = None
                weighted_response_energy = None
                down_energy = down.float().square().sum(0)
                statistic_scope = "no_real_token_statistic"
            if fit_energy_table is not None:
                down_energy = down.float().square().sum(0)
            holdout_response_energy = (
                holdout_energy_table[layer_positions[layer_id], expert_id].float()
                if holdout_energy_table is not None else torch.zeros_like(down_energy)
            )
            structural = structural_utility(
                model_path, weight_map, adapter, layer_id, expert_id,
                gate, up, down, structural_audit, device,
            )
            fit_captured_mass = float(fit_record.get("captured_route_mass", fit_weights.sum().item()))
            score_token_count = (
                int(fit_record["total_route_count"])
                if statistic_scope == "all_routed_tokens" else int(fit_record["captured_token_count"])
            )
            score_route_mass = (
                float(fit_record["total_route_mass"])
                if statistic_scope == "all_routed_tokens" else fit_captured_mass
            )
            selection = select_channel_scores(
                calibrated,
                structural,
                aimer,
                score_token_count,
                score_route_mass,
                args.min_fit_tokens,
                args.min_fit_route_mass,
            )
            order = nested_order(selection.scores, zero_mask, aimer)
            channel_scores.append(selection.scores.detach().cpu())
            calibrated_scores.append(
                calibrated.detach().cpu() if calibrated is not None else torch.full_like(aimer.cpu(), torch.nan)
            )
            route_weighted_response_energies.append(
                weighted_response_energy.detach().cpu()
                if weighted_response_energy is not None else torch.zeros_like(aimer.cpu())
            )
            holdout_route_weighted_response_energies.append(holdout_response_energy.detach().cpu())
            down_channel_energies.append(down_energy.detach().cpu())
            structural_scores.append(
                structural.detach().cpu() if structural is not None else torch.full_like(aimer.cpu(), torch.nan)
            )
            aimer_scores.append(aimer.detach().cpu())
            zero_masks.append(zero_mask.detach().cpu())
            rankings.append(order.detach().cpu())
            fit_losses.append(prefix_losses(
                fit_inputs, fit_weights, gate, up, down, order, widths, architecture.activation
            ))
            holdout_losses.append(prefix_losses(
                holdout_inputs, holdout_weights, gate, up, down, order, widths, architecture.activation
            ))
            score_sources.append(selection.source)
            score_statistic_scopes.append(statistic_scope)
            coverage_confidences.append(selection.coverage_confidence)
            fit_counts.append(int(fit_record["captured_token_count"]))
            holdout_counts.append(int(holdout_record["captured_token_count"]))
            fit_captured_masses.append(fit_captured_mass)
            holdout_captured_masses.append(
                float(holdout_record.get("captured_route_mass", holdout_weights.sum().item()))
            )
            fit_total_counts.append(int(fit_record["total_route_count"]))
            holdout_total_counts.append(int(holdout_record["total_route_count"]))
            fit_total_masses.append(float(fit_record["total_route_mass"]))
            holdout_total_masses.append(float(holdout_record["total_route_mass"]))
            layer_rows.append({
                "layer_id": layer_id,
                "expert_id": expert_id,
                "score_source": selection.source,
                "score_statistic_scope": statistic_scope,
                "coverage_confidence": selection.coverage_confidence,
                "fit_token_count": fit_counts[-1],
                "holdout_token_count": holdout_counts[-1],
                "fit_captured_route_mass": fit_captured_masses[-1],
                "holdout_captured_route_mass": holdout_captured_masses[-1],
                "fit_total_route_count": fit_total_counts[-1],
                "holdout_total_route_count": holdout_total_counts[-1],
                "fit_total_route_mass": fit_total_masses[-1],
                "holdout_total_route_mass": holdout_total_masses[-1],
            })
        full_orders = nested_rankings_by_width(torch.stack(rankings), widths)
        tables[layer_id] = {
            "ranked_indices": full_orders[:, 0],
            "ranked_indices_by_width": full_orders,
            "channel_scores": torch.stack(channel_scores),
            "real_token_channel_scores": torch.stack(calibrated_scores),
            "route_weighted_response_energy": torch.stack(route_weighted_response_energies),
            "holdout_route_weighted_response_energy": torch.stack(holdout_route_weighted_response_energies),
            "down_channel_energy": torch.stack(down_channel_energies),
            "structural_channel_scores": torch.stack(structural_scores),
            "stable_aimer_scores": torch.stack(aimer_scores),
            "effective_zero_masks": torch.stack(zero_masks),
            "width_options": torch.tensor(widths, dtype=torch.long),
            "fit_losses_by_width": torch.tensor(fit_losses, dtype=torch.float32),
            "holdout_losses_by_width": torch.tensor(holdout_losses, dtype=torch.float32),
            "fit_token_counts": torch.tensor(fit_counts, dtype=torch.long),
            "holdout_token_counts": torch.tensor(holdout_counts, dtype=torch.long),
            "fit_captured_route_mass": torch.tensor(fit_captured_masses, dtype=torch.float32),
            "holdout_captured_route_mass": torch.tensor(holdout_captured_masses, dtype=torch.float32),
            "fit_total_route_counts": torch.tensor(fit_total_counts, dtype=torch.long),
            "holdout_total_route_counts": torch.tensor(holdout_total_counts, dtype=torch.long),
            "fit_total_route_mass": torch.tensor(fit_total_masses, dtype=torch.float32),
            "holdout_total_route_mass": torch.tensor(holdout_total_masses, dtype=torch.float32),
            "coverage_confidence": torch.tensor(coverage_confidences, dtype=torch.float32),
            "score_sources": score_sources,
            "score_statistic_scopes": score_statistic_scopes,
            "intermediate_size": architecture.source_intermediate_size,
            "ranking_is_nested": True,
        }
        layer_diagnostics.extend(layer_rows)
        print(f"Built CHANNEL layer {layer_id + 1}/{len(layer_ids)}", flush=True)
    rankings_payload = {
        "schema_version": 4,
        "purpose": "channel_calibrated_nested_ranking",
        "model_path": str(model_path),
        "model_family": adapter.model_family,
        "source_intermediate_size": architecture.source_intermediate_size,
        "channel_alignment": architecture.channel_alignment,
        "activation": architecture.activation,
        "width_options": widths,
        "ranking_is_nested": True,
        "capture_path": str(capture_path),
        "capture_sha256": file_sha256(capture_path),
        "calibration": capture.get("calibration"),
        "model_provenance": capture.get("model_provenance"),
        "table": tables,
        "diagnostics": layer_diagnostics,
    }
    torch.save(rankings_payload, output_dir / "rankings.pt")
    summary = {
        "schema_version": 4,
        "model_path": str(model_path),
        "model_family": adapter.model_family,
        "layers": layer_ids,
        "num_experts": architecture.num_experts,
        "source_intermediate_size": architecture.source_intermediate_size,
        "channel_alignment": architecture.channel_alignment,
        "activation": architecture.activation,
        "width_options": widths,
        "ranking_is_nested": True,
        "score_source_counts": {
            source: sum(row["score_source"] == source for row in layer_diagnostics)
            for source in sorted({row["score_source"] for row in layer_diagnostics})
        },
        "capture_path": str(capture_path),
        "capture_sha256": rankings_payload["capture_sha256"],
        "calibration": capture.get("calibration"),
        "model_provenance": capture.get("model_provenance"),
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())