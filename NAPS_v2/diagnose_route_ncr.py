from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.analyze_aimer_proxy_damage import spearman, top_overlap
from NAPS_v2.build_naps_v2_artifacts import file_sha256, iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import output_for_set, swiglu_response


METHODS = {
    "stable_aimer": ("stable_aimer_scores", "stable_aimer_ranked_indices"),
    "isotropic_gaussian_response": (
        "isotropic_gaussian_response_scores",
        "isotropic_gaussian_ranked_indices",
    ),
    "route_ncr": ("route_ncr_scores", "route_ncr_ranked_indices"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose RouteNCR against data-free baselines and CHANNEL oracle.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--route-rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-rankings", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def layer_table(payload: dict[str, Any], layer_id: int) -> dict[str, Any]:
    table = payload["table"].get(layer_id, payload["table"].get(str(layer_id)))
    if table is None:
        raise KeyError(f"Missing ranking table for layer {layer_id}")
    return table


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(torch.tensor(values, dtype=torch.float64).mean().item())


def validate_route_rankings(
    model_path: Path,
    rankings: dict[str, Any],
    adapter: PurePseudoModelAdapter,
) -> list[int]:
    if rankings.get("method") != "route_ncr" or rankings.get("purpose") != "strict_data_free_channel_ranking":
        raise ValueError("RouteNCR diagnostics require a strict data-free RouteNCR artifact")
    if Path(rankings["model_path"]).resolve() != model_path:
        raise ValueError("RouteNCR rankings and requested model paths do not match")
    construction = rankings.get("construction", {})
    if construction.get("data_free") is not True or any(
        construction.get(key) is not False
        for key in ("real_tokens_used", "text_used", "datasets_used", "labels_used")
    ):
        raise ValueError("RouteNCR artifact does not declare strict data-free construction")
    layer_ids = [int(layer_id) for layer_id in rankings["layer_ids"]]
    if not layer_ids:
        raise ValueError("RouteNCR artifact contains no layers")
    channels = adapter.intermediate_size
    expected = torch.arange(channels)
    for layer_id in layer_ids:
        table = layer_table(rankings, layer_id)
        for score_key, order_key in METHODS.values():
            scores = table[score_key]
            orders = table[order_key].to(torch.long)
            expected_shape = (adapter.num_experts, channels)
            if tuple(scores.shape) != expected_shape or tuple(orders.shape) != expected_shape:
                raise ValueError(f"Layer {layer_id} {score_key} shape is invalid")
            if not bool(torch.isfinite(scores).all()):
                raise ValueError(f"Layer {layer_id} {score_key} contains non-finite values")
            if not torch.equal(torch.sort(orders, dim=1).values, expected.expand_as(orders)):
                raise ValueError(f"Layer {layer_id} {order_key} is not a channel permutation")
    return layer_ids


def pairwise_data_free_metrics(
    rankings: dict[str, Any],
    layer_ids: list[int],
    widths: tuple[int, ...],
) -> dict[str, Any]:
    pairs = (
        ("route_ncr", "isotropic_gaussian_response"),
        ("route_ncr", "stable_aimer"),
        ("isotropic_gaussian_response", "stable_aimer"),
    )
    metrics = {}
    for left_name, right_name in pairs:
        correlations = []
        overlaps = {width: [] for width in widths}
        for layer_id in layer_ids:
            table = layer_table(rankings, layer_id)
            left_scores = table[METHODS[left_name][0]]
            right_scores = table[METHODS[right_name][0]]
            for expert_id in range(left_scores.shape[0]):
                correlations.append(spearman(left_scores[expert_id], right_scores[expert_id], 1.0e-12))
                for width in widths:
                    overlaps[width].append(
                        top_overlap(left_scores[expert_id], right_scores[expert_id], width)
                    )
        label = f"{left_name}_vs_{right_name}"
        metrics[label] = {
            "spearman_mean": mean(correlations),
            "top_k_overlap_mean": {
                str(width): mean(values)
                for width, values in overlaps.items()
            },
        }
    return metrics


def oracle_rank_metrics(
    route_rankings: dict[str, Any],
    oracle_rankings: dict[str, Any],
    layer_ids: list[int],
    widths: tuple[int, ...],
) -> dict[str, Any]:
    metrics = {
        method: {"spearman": [], "overlap": {width: [] for width in widths}}
        for method in METHODS
    }
    score_source_counts: dict[str, int] = {}
    for layer_id in layer_ids:
        route_table = layer_table(route_rankings, layer_id)
        oracle_table = layer_table(oracle_rankings, layer_id)
        oracle_scores = oracle_table["channel_scores"].float()
        oracle_orders = oracle_table["ranked_indices"].to(torch.long)
        if oracle_scores.shape != route_table["route_ncr_scores"].shape:
            raise ValueError(f"Layer {layer_id} CHANNEL oracle shape does not match RouteNCR")
        for source in oracle_table.get("score_sources", []):
            score_source_counts[source] = score_source_counts.get(source, 0) + 1
        for method, (score_key, order_key) in METHODS.items():
            scores = route_table[score_key]
            orders = route_table[order_key].to(torch.long)
            for expert_id in range(scores.shape[0]):
                metrics[method]["spearman"].append(
                    spearman(scores[expert_id], oracle_scores[expert_id], 1.0e-12)
                )
                for width in widths:
                    retained = set(orders[expert_id, :width].tolist())
                    oracle_retained = set(oracle_orders[expert_id, :width].tolist())
                    metrics[method]["overlap"][width].append(len(retained & oracle_retained) / width)
    return {
        "score_source_counts": score_source_counts,
        "methods": {
            method: {
                "spearman_mean": mean(values["spearman"]),
                "top_k_overlap_mean": {
                    str(width): mean(overlaps)
                    for width, overlaps in values["overlap"].items()
                },
            }
            for method, values in metrics.items()
        },
    }


def holdout_reconstruction_metrics(
    model_path: Path,
    route_rankings: dict[str, Any],
    capture: dict[str, Any],
    adapter: PurePseudoModelAdapter,
    layer_ids: list[int],
    widths: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    if Path(capture["model_path"]).resolve() != model_path:
        raise ValueError("Holdout capture and requested model paths do not match")
    capture_layers = {int(layer_id) for layer_id in capture["layers"]}
    if not set(layer_ids).issubset(capture_layers):
        raise ValueError("Holdout capture does not contain every RouteNCR layer")
    holdout_layers = capture["splits"]["holdout"]["layers"]
    weight_map = load_weight_map(model_path)
    totals = {
        method: {width: {"residual": 0.0, "denominator": 0.0, "experts": 0} for width in widths}
        for method in METHODS
    }
    for layer_position, layer_id in enumerate(layer_ids, start=1):
        table = layer_table(route_rankings, layer_id)
        holdout_layer = holdout_layers.get(layer_id, holdout_layers.get(str(layer_id)))
        if holdout_layer is None:
            raise KeyError(f"Holdout capture is missing layer {layer_id}")
        for expert_id, gate, up, down in iter_expert_weights(
            model_path,
            weight_map,
            adapter,
            layer_id,
            device,
        ):
            record = holdout_layer.get(expert_id, holdout_layer.get(str(expert_id)))
            if record is None:
                raise KeyError(f"Holdout capture is missing layer {layer_id} expert {expert_id}")
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
            for method, (_, order_key) in METHODS.items():
                order = table[order_key][expert_id].to(device)
                for width in widths:
                    retained_output = output_for_set(responses, down, order[:width])
                    residual = float(
                        ((full_output - retained_output).square().sum(1) * factors).sum().item()
                    )
                    totals[method][width]["residual"] += residual
                    totals[method][width]["denominator"] += denominator
                    totals[method][width]["experts"] += 1
        print(f"Diagnosed RouteNCR holdout layer {layer_position}/{len(layer_ids)}", flush=True)
    return {
        method: {
            str(width): {
                "normalized_frobenius_error": math.sqrt(
                    values["residual"] / max(values["denominator"], 1.0e-12)
                ),
                "squared_relative_error": values["residual"] / max(values["denominator"], 1.0e-12),
                "compared_experts": values["experts"],
            }
            for width, values in width_totals.items()
        }
        for method, width_totals in totals.items()
    }


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    route_path = args.route_rankings.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    route_rankings = torch.load(route_path, map_location="cpu", weights_only=True)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    layer_ids = validate_route_rankings(model_path, route_rankings, adapter)
    widths = tuple(int(width) for width in route_rankings["width_options"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "route_ncr_ranking_diagnostic",
        "model_path": str(model_path),
        "model_family": adapter.model_family,
        "route_rankings_path": str(route_path),
        "route_rankings_sha256": file_sha256(route_path),
        "layer_ids": layer_ids,
        "width_options": widths,
        "pairwise_data_free": pairwise_data_free_metrics(route_rankings, layer_ids, widths),
        "oracle_available": False,
    }
    if (args.oracle_rankings is None) != (args.capture is None):
        raise ValueError("CHANNEL oracle rankings and capture must be provided together")
    if args.oracle_rankings is not None and args.capture is not None:
        oracle_path = args.oracle_rankings.expanduser().resolve()
        capture_path = args.capture.expanduser().resolve()
        oracle_rankings = torch.load(oracle_path, map_location="cpu", weights_only=True)
        capture = torch.load(capture_path, map_location="cpu", weights_only=True)
        if Path(oracle_rankings["model_path"]).resolve() != model_path:
            raise ValueError("CHANNEL oracle and requested model paths do not match")
        payload.update({
            "oracle_available": True,
            "oracle_rankings_path": str(oracle_path),
            "oracle_rankings_sha256": file_sha256(oracle_path),
            "capture_path": str(capture_path),
            "capture_sha256": file_sha256(capture_path),
            "oracle_ranking_metrics": oracle_rank_metrics(
                route_rankings,
                oracle_rankings,
                layer_ids,
                widths,
            ),
            "holdout_reconstruction": holdout_reconstruction_metrics(
                model_path,
                route_rankings,
                capture,
                adapter,
                layer_ids,
                widths,
                torch.device(args.device),
            ),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())