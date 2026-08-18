from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_naps_v2_artifacts import file_sha256, iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import swiglu_response
from NAPS_v2.puzzlecomp import PuzzleCompConfig, pairwise_puzzle_compensate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate storage-equivalent channel Puzzle candidates on frozen Gemma4 routed tokens."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--pairs-per-layer", type=int, default=4)
    parser.add_argument("--retained-width", type=int, default=384)
    parser.add_argument("--reserve-channels", type=int, default=32)
    parser.add_argument("--similarity-threshold", type=float, default=0.4)
    parser.add_argument("--acceptance-tolerance", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _layer(payload: dict[str, Any], key: str, layer_id: int) -> dict[str, Any]:
    layer = payload[key].get(layer_id, payload[key].get(str(layer_id)))
    if layer is None:
        raise KeyError(f"Missing layer {layer_id} in {key}")
    return layer


def _record(capture: dict[str, Any], split: str, layer_id: int, expert_id: int) -> dict[str, Any]:
    layer = _layer(capture["splits"][split], "layers", layer_id)
    record = layer.get(expert_id, layer.get(str(expert_id)))
    if record is None:
        raise KeyError(f"Missing {split} layer {layer_id} expert {expert_id}")
    return record


def _weighted_metrics(
    full_output: torch.Tensor,
    approximate_output: torch.Tensor,
    route_weights: torch.Tensor,
) -> dict[str, float]:
    full = full_output.float()
    approximate = approximate_output.float()
    factors = route_weights.float().square()
    residual = ((full - approximate).square().sum(1) * factors).sum()
    denominator = (full.square().sum(1) * factors).sum().clamp_min(1.0e-12)
    dot = (full * approximate).sum(1)
    cosine = dot / (full.norm(dim=1) * approximate.norm(dim=1)).clamp_min(1.0e-12)
    norm_ratio = approximate.norm(dim=1) / full.norm(dim=1).clamp_min(1.0e-12)
    factor_sum = factors.sum().clamp_min(1.0e-12)
    return {
        "relative_output_loss": float((residual / denominator).item()),
        "weighted_cosine": float((cosine * factors).sum().div(factor_sum).item()),
        "weighted_norm_ratio": float((norm_ratio * factors).sum().div(factor_sum).item()),
        "residual": float(residual.item()),
        "denominator": float(denominator.item()),
    }


def _expert_output(inputs: torch.Tensor, weights: dict[str, torch.Tensor], activation: str) -> torch.Tensor:
    responses = swiglu_response(inputs, weights["gate"], weights["up"], activation=activation)
    return responses @ weights["down"].float().transpose(0, 1)


def _selected_weights(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    ranking: torch.Tensor,
    width: int,
) -> dict[str, torch.Tensor]:
    selected = ranking[:width].to(device=gate.device, dtype=torch.long)
    return {
        "gate": gate.index_select(0, selected).float(),
        "up": up.index_select(0, selected).float(),
        "down": down.index_select(1, selected).float(),
    }


def _pair_experts(scores: torch.Tensor, pair_count: int) -> list[tuple[int, int, float]]:
    normalized = torch.nn.functional.normalize(scores.float(), dim=1)
    similarities = normalized @ normalized.transpose(0, 1)
    similarities.fill_diagonal_(-torch.inf)
    available = set(range(scores.shape[0]))
    pairs = []
    while len(available) >= 2 and len(pairs) < pair_count:
        candidates = torch.tensor(sorted(available), dtype=torch.long)
        submatrix = similarities.index_select(0, candidates).index_select(1, candidates)
        flat_position = int(torch.argmax(submatrix).item())
        row = flat_position // submatrix.shape[1]
        column = flat_position % submatrix.shape[1]
        left = int(candidates[row].item())
        right = int(candidates[column].item())
        if left == right:
            raise RuntimeError("Expert pairing selected the same expert twice")
        pairs.append((left, right, float(similarities[left, right].item())))
        available.remove(left)
        available.remove(right)
    return pairs


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No Puzzle validation records were generated")
    summary: dict[str, Any] = {
        "pair_count": len(records),
        "expert_count": 2 * len(records),
        "accepted_pairs": sum(bool(record["accepted"]) for record in records),
    }
    for split in ("fit", "holdout"):
        for method in ("mask_384", "channel_416", "puzzle_416"):
            sides = [record[split][side][method] for record in records for side in ("left", "right")]
            residual = sum(item["residual"] for item in sides)
            denominator = max(sum(item["denominator"] for item in sides), 1.0e-12)
            summary[f"{split}_{method}_global_loss"] = residual / denominator
            summary[f"{split}_{method}_mean_cosine"] = sum(item["weighted_cosine"] for item in sides) / len(sides)
            summary[f"{split}_{method}_mean_norm_ratio"] = (
                sum(item["weighted_norm_ratio"] for item in sides) / len(sides)
            )
    mask_loss = summary["holdout_mask_384_global_loss"]
    for method in ("channel_416", "puzzle_416"):
        summary[f"holdout_{method}_relative_to_mask_384"] = (
            summary[f"holdout_{method}_global_loss"] / max(mask_loss, 1.0e-12) - 1.0
        )
    return summary


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    rankings_path = args.rankings.expanduser().resolve()
    capture_path = args.capture.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings = torch.load(rankings_path, map_location="cpu", weights_only=True)
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    if Path(rankings["model_path"]).resolve() != model_path or Path(capture["model_path"]).resolve() != model_path:
        raise ValueError("Model, rankings, and capture paths do not match")
    if rankings.get("capture_sha256") != file_sha256(capture_path):
        raise ValueError("Rankings and capture provenance do not match")
    if args.reserve_channels <= 0 or args.retained_width + args.reserve_channels > rankings["source_intermediate_size"]:
        raise ValueError("Reserve channels produce an invalid effective width")
    device = torch.device(args.device)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    activation = adapter.channel_architecture.activation
    reserve_fraction = args.reserve_channels / int(rankings["source_intermediate_size"])
    config = PuzzleCompConfig(
        similarity_threshold=args.similarity_threshold,
        reserve_fraction=reserve_fraction,
        acceptance_tolerance=args.acceptance_tolerance,
        activation=activation,
    )
    records = []
    materialized: dict[int, dict[str, Any]] = {}
    for layer_id in args.layers:
        table = _layer(rankings, "table", layer_id)
        pair_scores = table["real_token_channel_scores"]
        pairs = _pair_experts(pair_scores, args.pairs_per_layer)
        weights = {
            expert_id: {"gate": gate, "up": up, "down": down}
            for expert_id, gate, up, down in iter_expert_weights(model_path, weight_map, adapter, layer_id, device)
        }
        layer_materialized = {}
        for left_id, right_id, pair_similarity in pairs:
            left_weights = weights[left_id]
            right_weights = weights[right_id]
            left_fit = _record(capture, "fit", layer_id, left_id)
            right_fit = _record(capture, "fit", layer_id, right_id)
            left_holdout = _record(capture, "holdout", layer_id, left_id)
            right_holdout = _record(capture, "holdout", layer_id, right_id)
            left_ranking = table["ranked_indices"][left_id].to(device)
            right_ranking = table["ranked_indices"][right_id].to(device)
            result = pairwise_puzzle_compensate(
                left_weights["gate"], left_weights["up"], left_weights["down"],
                right_weights["gate"], right_weights["up"], right_weights["down"],
                left_ranking, right_ranking, args.retained_width,
                left_fit["inputs"].to(device), right_fit["inputs"].to(device),
                left_holdout["inputs"].to(device), right_holdout["inputs"].to(device),
                left_holdout["route_weights"].to(device), right_holdout["route_weights"].to(device),
                table["effective_zero_masks"][left_id].to(device),
                table["effective_zero_masks"][right_id].to(device),
                config,
            )
            record: dict[str, Any] = {
                "layer_id": layer_id,
                "left_expert_id": left_id,
                "right_expert_id": right_id,
                "pair_score_cosine": pair_similarity,
                "accepted": bool(result["accepted"]),
                "diagnostics": result["diagnostics"],
                "fit": {},
                "holdout": {},
            }
            if result["accepted"]:
                layer_materialized[f"{left_id}_{right_id}"] = {
                    "left_expert_id": left_id,
                    "right_expert_id": right_id,
                    "accepted": True,
                    "left": {key: value.cpu() for key, value in result["candidate_left"].items()},
                    "right": {key: value.cpu() for key, value in result["candidate_right"].items()},
                }
            for split, left_source, right_source in (
                ("fit", left_fit, right_fit),
                ("holdout", left_holdout, right_holdout),
            ):
                for side, expert_id, source, source_weights, ranking, puzzle_weights in (
                    ("left", left_id, left_source, left_weights, left_ranking, result["candidate_left"]),
                    ("right", right_id, right_source, right_weights, right_ranking, result["candidate_right"]),
                ):
                    inputs = source["inputs"].to(device)
                    route_weights = source["route_weights"].to(device)
                    full_output = _expert_output(inputs, source_weights, activation)
                    candidates = {
                        "mask_384": _selected_weights(
                            source_weights["gate"], source_weights["up"], source_weights["down"],
                            ranking, args.retained_width,
                        ),
                        "channel_416": _selected_weights(
                            source_weights["gate"], source_weights["up"], source_weights["down"],
                            ranking, args.retained_width + args.reserve_channels,
                        ),
                        "puzzle_416": puzzle_weights,
                    }
                    record[split][side] = {
                        method: _weighted_metrics(
                            full_output,
                            _expert_output(inputs, candidate, activation),
                            route_weights,
                        )
                        for method, candidate in candidates.items()
                    }
                    record[split][side]["expert_id"] = expert_id
            records.append(record)
        materialized[layer_id] = layer_materialized
        print(f"Validated layer {layer_id}: {len(pairs)} pairs", flush=True)
    summary = _summarize(records)
    payload = {
        "schema_version": 1,
        "purpose": "channel_puzzle_scheme_a_and_materialized_scheme_b_validation",
        "model_path": str(model_path),
        "rankings_path": str(rankings_path),
        "rankings_sha256": file_sha256(rankings_path),
        "capture_path": str(capture_path),
        "capture_sha256": file_sha256(capture_path),
        "layers": args.layers,
        "pairs_per_layer": args.pairs_per_layer,
        "retained_width": args.retained_width,
        "reserve_channels": args.reserve_channels,
        "effective_width": args.retained_width + args.reserve_channels,
        "pair_storage_width": 2 * args.retained_width,
        "benchmark_metrics_used": False,
        "config": {
            "similarity_threshold": args.similarity_threshold,
            "reserve_fraction": reserve_fraction,
            "acceptance_tolerance": args.acceptance_tolerance,
        },
        "summary": summary,
        "records": records,
    }
    torch.save(materialized, output_dir / "materialized_width416_pairs.pt")
    (output_dir / "diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())