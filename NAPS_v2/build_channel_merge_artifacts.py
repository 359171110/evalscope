from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_naps_v2_artifacts import (
    file_sha256,
    iter_expert_weights,
    load_weight_map,
)
from NAPS_v2.channel_merge import (
    ChannelMergeConfig,
    evaluate_channel_merge_plan,
    fit_channel_merge_plan,
)
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import swiglu_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-width CHANNEL sparse response-merge plans with an independent holdout gate."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-cap", type=int, default=32)
    parser.add_argument("--min-fit-rows", type=int, default=8)
    parser.add_argument("--min-holdout-rows", type=int, default=1)
    parser.add_argument("--min-abs-correlation", type=float, default=0.35)
    parser.add_argument("--coefficient-cap", type=float, default=2.0)
    parser.add_argument("--max-update-ratio", type=float, default=0.05)
    parser.add_argument("--min-relative-fit-improvement", type=float, default=1.0e-4)
    parser.add_argument("--holdout-relative-tolerance", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _table(payload: dict[str, Any], layer_id: int) -> dict[str, Any]:
    table = payload["table"].get(layer_id, payload["table"].get(str(layer_id)))
    if table is None:
        raise KeyError(f"Missing CHANNEL ranking table for layer {layer_id}")
    return table


def _split_layer(capture: dict[str, Any], split: str, layer_id: int) -> dict[str, Any]:
    layers = capture["splits"][split]["layers"]
    layer = layers.get(layer_id, layers.get(str(layer_id)))
    if layer is None:
        raise KeyError(f"Capture is missing {split} layer {layer_id}")
    return layer


def _record(layer: dict[str, Any], expert_id: int) -> dict[str, Any]:
    record = layer.get(expert_id, layer.get(str(expert_id)))
    if record is None:
        raise KeyError(f"Capture is missing expert {expert_id}")
    return record


def _uniform_width(profile: dict[str, Any]) -> int:
    block_size = int(profile["channel_block_size"])
    widths = profile["profile_widths"].to(torch.long) * block_size
    unique = torch.unique(widths)
    if unique.numel() != 1:
        raise ValueError("CHANNEL sparse merge currently requires a uniform-width profile")
    width = int(unique.item())
    if int(profile["padded_intermediate_size"]) != width:
        raise ValueError("CHANNEL sparse merge requires physical and logical widths to match")
    return width


def _validate_inputs(
    model_path: Path,
    artifact_dir: Path,
    capture_path: Path,
    rankings: dict[str, Any],
    profile: dict[str, Any],
    capture: dict[str, Any],
    adapter: PurePseudoModelAdapter,
) -> tuple[list[int], int]:
    for label, payload_path in (
        ("ranking", rankings.get("model_path")),
        ("profile", profile.get("model_path")),
        ("capture", capture.get("model_path")),
    ):
        if payload_path is None or Path(payload_path).resolve() != model_path:
            raise ValueError(f"{label.capitalize()} and requested model paths do not match")
    if int(rankings.get("schema_version", -1)) != 4 or not rankings.get("ranking_is_nested", False):
        raise ValueError("CHANNEL sparse merge requires schema-4 nested rankings")
    if int(profile.get("schema_version", -1)) != 4:
        raise ValueError("CHANNEL sparse merge requires a schema-4 profile")
    if int(capture.get("schema_version", -1)) != 2:
        raise ValueError("CHANNEL sparse merge requires a schema-2 routed-token capture")
    if rankings.get("capture_sha256") != file_sha256(capture_path):
        raise ValueError("CHANNEL rankings and routed-token capture provenance do not match")
    if profile.get("capture_sha256") != rankings.get("capture_sha256"):
        raise ValueError("CHANNEL profile and rankings capture provenance do not match")
    if int(rankings["source_intermediate_size"]) != adapter.intermediate_size:
        raise ValueError("CHANNEL ranking width and source checkpoint do not match")
    layer_ids = [int(layer_id) for layer_id in capture["layers"]]
    if layer_ids != list(range(adapter.num_layers)):
        raise ValueError("CHANNEL capture does not cover every model layer")
    if tuple(profile["profile_widths"].shape) != (adapter.num_layers, adapter.num_experts):
        raise ValueError("CHANNEL profile shape does not match the model")
    return layer_ids, _uniform_width(profile)


def _summary(
    plans: dict[int, dict[int, dict[str, Any]]],
    aggregate: dict[str, dict[str, float]],
) -> dict[str, Any]:
    all_plans = [plan for layer in plans.values() for plan in layer.values()]
    accepted = [plan for plan in all_plans if plan["accepted"]]
    fallback_counts = Counter(
        str(plan["fallback_reason"] or "accepted")
        for plan in all_plans
    )
    result: dict[str, Any] = {
        "total_experts": len(all_plans),
        "accepted_experts": len(accepted),
        "accepted_expert_fraction": len(accepted) / max(len(all_plans), 1),
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "merged_target_count": sum(len(plan["target_channels"]) for plan in accepted),
        "mean_merged_targets_per_accepted_expert": (
            sum(len(plan["target_channels"]) for plan in accepted) / max(len(accepted), 1)
        ),
        "mean_final_update_ratio": (
            sum(float(plan["update_ratio_final"]) for plan in accepted) / max(len(accepted), 1)
        ),
    }
    for split, values in aggregate.items():
        denominator = max(values["denominator"], 1.0e-12)
        baseline = values["baseline_residual"] / denominator
        candidate = values["candidate_residual"] / denominator
        result[f"{split}_global_baseline_loss"] = baseline
        result[f"{split}_global_gated_candidate_loss"] = candidate
        result[f"{split}_relative_global_loss_change"] = candidate / max(baseline, 1.0e-12) - 1.0
    return result


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rankings_path = artifact_dir / "rankings.pt"
    profile_path = artifact_dir / "profile.pt"
    rankings = torch.load(rankings_path, map_location="cpu", weights_only=True)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    capture_path = (
        args.capture.expanduser().resolve()
        if args.capture is not None else Path(profile["capture_path"]).expanduser().resolve()
    )
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    layer_ids, retained_width = _validate_inputs(
        model_path,
        artifact_dir,
        capture_path,
        rankings,
        profile,
        capture,
        adapter,
    )
    config = ChannelMergeConfig(
        target_cap=args.target_cap,
        min_fit_rows=args.min_fit_rows,
        min_holdout_rows=args.min_holdout_rows,
        min_abs_correlation=args.min_abs_correlation,
        coefficient_cap=args.coefficient_cap,
        max_update_ratio=args.max_update_ratio,
        min_relative_fit_improvement=args.min_relative_fit_improvement,
        holdout_relative_tolerance=args.holdout_relative_tolerance,
    )
    device = torch.device(args.device)
    plans: dict[int, dict[int, dict[str, Any]]] = {}
    aggregate = {
        split: {"denominator": 0.0, "baseline_residual": 0.0, "candidate_residual": 0.0}
        for split in ("fit", "holdout")
    }
    for layer_position, layer_id in enumerate(layer_ids, start=1):
        table = _table(rankings, layer_id)
        fit_layer = _split_layer(capture, "fit", layer_id)
        holdout_layer = _split_layer(capture, "holdout", layer_id)
        layer_plans = {}
        for expert_id, gate, up, down in iter_expert_weights(
            model_path, weight_map, adapter, layer_id, device
        ):
            fit_record = _record(fit_layer, expert_id)
            holdout_record = _record(holdout_layer, expert_id)
            fit_inputs = fit_record["inputs"].to(device)
            holdout_inputs = holdout_record["inputs"].to(device)
            fit_weights = fit_record["route_weights"].to(device).float()
            holdout_weights = holdout_record["route_weights"].to(device).float()
            fit_responses = swiglu_response(
                fit_inputs, gate, up, activation=adapter.channel_architecture.activation
            )
            holdout_responses = swiglu_response(
                holdout_inputs, gate, up, activation=adapter.channel_architecture.activation
            )
            retained = table["ranked_indices"][expert_id, :retained_width].to(device)
            utility = (
                table["route_weighted_response_energy"][expert_id].float()
                * table["down_channel_energy"][expert_id].float()
            ).to(device)
            zero_mask = table["effective_zero_masks"][expert_id].to(device)
            plan = fit_channel_merge_plan(
                fit_responses,
                fit_weights,
                holdout_responses,
                holdout_weights,
                down,
                retained,
                utility,
                zero_mask,
                config,
            )
            plan.update({
                "layer_id": layer_id,
                "expert_id": expert_id,
                "score_source": table["score_sources"][expert_id],
                "fit_total_route_count": int(fit_record["total_route_count"]),
                "holdout_total_route_count": int(holdout_record["total_route_count"]),
            })
            layer_plans[expert_id] = plan
            for split, responses, weights in (
                ("fit", fit_responses, fit_weights),
                ("holdout", holdout_responses, holdout_weights),
            ):
                metrics = evaluate_channel_merge_plan(responses, weights, down, retained, plan)
                for key in aggregate[split]:
                    aggregate[split][key] += metrics[key]
        plans[layer_id] = layer_plans
        accepted = sum(plan["accepted"] for plan in layer_plans.values())
        print(
            f"Built CHANNEL sparse merge layer {layer_position}/{len(layer_ids)}: "
            f"accepted {accepted}/{adapter.num_experts}",
            flush=True,
        )
    summary = _summary(plans, aggregate)
    payload = {
        "schema_version": 1,
        "purpose": "channel_fixed_width_sparse_response_merge",
        "method": "channel_sparse_response_merge",
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
        "rankings_path": str(rankings_path),
        "rankings_sha256": file_sha256(rankings_path),
        "profile_path": str(profile_path),
        "profile_sha256": file_sha256(profile_path),
        "capture_path": str(capture_path),
        "capture_sha256": file_sha256(capture_path),
        "calibration": profile.get("calibration"),
        "model_provenance": profile.get("model_provenance"),
        "retained_width": retained_width,
        "physical_width": int(profile["padded_intermediate_size"]),
        "holdout_used_for_acceptance": True,
        "benchmark_metrics_used": False,
        "config": vars(args) | {},
        "layers": plans,
        "summary": summary,
    }
    payload["config"] = {
        key: value
        for key, value in payload["config"].items()
        if key not in {"model_path", "artifact_dir", "capture", "output_dir", "device", "force"}
    }
    torch.save(payload, output_dir / "channel_merge_plan.pt")
    diagnostics = {
        key: value
        for key, value in payload.items()
        if key != "layers"
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())