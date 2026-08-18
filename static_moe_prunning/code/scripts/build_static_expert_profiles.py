from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.static_expert_pruning import build_static_profile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sorted_layer_ids(table: dict) -> list[int]:
    return sorted(int(layer_id) for layer_id in table)


def _lookup(table: dict, layer_id: int):
    if layer_id in table:
        return table[layer_id]
    return table[str(layer_id)]


def load_channel_inputs(
    path: Path,
) -> tuple[dict, list[int], torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("split") != "train":
        raise ValueError("channel cache must be calibrated on the train split.")
    if int(payload.get("sequence_length", -1)) <= 0:
        raise ValueError("channel cache must use a positive sequence length.")
    table = payload["table"]
    route_table = payload["route_counts"]
    layer_ids = _sorted_layer_ids(table)
    coverage = torch.stack(
        [_lookup(table, layer_id)["block_coverage_scores"].float() for layer_id in layer_ids]
    )
    route_counts = torch.stack(
        [_lookup(route_table, layer_id).float() for layer_id in layer_ids]
    )
    if coverage.shape[:2] != route_counts.shape:
        raise ValueError("channel coverage and route-count shapes do not match.")
    return payload, layer_ids, coverage, route_counts


def load_expert_scores(path: Path, layer_ids: list[int]) -> tuple[dict, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    table = payload["table"]
    scores = torch.stack([_lookup(table, layer_id).float() for layer_id in layer_ids])
    return payload, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze train-calibrated physical-expert static-width profiles."
    )
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("uniform", "rms", "route_rms", "dual_route_rms"),
        required=True,
    )
    parser.add_argument("--target-pruning-ratio", type=float, required=True)
    parser.add_argument(
        "--allocation-scope",
        choices=("global", "per_layer"),
        default="global",
    )
    parser.add_argument(
        "--retained-blocks-per-layer",
        type=int,
        default=None,
        help="Exact retained blocks in every layer for allocation-scope=per_layer.",
    )
    parser.add_argument("--min-blocks-per-expert", type=int, default=0)
    parser.add_argument("--amp-score-cache", type=Path, default=None)
    parser.add_argument("--aimer-score-cache", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = float(args.target_pruning_ratio)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target-pruning-ratio must be in [0, 1].")

    channel_payload, layer_ids, coverage, route_counts = load_channel_inputs(
        args.channel_cache
    )
    amp_payload = None
    aimer_payload = None
    amp = None
    aimer = None
    if args.mode == "dual_route_rms":
        if args.amp_score_cache is None or args.aimer_score_cache is None:
            raise ValueError("dual_route_rms requires AMP and AIMER score caches.")
        amp_payload, amp = load_expert_scores(args.amp_score_cache, layer_ids)
        aimer_payload, aimer = load_expert_scores(args.aimer_score_cache, layer_ids)
        if amp.shape != coverage.shape[:2] or aimer.shape != coverage.shape[:2]:
            raise ValueError("AMP/AIMER score shapes do not match the channel cache.")

    maximum_blocks = int(coverage.numel())
    layer_maximum_blocks = int(coverage.shape[1] * coverage.shape[2])
    total_blocks_by_layer = None
    if args.allocation_scope == "per_layer":
        retained_per_layer = args.retained_blocks_per_layer
        if retained_per_layer is None:
            retained_per_layer = int(round(layer_maximum_blocks * (1.0 - target)))
        if not 0 <= int(retained_per_layer) <= layer_maximum_blocks:
            raise ValueError(
                f"retained-blocks-per-layer must be in [0, {layer_maximum_blocks}]."
            )
        total_blocks_by_layer = torch.full(
            (int(coverage.shape[0]),),
            int(retained_per_layer),
            dtype=torch.long,
        )
        total_blocks = int(total_blocks_by_layer.sum().item())
    else:
        if args.retained_blocks_per_layer is not None:
            raise ValueError(
                "retained-blocks-per-layer requires allocation-scope=per_layer."
            )
        total_blocks = int(round(maximum_blocks * (1.0 - target)))
    widths = build_static_profile(
        coverage,
        mode=args.mode,
        total_blocks=total_blocks,
        total_blocks_by_layer=total_blocks_by_layer,
        route_counts=route_counts,
        amp=amp,
        aimer=aimer,
        min_blocks_per_expert=args.min_blocks_per_expert,
    )
    actual_pruning = 1.0 - int(widths.sum().item()) / maximum_blocks
    profile_digest = hashlib.sha256(widths.numpy().tobytes()).hexdigest()

    cache_provenance = {
        "calibration": {
            "sha256": channel_payload.get("calibration_cache_file_sha256"),
            "input_ids_sha256": channel_payload.get("calibration_input_ids_sha256"),
            "protocol_name": channel_payload.get("calibration_source", {}).get("protocol_name"),
            "split": channel_payload.get("split"),
            "sequence_length": channel_payload.get("sequence_length"),
            "calibration_sequences": channel_payload.get("calibration_sequences"),
            "calibration_tokens": channel_payload.get("calibration_tokens"),
        },
        "channel": {
            "path": str(args.channel_cache.resolve()),
            "sha256": file_sha256(args.channel_cache),
            "score_mode": channel_payload.get("score_mode"),
            "dataset": channel_payload.get("dataset"),
            "split": channel_payload.get("split"),
            "sequence_length": channel_payload.get("sequence_length"),
            "calibration_sequences": channel_payload.get("calibration_sequences"),
            "calibration_tokens": channel_payload.get("calibration_tokens"),
        }
    }
    if args.amp_score_cache is not None:
        cache_provenance["amp"] = {
            "path": str(args.amp_score_cache.resolve()),
            "sha256": file_sha256(args.amp_score_cache),
            "method": None if amp_payload is None else amp_payload.get("method"),
            "data_dependency": "weight_only",
        }
    if args.aimer_score_cache is not None:
        cache_provenance["aimer"] = {
            "path": str(args.aimer_score_cache.resolve()),
            "sha256": file_sha256(args.aimer_score_cache),
            "method": None if aimer_payload is None else aimer_payload.get("method"),
            "data_dependency": "weight_only",
        }

    payload = {
        "schema_version": 1,
        "method": f"static_expert_{args.mode}",
        "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": channel_payload.get("model_path"),
        "dataset": channel_payload.get("dataset"),
        "calibration_split": channel_payload.get("split"),
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": len(layer_ids),
        "num_experts": int(widths.shape[1]),
        "num_blocks": int(coverage.shape[2]),
        "channel_block_size": int(channel_payload.get("block_size", 0)),
        "target_pruning_ratio": target,
        "actual_structural_pruning_ratio": actual_pruning,
        "allocation_scope": args.allocation_scope,
        "target_blocks_by_layer": (
            None if total_blocks_by_layer is None else total_blocks_by_layer.tolist()
        ),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "total_blocks": int(widths.sum().item()),
        "maximum_blocks": maximum_blocks,
        "min_blocks_per_expert": int(args.min_blocks_per_expert),
        "profile_widths": widths.cpu(),
        "profile_sha256": profile_digest,
        "cache_provenance": cache_provenance,
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_profile)
    summary_path = args.output_profile.with_suffix(".json")
    summary = {key: value for key, value in payload.items() if key != "profile_widths"}
    summary["width_histogram"] = {
        str(int(width)): int(count)
        for width, count in zip(*torch.unique(widths, return_counts=True))
    }
    summary["output_profile"] = str(args.output_profile.resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output_profile.resolve())
    print(summary_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
