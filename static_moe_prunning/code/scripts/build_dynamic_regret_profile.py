from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.static_expert_pruning import (
    allocate_static_prefix_widths,
    build_protected_min_widths,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a frozen dynamic-regret teacher into a static profile."
    )
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float, required=True)
    parser.add_argument("--min-blocks-per-expert", type=int, default=0)
    parser.add_argument(
        "--value-key",
        choices=("block_values", "unconditional_block_values"),
        default="block_values",
    )
    parser.add_argument(
        "--protected-expert",
        action="append",
        default=[],
        metavar="LAYER:EXPERT:MIN_WIDTH",
        help="Protect a physical expert with a minimum prefix width; repeatable.",
    )
    return parser.parse_args()


def parse_protected_experts(specs: list[str]) -> list[tuple[int, int, int]]:
    parsed = []
    for spec in specs:
        parts = str(spec).split(":")
        if len(parts) != 3:
            raise ValueError(
                "protected-expert must use LAYER:EXPERT:MIN_WIDTH format."
            )
        try:
            parsed.append(tuple(int(part) for part in parts))
        except ValueError as error:
            raise ValueError("protected-expert fields must be integers.") from error
    return parsed


def main() -> int:
    args = parse_args()
    target = float(args.target_pruning_ratio)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target-pruning-ratio must be in [0, 1].")
    teacher = torch.load(args.teacher_cache, map_location="cpu", weights_only=True)
    if teacher.get("split") != "train" or teacher.get("test_metrics_used") is not False:
        raise ValueError("teacher cache must be train-only and independent of test metrics.")
    values_table = teacher[args.value_key]
    layer_ids = sorted(int(layer) for layer in values_table)
    values = torch.stack([values_table[layer].float() for layer in layer_ids])
    if values.ndim != 3:
        raise ValueError("teacher block values must stack to [layers, experts, blocks].")
    maximum_blocks = int(values.numel())
    total_blocks = int(round(maximum_blocks * (1.0 - target)))
    protected_experts = parse_protected_experts(args.protected_expert)
    min_widths = build_protected_min_widths(
        num_layers=int(values.shape[0]),
        num_experts=int(values.shape[1]),
        num_blocks=int(values.shape[2]),
        protected_experts=protected_experts,
    )
    widths = allocate_static_prefix_widths(
        values,
        total_blocks=total_blocks,
        min_blocks_per_expert=args.min_blocks_per_expert,
        min_widths=min_widths,
    )
    channel_payload = torch.load(
        args.channel_cache, map_location="cpu", weights_only=True
    )
    if channel_payload.get("split") != "train":
        raise ValueError("channel cache must be train-only.")
    profile_digest = hashlib.sha256(widths.numpy().tobytes()).hexdigest()
    if args.value_key == "block_values":
        mode = "dynamic_regret"
    else:
        parent_mode = str(teacher.get("parent_mode", "combined"))
        mode = (
            "dynamic_expected_utility"
            if parent_mode == "combined"
            else f"expected_utility_{parent_mode}"
        )
    if protected_experts:
        mode = f"{mode}_protected"
    payload = {
        "schema_version": 1,
        "method": f"static_expert_{mode}",
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": teacher.get("model_path"),
        "dataset": teacher.get("dataset"),
        "calibration_split": teacher.get("split"),
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": int(widths.shape[0]),
        "num_experts": int(widths.shape[1]),
        "num_blocks": int(widths.shape[2] if widths.ndim == 3 else values.shape[2]),
        "channel_block_size": int(channel_payload.get("block_size", 0)),
        "target_pruning_ratio": target,
        "actual_structural_pruning_ratio": 1.0
        - int(widths.sum().item()) / maximum_blocks,
        "total_blocks": int(widths.sum().item()),
        "maximum_blocks": maximum_blocks,
        "min_blocks_per_expert": int(args.min_blocks_per_expert),
        "protected_experts": [
            {"layer": layer, "expert": expert, "min_width": width}
            for layer, expert, width in protected_experts
        ],
        "profile_widths": widths.cpu(),
        "profile_sha256": profile_digest,
        "teacher_target_dynamic_pruning_ratio": teacher.get(
            "target_dynamic_pruning_ratio"
        ),
        "teacher_total_blocks_per_token_per_layer": teacher.get(
            "total_blocks_per_token_per_layer"
        ),
        "regret_value": teacher.get("regret_value"),
        "teacher_value_key": args.value_key,
        "cache_provenance": {
            "channel": {
                "path": str(args.channel_cache.resolve()),
                "sha256": file_sha256(args.channel_cache),
                "score_mode": channel_payload.get("score_mode"),
                "dataset": channel_payload.get("dataset"),
                "split": channel_payload.get("split"),
                "sequence_length": channel_payload.get("sequence_length"),
                "calibration_sequences": channel_payload.get("calibration_sequences"),
                "calibration_tokens": channel_payload.get("calibration_tokens"),
            },
            "dynamic_regret_teacher": {
                "path": str(args.teacher_cache.resolve()),
                "sha256": file_sha256(args.teacher_cache),
                "teacher": teacher.get("teacher"),
                "split": teacher.get("split"),
                "calibration_tokens": teacher.get("calibration_tokens"),
            },
        },
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_profile)
    unique, counts = torch.unique(widths, return_counts=True)
    summary = {key: value for key, value in payload.items() if key != "profile_widths"}
    summary["width_histogram"] = {
        str(int(width)): int(count)
        for width, count in zip(unique.tolist(), counts.tolist())
    }
    summary["output_profile"] = str(args.output_profile.resolve())
    summary_path = args.output_profile.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output_profile.resolve())
    print(summary_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
