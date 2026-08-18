from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset

from src.channel_runtime import channel_table_from_payload
from src.multiple_choice_eval import (
    aggregate_truthfulqa_metrics,
    batched_conditional_loglikelihood,
)
from src.static_expert_pruning import (
    StaticExpertRuntimeStats,
    patch_qwen3_moe_blocks_static_expert,
    profile_widths_by_layer,
    validate_static_profile_payload,
)
from src.model_loading import load_supported_moe


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense or frozen static-expert Qwen3 on TruthfulQA MC."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--channel-cache", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--moe-backend",
        choices=("torch", "triton", "torch_index_add"),
        default="torch_index_add",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.profile is None) != (args.channel_cache is None):
        raise ValueError("profile and channel-cache must be supplied together.")
    if args.sequence_length != 2048:
        raise ValueError("TruthfulQA protocol requires sequence_length=2048.")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("max-examples must be positive when provided.")

    dataset = load_dataset(
        "truthfulqa/truthful_qa", "multiple_choice", split="validation"
    )
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    requests: list[tuple[str, str]] = []
    spans: list[dict[str, object]] = []
    for example in dataset:
        prompt = f"Q: {example['question']}\nA:"
        row_spans: dict[str, object] = {}
        for key in ("mc1_targets", "mc2_targets"):
            targets = example[key]
            begin = len(requests)
            requests.extend((prompt, f" {choice}") for choice in targets["choices"])
            row_spans[key] = {
                "begin": begin,
                "end": len(requests),
                "labels": [int(label) for label in targets["labels"]],
            }
        spans.append(row_spans)

    model, tokenizer = load_supported_moe(args.model_path)
    profile_payload = None
    profile_widths = None
    runtime_stats = None
    patch_context = nullcontext(model)
    channel_payload = None
    if args.profile is not None and args.channel_cache is not None:
        profile_payload = torch.load(args.profile, map_location="cpu", weights_only=True)
        if profile_payload.get("model_path") != args.model_path:
            raise ValueError("profile model_path does not match evaluation model.")
        profile_widths = validate_static_profile_payload(profile_payload)
        channel_payload = torch.load(
            args.channel_cache, map_location="cpu", weights_only=True
        )
        expected_sha = (
            profile_payload.get("cache_provenance", {})
            .get("channel", {})
            .get("sha256")
        )
        if expected_sha != file_sha256(args.channel_cache):
            raise ValueError("channel cache SHA256 does not match profile provenance.")
        channel_table = channel_table_from_payload(channel_payload["table"])
        layer_ids = [int(layer) for layer in profile_payload["layer_ids"]]
        width_table = profile_widths_by_layer(profile_widths, layer_ids=layer_ids)
        runtime_stats = StaticExpertRuntimeStats(
            profile_widths=profile_widths,
            num_blocks=int(profile_payload["num_blocks"]),
        )
        patch_context = patch_qwen3_moe_blocks_static_expert(
            model,
            width_table,
            channel_table,
            correction_mode="none",
            runtime_stats=runtime_stats,
            moe_backend=args.moe_backend,
        )

    with patch_context:
        scores = batched_conditional_loglikelihood(
            model,
            tokenizer,
            requests,
            batch_size=args.batch_size,
            max_length=args.sequence_length,
        )

    metric_rows = []
    for row_spans in spans:
        row = {}
        for prefix, key in (("mc1", "mc1_targets"), ("mc2", "mc2_targets")):
            span = row_spans[key]
            row[f"{prefix}_scores"] = scores[span["begin"] : span["end"]]
            row[f"{prefix}_labels"] = span["labels"]
        metric_rows.append(row)
    metrics = aggregate_truthfulqa_metrics(metric_rows)

    mode = "dense" if profile_payload is None else str(profile_payload["mode"])
    output = {
        "protocol_name": "truthfulqa_multiple_choice_validation_v1",
        "formal_protocol": args.max_examples is None and len(dataset) == 817,
        "dataset": "truthfulqa/truthful_qa",
        "dataset_config": "multiple_choice",
        "split": "validation",
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "examples": len(dataset),
        "prompt_template": "Q: {question}\\nA:",
        "choice_prefix": "single_space",
        "metric_definition": {
            "mc1_accuracy": "argmax answer log-likelihood has binary truth label 1",
            "mc2_true_probability": "softmax-normalized likelihood mass assigned to all true answers",
        },
        "model_path": args.model_path,
        "mode": mode,
        "metrics": metrics,
        "profile_path": None if args.profile is None else str(args.profile.resolve()),
        "profile_file_sha256": None
        if args.profile is None
        else file_sha256(args.profile),
        "profile_sha256": None
        if profile_payload is None
        else profile_payload.get("profile_sha256"),
        "profile_frozen_before_evaluation": None
        if profile_payload is None
        else profile_payload.get("calibration_frozen_before_evaluation"),
        "test_metrics_used_for_profile": None
        if profile_payload is None
        else profile_payload.get("test_metrics_used_for_profile"),
        "target_pruning_ratio": None
        if profile_payload is None
        else profile_payload.get("target_pruning_ratio"),
        "structural_pruning_ratio": None
        if runtime_stats is None
        else runtime_stats.structural_pruning_ratio(),
        "routed_compute_pruning_ratio": None
        if runtime_stats is None
        else runtime_stats.routed_pruning_ratio(),
        "channel_cache_path": None
        if args.channel_cache is None
        else str(args.channel_cache.resolve()),
        "channel_cache_sha256": None
        if args.channel_cache is None
        else file_sha256(args.channel_cache),
        "channel_score_mode": None
        if channel_payload is None
        else channel_payload.get("score_mode"),
        "moe_backend": args.moe_backend,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
