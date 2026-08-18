from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.corpus_ppl import FullWikiTextPerplexity
from src.channel_runtime import channel_table_from_payload
from src.static_expert_pruning import (
    StaticExpertRuntimeStats,
    patch_qwen3_moe_blocks_static_expert,
    profile_widths_by_layer,
    validate_static_profile_payload,
)
from src.corpus_ppl import (
    FrozenTokenCorpusPerplexity,
    frozen_protocol_matches,
    validate_token_cache_payload,
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
        description="Evaluate a frozen physical-expert static-width profile."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-token-cache", type=Path, default=None)
    parser.add_argument(
        "--correction-modes",
        choices=("none", "global", "agreement_global", "local", "hierarchical"),
        nargs="+",
        default=("none",),
    )
    parser.add_argument("--max-correction-ratio", type=float, default=0.20)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--moe-backend",
        choices=("torch", "triton", "torch_index_add"),
        default="torch_index_add",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sequence_length != 2048:
        raise ValueError("static expert PPL protocol requires sequence_length=2048.")
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("max-windows must be positive when provided.")
    if args.max_correction_ratio < 0:
        raise ValueError("max-correction-ratio must be non-negative.")

    profile_payload = torch.load(args.profile, map_location="cpu", weights_only=True)
    profile_widths = validate_static_profile_payload(profile_payload)
    if profile_payload.get("model_path") != args.model_path:
        raise ValueError("profile model_path does not match the evaluation model.")

    channel_payload = torch.load(
        args.channel_cache, map_location="cpu", weights_only=True
    )
    channel_digest = file_sha256(args.channel_cache)
    expected_digest = (
        profile_payload.get("cache_provenance", {})
        .get("channel", {})
        .get("sha256")
    )
    if expected_digest != channel_digest:
        raise ValueError("channel cache SHA256 does not match profile provenance.")
    if channel_payload.get("split") != "train":
        raise ValueError("channel cache must come from the train split.")
    if int(channel_payload.get("sequence_length", -1)) != 2048:
        raise ValueError("channel cache must use sequence_length=2048.")
    channel_table = channel_table_from_payload(channel_payload["table"])
    layer_ids = [int(layer_id) for layer_id in profile_payload["layer_ids"]]
    width_table = profile_widths_by_layer(profile_widths, layer_ids=layer_ids)
    retained_experts_by_layer = None
    retained_expert_mask = profile_payload.get("retained_expert_mask")
    if isinstance(retained_expert_mask, torch.Tensor):
        retained_experts_by_layer = {
            layer_id: retained_expert_mask[row].detach().cpu().to(torch.bool)
            for row, layer_id in enumerate(layer_ids)
        }
    if set(width_table) != set(channel_table):
        raise ValueError("profile and channel cache layer sets do not match.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_payload = None
    evaluation_cache_digest = None
    model, tokenizer = load_supported_moe(args.model_path)
    if args.evaluation_token_cache is None:
        evaluator = FullWikiTextPerplexity(model, tokenizer)
        evaluation_dataset = "wikitext-2-raw-v1"
        evaluation_config = "wikitext-2-raw-v1"
        evaluation_split = "test"
        protocol_name = "wikitext2_raw_test_full_v1"
        expected_windows = 114
        expected_tokens = 233368
        output_name = "static_expert_wikitext_ppl.json"
        partial_name = "static_expert_wikitext_ppl.partial.json"
    else:
        evaluation_payload = torch.load(
            args.evaluation_token_cache, map_location="cpu", weights_only=True
        )
        token_ids = validate_token_cache_payload(
            evaluation_payload,
            required_sequence_length=args.sequence_length,
            model_path=args.model_path,
            require_identity=True,
        )
        evaluator = FrozenTokenCorpusPerplexity(model, token_ids)
        evaluation_dataset = evaluation_payload.get("dataset")
        evaluation_config = evaluation_payload.get("dataset_config")
        evaluation_split = evaluation_payload.get("split")
        protocol_name = evaluation_payload.get("protocol_name")
        expected_windows = int(evaluation_payload["evaluation_windows"])
        expected_tokens = int(evaluation_payload["evaluation_tokens"])
        evaluation_cache_digest = file_sha256(args.evaluation_token_cache)
        output_name = "static_expert_corpus_ppl.json"
        partial_name = "static_expert_corpus_ppl.partial.json"
    rows = []
    partial_path = args.output_dir / partial_name
    for correction_mode in args.correction_modes:
        stats = StaticExpertRuntimeStats(
            profile_widths=profile_widths,
            num_blocks=int(profile_payload["num_blocks"]),
        )
        with patch_qwen3_moe_blocks_static_expert(
            model,
            width_table,
            channel_table,
            retained_experts_by_layer=retained_experts_by_layer,
            correction_mode=correction_mode,
            max_correction_ratio=args.max_correction_ratio,
            runtime_stats=stats,
            moe_backend=args.moe_backend,
        ):
            metrics = evaluator.calculate_corpus_ppl(
                n_ctx=args.sequence_length,
                max_windows=args.max_windows,
            )
        if evaluation_payload is None:
            formal_protocol = (
                args.max_windows is None
                and int(metrics["windows"]) == expected_windows
                and int(metrics["tokens"]) == expected_tokens
            )
        else:
            formal_protocol = frozen_protocol_matches(
                metrics, evaluation_payload, max_windows=args.max_windows
            )
        widths_unique, widths_counts = torch.unique(profile_widths, return_counts=True)
        row = {
            "method": profile_payload["method"],
            "mode": profile_payload["mode"],
            "correction_mode": correction_mode,
            "max_correction_ratio": args.max_correction_ratio,
            "ppl": float(metrics["ppl"]),
            "windows": int(metrics["windows"]),
            "tokens": int(metrics["tokens"]),
            "sequence_length": args.sequence_length,
            "protocol_name": protocol_name,
            "formal_protocol": formal_protocol,
            "standard_protocol": args.evaluation_token_cache is None
            and args.max_windows is None,
            "dataset": evaluation_dataset,
            "dataset_config": evaluation_config,
            "split": evaluation_split,
            "model_path": args.model_path,
            "profile_path": str(args.profile.resolve()),
            "profile_file_sha256": file_sha256(args.profile),
            "profile_sha256": profile_payload.get("profile_sha256"),
            "profile_frozen_before_evaluation": profile_payload.get(
                "calibration_frozen_before_evaluation"
            ),
            "test_metrics_used_for_profile": profile_payload.get(
                "test_metrics_used_for_profile"
            ),
            "target_pruning_ratio": profile_payload.get("target_pruning_ratio"),
            "structural_pruning_ratio": stats.structural_pruning_ratio(),
            "routed_compute_pruning_ratio": stats.routed_pruning_ratio(),
            "total_profile_blocks": int(profile_widths.sum().item()),
            "maximum_profile_blocks": int(profile_payload["maximum_blocks"]),
            "profile_width_histogram": {
                str(int(width)): int(count)
                for width, count in zip(widths_unique.tolist(), widths_counts.tolist())
            },
            "routed_width_histogram": {
                str(width): count
                for width, count in stats.aggregate_width_histogram().items()
            },
            "routed_pruning_by_layer": {
                str(layer): ratio
                for layer, ratio in stats.routed_pruning_by_layer().items()
            },
            "channel_block_size": profile_payload.get("channel_block_size"),
            "channel_score_mode": channel_payload.get("score_mode"),
            "cache_provenance": profile_payload.get("cache_provenance"),
            "moe_backend": args.moe_backend,
            "evaluation_token_cache": None
            if args.evaluation_token_cache is None
            else str(args.evaluation_token_cache.resolve()),
            "evaluation_token_cache_sha256": evaluation_cache_digest,
            "evaluation_source": None
            if evaluation_payload is None
            else evaluation_payload.get("source"),
        }
        rows.append(row)
        partial_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    final_path = args.output_dir / output_name
    final_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    partial_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(final_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
