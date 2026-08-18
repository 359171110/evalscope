from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.calibration_data import (
    build_model_cache_identity,
    collect_contiguous_text_tokens,
    load_calibration_text_dataset,
    token_tensor_sha256,
)
from src.crossfit_parent_selection import validate_disjoint_token_ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a train-only token fold for cross-fitted profile selection."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--arrow-file", action="append", type=Path, default=[])
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--selection-windows", type=int, default=16)
    parser.add_argument("--token-offset", type=int, required=True)
    parser.add_argument("--calibration-token-start", type=int, default=0)
    parser.add_argument("--calibration-token-end", type=int, required=True)
    parser.add_argument("--row-batch-size", type=int, default=1024)
    parser.add_argument("--protocol-name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.split != "train":
        raise ValueError("cross-fitted selection cache must use the train split.")
    if args.sequence_length != 2048 or args.selection_windows <= 0:
        raise ValueError("selection cache requires positive 2048-token windows.")
    total_tokens = int(args.sequence_length) * int(args.selection_windows)
    selection_range = (int(args.token_offset), int(args.token_offset) + total_tokens)
    validate_disjoint_token_ranges(
        calibration_range=(
            int(args.calibration_token_start),
            int(args.calibration_token_end),
        ),
        selection_ranges=[selection_range],
    )
    dataset, source = load_calibration_text_dataset(
        dataset_name=args.dataset,
        dataset_config=args.config,
        split=args.split,
        text_field=args.text_field,
        arrow_files=args.arrow_file,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    input_ids, stream = collect_contiguous_text_tokens(
        tokenizer,
        dataset,
        text_field=args.text_field,
        total_tokens=total_tokens,
        token_offset=int(args.token_offset),
        row_batch_size=args.row_batch_size,
    )
    payload = {
        "schema_version": 1,
        "purpose": "train_only_crossfit_profile_selection",
        "protocol_name": args.protocol_name,
        "model_path": args.model_path,
        "model_identity": build_model_cache_identity(args.model_path),
        "dataset": args.dataset,
        "dataset_config": args.config,
        "split": "train",
        "text_field": args.text_field,
        "sequence_length": int(args.sequence_length),
        "evaluation_windows": int(args.selection_windows),
        "evaluation_tokens": int(input_ids.shape[1]),
        "evaluation_token_offset": selection_range[0],
        "evaluation_token_end": selection_range[1],
        "calibration_token_start": int(args.calibration_token_start),
        "calibration_token_end": int(args.calibration_token_end),
        "source": source,
        "token_stream": stream,
        "input_ids": input_ids.cpu(),
        "input_ids_sha256": token_tensor_sha256(input_ids),
        "attention_mask_semantics": "all_ones_no_padding",
        "frozen_before_selection": True,
        "frozen_before_evaluation": True,
        "test_metrics_used": False,
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_cache)
    print(args.output_cache.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
