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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze one train-only token artifact shared by MoE pruning methods."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--arrow-file", action="append", type=Path, default=[])
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--calibration-sequences", type=int, default=128)
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--row-batch-size", type=int, default=1024)
    parser.add_argument("--protocol-name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.split != "train":
        raise ValueError("shared MoE calibration must use the train split.")
    if args.sequence_length <= 0 or args.calibration_sequences <= 0:
        raise ValueError("sequence length and calibration sequences must be positive.")
    dataset, source = load_calibration_text_dataset(
        dataset_name=args.dataset,
        dataset_config=args.config,
        split=args.split,
        text_field=args.text_field,
        arrow_files=args.arrow_file,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    total_tokens = int(args.sequence_length) * int(args.calibration_sequences)
    input_ids, stream = collect_contiguous_text_tokens(
        tokenizer,
        dataset,
        text_field=args.text_field,
        total_tokens=total_tokens,
        token_offset=int(args.token_offset),
        row_batch_size=int(args.row_batch_size),
    )
    payload = {
        "schema_version": 1,
        "purpose": "shared_moe_pruning_calibration",
        "protocol_name": args.protocol_name,
        "model_path": args.model_path,
        "model_identity": build_model_cache_identity(args.model_path),
        "dataset": args.dataset,
        "dataset_config": args.config,
        "split": "train",
        "text_field": args.text_field,
        "sequence_length": int(args.sequence_length),
        "calibration_sequences": int(args.calibration_sequences),
        "calibration_tokens": int(input_ids.shape[1]),
        "calibration_token_offset": int(args.token_offset),
        "calibration_token_end": int(args.token_offset) + int(input_ids.shape[1]),
        "source": source,
        "token_stream": stream,
        "input_ids": input_ids.cpu(),
        "input_ids_sha256": token_tensor_sha256(input_ids),
        "attention_mask_semantics": "all_ones_no_padding",
        "frozen_before_profile": True,
        "test_metrics_used": False,
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_cache)
    print(args.output_cache.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
