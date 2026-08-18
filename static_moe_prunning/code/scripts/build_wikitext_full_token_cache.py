from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.calibration_data import (
    build_model_cache_identity,
    load_calibration_text_dataset,
    token_tensor_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the repository's full raw WikiText-2 evaluation stream with "
            "model-specific tokenization."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--arrow-file", action="append", type=Path, default=[])
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--expected-windows", type=int, default=114)
    parser.add_argument("--min-text-length", type=int, default=512)
    parser.add_argument(
        "--protocol-name", default="wikitext2_raw_test_full_model_tokenizer_v1"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sequence_length != 2048:
        raise ValueError("formal WikiText protocol requires sequence_length=2048.")
    if args.expected_windows <= 0 or args.min_text_length < 0:
        raise ValueError("expected-windows must be positive and min-text-length non-negative.")
    dataset, source = load_calibration_text_dataset(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="test",
        text_field="text",
        arrow_files=args.arrow_file,
        require_train=False,
    )
    texts = [
        str(value)
        for value in dataset["text"]
        if len(str(value)) >= int(args.min_text_length)
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.model_max_length = 2**31 - 1
    encoded = tokenizer(
        "".join(texts),
        truncation=False,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.detach().to(dtype=torch.long, device="cpu")
    token_count = int(input_ids.shape[1])
    window_count = math.ceil(token_count / int(args.sequence_length))
    if window_count != int(args.expected_windows):
        raise ValueError(
            f"model-tokenized WikiText stream has {window_count} windows, "
            f"expected {args.expected_windows}."
        )
    payload = {
        "schema_version": 1,
        "purpose": "formal_full_wikitext_model_tokenizer_evaluation",
        "protocol_name": args.protocol_name,
        "model_path": args.model_path,
        "model_identity": build_model_cache_identity(args.model_path),
        "dataset": "wikitext-2-raw-v1",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "test",
        "text_field": "text",
        "sequence_length": int(args.sequence_length),
        "evaluation_windows": int(window_count),
        "evaluation_tokens": int(token_count),
        "source": source,
        "token_stream": {
            "tokenization_strategy": "repository_full_wikitext_filtered_rows",
            "model_tokenizer": str(args.model_path),
            "min_text_length": int(args.min_text_length),
            "rows_selected": len(texts),
            "document_separator": "",
            "add_special_tokens": "tokenizer_default",
            "selected_tokens": int(token_count),
        },
        "input_ids": input_ids,
        "input_ids_sha256": token_tensor_sha256(input_ids),
        "attention_mask_semantics": "all_ones_no_padding",
        "frozen_before_evaluation": True,
        "test_metrics_used": False,
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_cache)
    print(args.output_cache.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
