from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.calibration_data import (
    build_model_cache_identity,
    collect_contiguous_text_tokens,
    token_tensor_sha256,
    validate_calibration_token_cache_payload,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one deterministic train-only mixed calibration token cache."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--wikitext-cache", type=Path, required=True)
    parser.add_argument("--gsm8k-train", type=Path, required=True)
    parser.add_argument("--arc-train", action="append", type=Path, default=[])
    parser.add_argument("--mbpp-train", type=Path, required=True)
    parser.add_argument("--code-train", action="append", type=Path, default=[])
    parser.add_argument("--math-train", action="append", type=Path, default=[])
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--wikitext-sequences", type=int, default=64)
    parser.add_argument("--gsm8k-sequences", type=int, default=32)
    parser.add_argument("--arc-sequences", type=int, default=16)
    parser.add_argument("--mbpp-sequences", type=int, default=16)
    parser.add_argument("--math-sequences", type=int, default=0)
    parser.add_argument("--row-batch-size", type=int, default=1024)
    parser.add_argument(
        "--allow-source-repetition",
        action="store_true",
        help="Repeat the deterministic train source text when it lacks enough tokens.",
    )
    parser.add_argument("--protocol-name", required=True)
    return parser.parse_args()


def _load_rows(paths: list[Path]) -> Dataset:
    resolved = [path.expanduser().resolve() for path in paths]
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"calibration source does not exist: {path}")
    suffixes = {path.suffix.lower() for path in resolved}
    if suffixes == {".parquet"}:
        return Dataset.from_parquet([str(path) for path in resolved])
    if suffixes <= {".json", ".jsonl"}:
        records = []
        for path in resolved:
            with path.open("r", encoding="utf-8") as handle:
                records.extend(json.loads(line) for line in handle if line.strip())
        return Dataset.from_list(records)
    raise ValueError("calibration source files must all be parquet or all be JSON/JSONL.")


def _render_gsm8k(record: dict) -> str:
    return f"Question: {record['question']}\nSolution: {record['answer']}"


def _render_arc(record: dict) -> str:
    choices = record["choices"]
    labels = choices["label"]
    texts = choices["text"]
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return f"Question: {record['question']}\nOptions:\n{options}\nAnswer: {record['answerKey']}"


def _render_mbpp(record: dict) -> str:
    prompt = record.get("prompt", record.get("text", ""))
    code = record.get("code", "")
    tests = record.get("test_list", [])
    test_text = "\n".join(str(test) for test in tests)
    return f"Task: {prompt}\nSolution:\n{code}\nTests:\n{test_text}"


def _render_code_instruction(record: dict) -> str:
    instruction = record["instruction"]
    input_text = record.get("input", "")
    task = instruction if not input_text else f"{instruction}\nInput:\n{input_text}"
    return f"Task: {task}\nSolution:\n{record['output']}"


def _render_math(record: dict) -> str:
    return f"Problem: {record['problem']}\nSolution: {record['solution']}"


def _tokenize_source(
    tokenizer,
    dataset: Dataset,
    renderer: Callable[[dict], str],
    *,
    sequences: int,
    sequence_length: int,
    row_batch_size: int,
    allow_source_repetition: bool = False,
) -> tuple[torch.Tensor, dict]:
    if sequences <= 0:
        raise ValueError("source sequence quotas must be positive.")
    rendered_texts = [renderer(record) for record in dataset]
    if allow_source_repetition:
        rendered_texts = rendered_texts * max(1, (sequences * sequence_length) // max(1, len(rendered_texts)))
        rendered_texts = rendered_texts + rendered_texts[:1]
    rendered = Dataset.from_dict({"text": rendered_texts})
    tokens, stream = collect_contiguous_text_tokens(
        tokenizer,
        rendered,
        text_field="text",
        total_tokens=sequences * sequence_length,
        row_batch_size=row_batch_size,
    )
    return tokens.reshape(sequences, sequence_length), stream


def _round_robin_sequences(sources: list[tuple[str, torch.Tensor]]) -> tuple[torch.Tensor, list[str]]:
    mixed = []
    order = []
    maximum = max(int(tokens.shape[0]) for _, tokens in sources)
    for sequence_idx in range(maximum):
        for name, tokens in sources:
            if sequence_idx >= int(tokens.shape[0]):
                continue
            mixed.append(tokens[sequence_idx])
            order.append(name)
    return torch.stack(mixed).reshape(1, -1), order


def main() -> int:
    args = parse_args()
    sequence_length = int(args.sequence_length)
    if sequence_length <= 0:
        raise ValueError("sequence-length must be positive.")

    wikitext_path = args.wikitext_cache.expanduser().resolve()
    wikitext_payload = torch.load(wikitext_path, map_location="cpu", weights_only=True)
    source_sequence_length = int(wikitext_payload.get("sequence_length", -1))
    wikitext_tokens = validate_calibration_token_cache_payload(
        wikitext_payload,
        required_sequence_length=source_sequence_length,
        model_path=args.model_path,
        require_identity=True,
    ).reshape(-1, sequence_length)
    wikitext_sequences = int(args.wikitext_sequences)
    if wikitext_sequences <= 0 or wikitext_sequences > int(wikitext_tokens.shape[0]):
        raise ValueError("wikitext-sequences exceeds the frozen source cache.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    gsm8k_path = args.gsm8k_train.expanduser().resolve()
    arc_paths = [path.expanduser().resolve() for path in args.arc_train]
    mbpp_path = args.mbpp_train.expanduser().resolve()
    code_paths = [path.expanduser().resolve() for path in args.code_train]
    math_paths = [path.expanduser().resolve() for path in args.math_train]
    gsm8k_tokens, gsm8k_stream = _tokenize_source(
        tokenizer,
        _load_rows([gsm8k_path]),
        _render_gsm8k,
        sequences=int(args.gsm8k_sequences),
        sequence_length=sequence_length,
        row_batch_size=int(args.row_batch_size),
        allow_source_repetition=args.allow_source_repetition,
    )
    mbpp_texts = [_render_mbpp(record) for record in _load_rows([mbpp_path])]
    for record in _load_rows(code_paths) if code_paths else []:
        mbpp_texts.append(_render_code_instruction(record))
    mbpp_tokens, mbpp_stream = _tokenize_source(
        tokenizer,
        Dataset.from_dict({"text": mbpp_texts}),
        lambda record: record["text"],
        sequences=int(args.mbpp_sequences),
        sequence_length=sequence_length,
        row_batch_size=int(args.row_batch_size),
        allow_source_repetition=args.allow_source_repetition,
    )
    token_sources = [
        ("wikitext", wikitext_tokens[:wikitext_sequences]),
        ("gsm8k", gsm8k_tokens),
        ("mbpp", mbpp_tokens),
    ]
    component_streams = {
        "wikitext": wikitext_payload.get("token_stream"),
        "gsm8k": gsm8k_stream,
        "mbpp": mbpp_stream,
    }
    quotas = {
        "wikitext": wikitext_sequences,
        "gsm8k": int(args.gsm8k_sequences),
        "mbpp": int(args.mbpp_sequences),
    }
    source_files = {
        "wikitext": [wikitext_path],
        "gsm8k": [gsm8k_path],
        "mbpp": [mbpp_path, *code_paths],
    }
    if int(args.arc_sequences) > 0:
        if not arc_paths:
            raise ValueError("arc-train is required when arc-sequences is positive.")
        arc_tokens, arc_stream = _tokenize_source(
            tokenizer,
            _load_rows(arc_paths),
            _render_arc,
            sequences=int(args.arc_sequences),
            sequence_length=sequence_length,
            row_batch_size=int(args.row_batch_size),
            allow_source_repetition=args.allow_source_repetition,
        )
        token_sources.append(("arc", arc_tokens))
        component_streams["arc"] = arc_stream
        quotas["arc"] = int(args.arc_sequences)
        source_files["arc"] = arc_paths
    if int(args.math_sequences) > 0:
        if not math_paths:
            raise ValueError("math-train is required when math-sequences is positive.")
        math_tokens, math_stream = _tokenize_source(
            tokenizer,
            _load_rows(math_paths),
            _render_math,
            sequences=int(args.math_sequences),
            sequence_length=sequence_length,
            row_batch_size=int(args.row_batch_size),
            allow_source_repetition=args.allow_source_repetition,
        )
        token_sources.append(("math", math_tokens))
        component_streams["math"] = math_stream
        quotas["math"] = int(args.math_sequences)
        source_files["math"] = math_paths
    input_ids, sequence_order = _round_robin_sequences(token_sources)
    dataset_config = "_".join(f"{name}{quota}" for name, quota in quotas.items())
    components = [
        {
            "name": name,
            "split": "train",
            "sequence_quota": quotas[name],
            "files": [
                {
                    "path": str(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
                for path in paths
            ],
        }
        for name, paths in source_files.items()
    ]
    payload = {
        "schema_version": 1,
        "purpose": "shared_moe_pruning_calibration",
        "protocol_name": args.protocol_name,
        "model_path": args.model_path,
        "model_identity": build_model_cache_identity(args.model_path),
        "dataset": "mixed_train",
        "dataset_config": dataset_config,
        "split": "train",
        "text_field": "text",
        "sequence_length": sequence_length,
        "calibration_sequences": int(input_ids.shape[1] // sequence_length),
        "calibration_tokens": int(input_ids.shape[1]),
        "calibration_token_offset": 0,
        "calibration_token_end": int(input_ids.shape[1]),
        "source": {
            "dataset": "mixed_train",
            "config": dataset_config,
            "split": "train",
            "text_field": "text",
            "source_type": "deterministic_sequence_mix",
            "num_rows": None,
            "arrow_files": [],
            "components": components,
            "mixing_policy": "round_robin_sequences_in_source_order",
            "source_order": [name for name, _ in token_sources],
            "sequence_order": sequence_order,
            "allow_source_repetition": bool(args.allow_source_repetition),
        },
        "token_stream": {
            "mixing_policy": "round_robin_sequences_in_source_order",
            "sequence_quotas": quotas,
            "component_streams": component_streams,
            "allow_source_repetition": bool(args.allow_source_repetition),
        },
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