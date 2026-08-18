from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

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
    parser = argparse.ArgumentParser(description="Evaluate native dense MoE corpus PPL.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--evaluation-token-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = torch.load(
        args.evaluation_token_cache, map_location="cpu", weights_only=True
    )
    tokens = validate_token_cache_payload(
        payload,
        required_sequence_length=args.sequence_length,
        model_path=args.model_path,
        require_identity=True,
    )
    model, _ = load_supported_moe(args.model_path)
    evaluator = FrozenTokenCorpusPerplexity(model, tokens)
    metrics = evaluator.calculate_corpus_ppl(
        n_ctx=args.sequence_length, max_windows=args.max_windows
    )
    formal = frozen_protocol_matches(metrics, payload, max_windows=args.max_windows)
    row = {
        "method": "native_dense_moe",
        "mode": "dense",
        "ppl": float(metrics["ppl"]),
        "windows": int(metrics["windows"]),
        "tokens": int(metrics["tokens"]),
        "sequence_length": int(args.sequence_length),
        "dataset": payload.get("dataset"),
        "dataset_config": payload.get("dataset_config"),
        "split": payload.get("split"),
        "protocol_name": payload.get("protocol_name"),
        "formal_protocol": formal,
        "standard_protocol": False,
        "model_path": args.model_path,
        "evaluation_token_cache": str(args.evaluation_token_cache.resolve()),
        "evaluation_token_cache_sha256": file_sha256(args.evaluation_token_cache),
        "evaluation_source": payload.get("source"),
        "native_model_forward": True,
        "structural_pruning_ratio": 0.0,
        "routed_compute_pruning_ratio": 0.0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "dense_corpus_ppl.json"
    output.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
