from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.corpus_ppl import FullWikiTextPerplexity
from src.model_loading import load_supported_moe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate native dense MoE on the formal WikiText-2 protocol."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--device-map",
        choices=("single", "auto"),
        default="single",
        help="Use one visible GPU or Accelerate auto-sharding across visible GPUs.",
    )
    parser.add_argument("--method", default="native_dense_moe")
    parser.add_argument("--mode", default="dense")
    parser.add_argument("--structural-pruning-ratio", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sequence_length != 2048:
        raise ValueError("formal WikiText protocol requires sequence_length=2048.")
    device_map = "auto" if args.device_map == "auto" else None
    model, tokenizer = load_supported_moe(args.model_path, device_map=device_map)
    metrics = FullWikiTextPerplexity(model, tokenizer).calculate_corpus_ppl(
        n_ctx=args.sequence_length
    )
    row = {
        "method": args.method,
        "mode": args.mode,
        "ppl": float(metrics["ppl"]),
        "windows": int(metrics["windows"]),
        "tokens": int(metrics["tokens"]),
        "sequence_length": int(args.sequence_length),
        "protocol_name": "wikitext2_raw_test_full_v1",
        "formal_protocol": int(metrics["windows"]) == 114
        and int(metrics["tokens"]) == 233368,
        "standard_protocol": True,
        "dataset": "wikitext-2-raw-v1",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "test",
        "model_path": args.model_path,
        "native_model_forward": True,
        "structural_pruning_ratio": args.structural_pruning_ratio,
        "routed_compute_pruning_ratio": 0.0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "dense_wikitext_ppl.json"
    output.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
