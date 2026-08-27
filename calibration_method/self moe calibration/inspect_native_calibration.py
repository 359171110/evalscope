"""Inspect language, topic, and mechanical degeneration in native calibration caches."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PATTERNS = {
    "code_or_software": r"```|\b(def|class|import|function|const|var|return|print|SELECT|FROM|API|CLI)\b",
    "math_or_science": r"\\frac|\\sqrt|\b(theorem|equation|probability|matrix|physics|chemistry|biology|algorithm)\b|[∑∫√]",
    "instructions_or_lists": r"(^|\n)\s*(\d+[.)]|[-*•])\s+|\b(step|instructions?|how to|guide|tips|overview|objective)\b",
    "dialogue_or_qa": r"\b(user|assistant|system):|(^|\n)\s*[QA]:|\bquestion\b.*\banswer\b",
    "story_or_narrative": r"\b(said|asked|looked|walked|room|door|story|chapter|character|novel)\b",
    "business_policy_or_society": r"\b(company|business|market|customer|policy|government|economic|financial|management|project|environmental)\b",
    "history_or_culture": r"\b(history|historical|century|ancient|culture|religion|war|empire|society)\b",
    "health_or_lifestyle": r"\b(health|medical|patient|diet|exercise|food|recipe|sleep|mental|disease)\b",
    "web_or_structured": r"https?://|www\.|\{\s*[\"']|<html|<div|\[\s*\{",
}


def parse_args() -> argparse.Namespace:
    """Parse inspection arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def _token_metrics(tokens: list[int]) -> dict[str, float]:
    """Calculate robust token-level repetition indicators."""

    if not tokens:
        return {
            "distinct_token_ratio": 0.0,
            "dominant_token_ratio": 1.0,
            "max_run_ratio": 1.0,
            "repeated_4gram_ratio": 1.0,
        }
    counts = Counter(tokens)
    max_run = run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if previous == current else 1
        max_run = max(max_run, run)
    repeated_positions: set[int] = set()
    seen: set[tuple[int, ...]] = set()
    for index in range(max(0, len(tokens) - 3)):
        ngram = tuple(tokens[index:index + 4])
        if ngram in seen:
            repeated_positions.update(range(index, index + 4))
        seen.add(ngram)
    return {
        "distinct_token_ratio": len(counts) / len(tokens),
        "dominant_token_ratio": max(counts.values()) / len(tokens),
        "max_run_ratio": max_run / len(tokens),
        "repeated_4gram_ratio": len(repeated_positions) / len(tokens),
    }


def inspect_cache(cache: Path, model_path: Path, sample_count: int) -> dict[str, Any]:
    """Inspect one cache and return a JSON-serializable health report."""

    from transformers import AutoTokenizer

    payload = torch.load(cache.expanduser().resolve(), map_location="cpu", weights_only=False)
    tokens = payload["input_ids"].reshape(int(payload["calibration_sequences"]), -1)
    tokenizer = AutoTokenizer.from_pretrained(model_path.expanduser().resolve(), trust_remote_code=True)
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in tokens]
    language = Counter()
    categories = Counter()
    metrics = []
    for row, text in zip(tokens, texts):
        metric = _token_metrics(row.tolist())
        metrics.append(metric)
        scripts = {
            "latin_dominant": len(re.findall(r"[A-Za-z]", text)),
            "cjk_dominant": len(re.findall(r"[\u3400-\u9fff]", text)),
            "cyrillic_dominant": len(re.findall(r"[\u0400-\u04ff]", text)),
            "arabic_dominant": len(re.findall(r"[\u0600-\u06ff]", text)),
        }
        language[max(scripts, key=scripts.get)] += 1
        for category, pattern in PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                categories[category] += 1
    health = {
        "cache": str(cache.expanduser().resolve()),
        "protocol_name": payload.get("protocol_name"),
        "shape": list(payload["input_ids"].shape),
        "blocks": int(payload["calibration_sequences"]),
        "block_length": int(payload["sequence_length"]),
        "language_rows": dict(sorted(language.items())),
        "overlapping_category_rows": dict(sorted(categories.items())),
        "quality": {
            "mean_distinct_token_ratio": sum(item["distinct_token_ratio"] for item in metrics) / len(metrics),
            "rows_distinct_below_0_02": sum(item["distinct_token_ratio"] < 0.02 for item in metrics),
            "rows_dominant_token_above_0_50": sum(item["dominant_token_ratio"] > 0.50 for item in metrics),
            "rows_max_run_above_0_25": sum(item["max_run_ratio"] > 0.25 for item in metrics),
            "rows_repeated_4gram_above_0_85": sum(item["repeated_4gram_ratio"] > 0.85 for item in metrics),
        },
        "generation_health": payload.get("generation_health", {}),
        "samples": [" ".join(text.split())[:700] for text in texts[:max(0, sample_count)]],
    }
    return health


def main() -> int:
    """Print a calibration health report."""

    args = parse_args()
    print(json.dumps(inspect_cache(args.cache, args.model_path, args.sample_count), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())