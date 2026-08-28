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

CLARIFICATION_PATTERN = re.compile(
    r"\b(please clarify|could you clarify|message .* incomplete|provide more "
    r"(?:details|information)|not sure what you mean|what would you like|"
    r"please provide more context|could you provide more details)\b|"
    r"请补充|请澄清|信息不完整|无法确定您指的是|请提供更多",
    re.IGNORECASE,
)
REFUSAL_PATTERN = re.compile(
    r"\b(i can(?:not|'t)|i am unable|i'm unable|cannot|can't|unable to|"
    r"not able to|i won't|i will not)\b.{0,80}\b(help|assist|provide|answer|do)\b|"
    r"我无法|不能协助|无法提供|不能帮助",
    re.IGNORECASE | re.DOTALL,
)
FIXED_FORMAT_PATTERN = re.compile(
    r"(^|\n)\s*(#{1,6}\s+|(?:answer|solution|explanation|summary|output|result)\s*:)|"
    r"(^|\n)\s*```[\w+-]*\s*$|(^|\n)\s*\|[^\n]+\|\s*$",
    re.IGNORECASE | re.MULTILINE,
)


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


def _aggregate_quality(metrics: list[dict[str, float]]) -> dict[str, float | int]:
    """Aggregate repetition diagnostics for one inspection granularity."""

    if not metrics:
        return {"count": 0}
    return {
        "count": len(metrics),
        "mean_distinct_token_ratio": sum(item["distinct_token_ratio"] for item in metrics) / len(metrics),
        "rows_distinct_below_0_02": sum(item["distinct_token_ratio"] < 0.02 for item in metrics),
        "rows_dominant_token_above_0_50": sum(item["dominant_token_ratio"] > 0.50 for item in metrics),
        "rows_max_run_above_0_25": sum(item["max_run_ratio"] > 0.25 for item in metrics),
        "rows_repeated_4gram_above_0_85": sum(item["repeated_4gram_ratio"] > 0.85 for item in metrics),
    }


def _semantic_modes(texts: list[str]) -> dict[str, Any]:
    """Summarize semantic modes without treating any mode as invalid."""

    counts = Counter()
    forms = Counter()
    for text in texts:
        compact = " ".join(text.split())
        for category, pattern in PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                counts[category] += 1
        if CLARIFICATION_PATTERN.search(text):
            counts["clarification"] += 1
            forms[compact.casefold()] += 1
        if REFUSAL_PATTERN.search(text):
            counts["refusal"] += 1
        if FIXED_FORMAT_PATTERN.search(text):
            counts["fixed_format"] += 1
    return {
        "count": len(texts),
        "mode_rows": dict(sorted(counts.items())),
        "clarification_rate": counts["clarification"] / max(len(texts), 1),
        "refusal_rate": counts["refusal"] / max(len(texts), 1),
        "fixed_format_rate": counts["fixed_format"] / max(len(texts), 1),
        "distinct_clarification_forms": len(forms),
        "top_clarification_forms": forms.most_common(10),
    }


def _reconstruct_episodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct episode and turn token ranges from packed-cache provenance."""

    records = payload.get("token_stream", {}).get("episode_boundaries", [])
    scaffold = payload.get("generation_health", {}).get("native_scaffold", {})
    prefix_tokens = int(scaffold.get("user_prefix_tokens", 0))
    bridge_tokens = int(scaffold.get("user_bridge_tokens", 0))
    flat = payload["input_ids"].reshape(-1).tolist()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["episode_id"]), []).append(record)
    episodes = []
    for episode_id, segments in sorted(grouped.items()):
        ordered = sorted(segments, key=lambda item: int(item.get("source_start", 0)))
        if not ordered or not bool(ordered[-1].get("complete", False)):
            continue
        episode_tokens = []
        for segment in ordered:
            episode_tokens.extend(flat[int(segment["start"]):int(segment["end"])])
        metadata = ordered[0]
        user_length = int(metadata.get("user_tokens", 0))
        assistant_length = int(metadata.get("assistant_tokens", 0))
        user_start = prefix_tokens
        assistant_start = prefix_tokens + user_length + bridge_tokens
        episodes.append({
            "episode_id": episode_id,
            "tokens": episode_tokens,
            "user_tokens": episode_tokens[user_start:user_start + user_length],
            "assistant_tokens": episode_tokens[assistant_start:assistant_start + assistant_length],
            "user_terminated": bool(metadata.get("user_terminated", False)),
            "assistant_terminated": bool(metadata.get("assistant_terminated", False)),
        })
    return episodes


def inspect_cache(cache: Path, model_path: Path, sample_count: int) -> dict[str, Any]:
    """Inspect one cache and return a JSON-serializable health report."""

    from transformers import AutoTokenizer

    payload = torch.load(cache.expanduser().resolve(), map_location="cpu", weights_only=False)
    tokens = payload["input_ids"].reshape(int(payload["calibration_sequences"]), -1)
    tokenizer = AutoTokenizer.from_pretrained(model_path.expanduser().resolve(), trust_remote_code=True)
    texts = [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in tokens]
    episodes = _reconstruct_episodes(payload)
    user_texts = [tokenizer.decode(item["user_tokens"], skip_special_tokens=True) for item in episodes]
    assistant_texts = [tokenizer.decode(item["assistant_tokens"], skip_special_tokens=True) for item in episodes]
    language = Counter()
    categories = Counter()
    block_metrics = []
    for row, text in zip(tokens, texts):
        metric = _token_metrics(row.tolist())
        block_metrics.append(metric)
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
    semantic_texts = assistant_texts if assistant_texts else texts
    episode_texts = [tokenizer.decode(item["tokens"], skip_special_tokens=True) for item in episodes]
    health = {
        "cache": str(cache.expanduser().resolve()),
        "protocol_name": payload.get("protocol_name"),
        "shape": list(payload["input_ids"].shape),
        "blocks": int(payload["calibration_sequences"]),
        "block_length": int(payload["sequence_length"]),
        "language_rows": dict(sorted(language.items())),
        "overlapping_category_rows": dict(sorted(categories.items())),
        "semantic_modes": {
            "block": _semantic_modes(texts),
            "episode": _semantic_modes(episode_texts),
            "user": _semantic_modes(user_texts),
            "assistant": _semantic_modes(assistant_texts),
        },
        "quality": {
            "block": _aggregate_quality(block_metrics),
            "episode": _aggregate_quality([_token_metrics(item["tokens"]) for item in episodes]),
            "user": _aggregate_quality([_token_metrics(item["user_tokens"]) for item in episodes]),
            "assistant": _aggregate_quality([_token_metrics(item["assistant_tokens"]) for item in episodes]),
        },
        "generation_health": payload.get("generation_health", {}),
        "samples": [" ".join(text.split())[:700] for text in texts[:max(0, sample_count)]],
        "user_samples": [" ".join(text.split())[:700] for text in user_texts[:max(0, sample_count)]],
        "assistant_samples": [" ".join(text.split())[:700] for text in assistant_texts[:max(0, sample_count)]],
    }
    return health


def main() -> int:
    """Print a calibration health report."""

    args = parse_args()
    print(json.dumps(inspect_cache(args.cache, args.model_path, args.sample_count), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())