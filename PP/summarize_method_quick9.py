from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_ORDER = ("arc", "hellaswag", "winogrande", "gsm8k", "math_500", "mmlu")
EXPECTED_COUNTS = {
    "arc": 600,
    "hellaswag": 1000,
    "winogrande": 400,
    "gsm8k": 128,
    "math_500": 100,
    "mmlu": 570,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize complete Quick9 experiment directories.")
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    return parser.parse_args()


def _report_dataset(path: Path) -> str | None:
    lowered = path.as_posix().lower()
    for dataset in DATASET_ORDER:
        if f"/{dataset}/" in lowered or f"/{dataset.replace('_', '-')}/" in lowered:
            return dataset
    return None


def _aggregate_score_and_count(payload: dict) -> tuple[float | None, int | None]:
    score = payload.get("score")
    metrics = payload.get("metrics")
    if not isinstance(score, (int, float)) or not isinstance(metrics, list) or not metrics:
        return None, None
    aggregate = metrics[0]
    if not isinstance(aggregate, dict) or not isinstance(aggregate.get("num"), int):
        return None, None
    return float(score), int(aggregate["num"])


def summarize_experiment(experiment_dir: Path) -> dict[str, object]:
    reports = sorted(experiment_dir.glob("**/reports/**/*.json"))
    candidates: dict[str, list[tuple[Path, dict]]] = {dataset: [] for dataset in DATASET_ORDER}
    for path in reports:
        dataset = _report_dataset(path)
        if dataset is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            candidates[dataset].append((path, payload))
    scores = {}
    counts = {}
    sources = {}
    for dataset in DATASET_ORDER:
        expected_count = EXPECTED_COUNTS[dataset]
        matches = []
        for path, payload in candidates[dataset]:
            score, count = _aggregate_score_and_count(payload)
            if score is not None and count == expected_count:
                matches.append((path, score, count))
        if len(matches) != 1:
            raise ValueError(
                f"{experiment_dir}: expected one {dataset} aggregate with {expected_count} samples, found {len(matches)}"
            )
        path, score, count = matches[0]
        scores[dataset] = score
        counts[dataset] = count
        sources[dataset] = str(path)
    return {
        "experiment_dir": str(experiment_dir.resolve()),
        "scores": scores,
        "counts": counts,
        "macro": sum(scores.values()) / len(DATASET_ORDER),
        "sources": sources,
    }


def main() -> int:
    summaries = [summarize_experiment(path.expanduser().resolve()) for path in parse_args().experiment_dirs]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())