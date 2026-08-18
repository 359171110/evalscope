from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DATASET_ORDER = ("arc", "hellaswag", "winogrande", "gsm8k", "math_500", "mmlu")
EXPECTED_SAMPLES = {
    "arc": 600,
    "hellaswag": 1000,
    "winogrande": 400,
    "gsm8k": 128,
    "math_500": 100,
    "mmlu": 570,
}
DENSE_SCORES = {
    "arc": 0.9783,
    "hellaswag": 0.7430,
    "winogrande": 0.7625,
    "gsm8k": 0.9766,
    "math_500": 0.9200,
    "mmlu": 0.8737,
}
DENSE_MACRO = sum(DENSE_SCORES.values()) / len(DENSE_SCORES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Pure-Pseudo block-aligned Quick9 capacity sweep.")
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        metavar="PRUNED_BLOCKS=EXPERIMENT_DIR",
        help="Add one completed block-aligned experiment.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def parse_job(value: str) -> tuple[int, Path]:
    pruned_text, separator, directory_text = value.partition("=")
    if not separator:
        raise ValueError(f"Invalid --job value: {value}")
    pruned_blocks = int(pruned_text)
    if not 1 <= pruned_blocks <= 6:
        raise ValueError("pruned blocks must be in [1, 6].")
    return pruned_blocks, Path(directory_text).expanduser().resolve()


def find_report(method_root: Path, dataset: str) -> Path:
    reports = list((method_root / dataset / "reports").glob("*/*.json"))
    if len(reports) != 1:
        raise ValueError(f"Expected one {dataset} report under {method_root}, found {len(reports)}.")
    return reports[0]


def summarize_experiment(pruned_blocks: int, experiment_dir: Path) -> dict:
    manifest_path = experiment_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    method = manifest["method"]
    method_root = experiment_dir / method
    datasets = []
    for dataset in DATASET_ORDER:
        report = json.loads(find_report(method_root, dataset).read_text(encoding="utf-8"))
        num = int(report["num"])
        score = float(report["score"])
        if num != EXPECTED_SAMPLES[dataset]:
            raise ValueError(f"Unexpected {dataset} sample count in {experiment_dir}: {num}")
        datasets.append(
            {
                "name": dataset,
                "num": num,
                "score": score,
                "dense_score": DENSE_SCORES[dataset],
                "dense_relative_retention": score / DENSE_SCORES[dataset],
            }
        )

    prediction_files = sorted(method_root.glob("*/predictions/**/*.jsonl"))
    prediction_records = 0
    prediction_errors = 0
    for prediction_file in prediction_files:
        with prediction_file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                prediction_records += 1
                record = json.loads(line)
                if record.get("model_output", {}).get("error") is not None:
                    prediction_errors += 1

    macro_average = sum(item["score"] for item in datasets) / len(datasets)
    integrity_passed = (
        len(prediction_files) == 67
        and prediction_records == sum(EXPECTED_SAMPLES.values())
        and prediction_errors == 0
    )
    if not integrity_passed:
        raise ValueError(
            f"Prediction integrity failed for {experiment_dir}: files={len(prediction_files)}, "
            f"records={prediction_records}, errors={prediction_errors}"
        )

    retained_blocks = 12 - pruned_blocks
    return {
        "pruned_blocks": pruned_blocks,
        "retained_blocks": retained_blocks,
        "pruning_ratio": pruned_blocks / 12,
        "retained_channels_per_expert": retained_blocks * 64,
        "experiment_dir": str(experiment_dir),
        "method": method,
        "datasets": datasets,
        "macro_average": macro_average,
        "dense_macro": DENSE_MACRO,
        "dense_relative_macro_retention": macro_average / DENSE_MACRO,
        "integrity": {
            "report_count": len(datasets),
            "prediction_file_count": len(prediction_files),
            "prediction_record_count": prediction_records,
            "prediction_non_null_error_count": prediction_errors,
            "passed": integrity_passed,
        },
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Pure-Pseudo K8/Q4 Block-Aligned Capacity Sweep",
        "",
        f"- Dense Quick9 macro: {summary['dense_macro']:.4f}",
        "- Ranking cache, router neighbors (K=8), Top-q (q=4), seed, prompts, and limits are frozen.",
        "- Retention is the pruned score divided by the corresponding dense score.",
        "",
        "| Pruned blocks | Retained channels | ARC | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | Macro | Macro retention |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in summary["points"]:
        scores = {item["name"]: item["score"] for item in point["datasets"]}
        lines.append(
            f"| {point['pruned_blocks']}/12 | {point['retained_channels_per_expert']} | "
            f"{scores['arc']:.4f} | {scores['hellaswag']:.4f} | {scores['winogrande']:.4f} | "
            f"{scores['gsm8k']:.4f} | {scores['math_500']:.4f} | {scores['mmlu']:.4f} | "
            f"{point['macro_average']:.4f} | {point['dense_relative_macro_retention']:.2%} |"
        )
    lines.extend(["", "All included points passed the 67-file, 2,798-record, zero-error integrity audit.", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    jobs = sorted((parse_job(value) for value in args.job), key=lambda item: item[0])
    if len({pruned_blocks for pruned_blocks, _ in jobs}) != len(jobs):
        raise ValueError("Each pruned-block point may be specified only once.")
    points = [summarize_experiment(pruned_blocks, experiment_dir) for pruned_blocks, experiment_dir in jobs]
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "pure_pseudo",
        "router_neighbors": 8,
        "top_q": 4,
        "channel_block_size": 64,
        "dense_scores": DENSE_SCORES,
        "dense_macro": DENSE_MACRO,
        "points": points,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(args.output_json)
    print(args.output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())