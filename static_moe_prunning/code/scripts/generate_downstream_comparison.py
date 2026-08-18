from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "results"
    / "qwen3_50pct_downstream_full_20260801"
)
VARIANT_ORDER = (
    "dense",
    "enp",
    "tenp",
    "aimer",
    "pure_pseudo",
    "wick_kernel",
    "wick_kernel_merge",
    "wick_pseudo_protect",
    "wick_pseudo_protect_merge",
    "official_reap",
    "route_tail_global",
    "route_tail_per_layer",
    "tail_risk_global",
    "tail_risk_per_layer",
)
DATASET_ORDER = (
    "arc",
    "hellaswag",
    "mmlu",
    "mmlu_pro",
    "winogrande",
    "boolq",
    "openbookqa",
    "rte",
    "gsm8k",
    "math_500",
    "humaneval_plus",
    "mbpp_plus",
    "live_code_bench",
    "ifeval",
)
DATASET_LABELS = {
    "arc": "ARC",
    "hellaswag": "HellaSwag",
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "math_500": "MATH-500",
    "winogrande": "WinoGrande",
    "boolq": "BoolQ",
    "openbookqa": "OpenBookQA",
    "rte": "GLUE RTE",
    "humaneval_plus": "HumanEval+",
    "mbpp_plus": "MBPP+",
    "live_code_bench": "LiveCodeBench",
    "ifeval": "IFEval",
    "mmlu_pro": "MMLU-Pro",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a manifest-driven Markdown comparison from EvalScope reports."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional frozen experiment manifest defining expected methods, datasets, and settings.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=f"Experiment result root. Defaults to the manifest value or {DEFAULT_RESULTS_ROOT}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown output path. The table is always printed to stdout.",
    )
    return parser.parse_args()


def _variant_sort_key(variant: str) -> tuple[int, str]:
    try:
        return VARIANT_ORDER.index(variant), variant
    except ValueError:
        return len(VARIANT_ORDER), variant


def _dataset_sort_key(dataset: str) -> tuple[int, str]:
    try:
        return DATASET_ORDER.index(dataset), dataset
    except ValueError:
        return len(DATASET_ORDER), dataset


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _format_number(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{number:.4f}"


def _extract_metrics(payload: dict[str, Any]) -> list[tuple[str, float | int]]:
    metrics: list[tuple[str, float | int]] = []
    raw_metrics = payload.get("metrics")
    if isinstance(raw_metrics, list):
        for metric in raw_metrics:
            if not isinstance(metric, dict):
                continue
            name = metric.get("name")
            score = _number(metric.get("score"))
            if isinstance(name, str) and score is not None:
                metrics.append((name, score))
    if metrics:
        return metrics

    score = _number(payload.get("score"))
    return [("score", score)] if score is not None else []


def _extract_report(variant: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    dataset = payload.get("dataset_name")
    if not isinstance(dataset, str) or not dataset:
        dataset = path.stem
    model_name = payload.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        model_name = payload.get("name", variant)

    perf_metrics = payload.get("perf_metrics")
    summary = perf_metrics.get("summary", {}) if isinstance(perf_metrics, dict) else {}
    latency = summary.get("latency", {}).get("mean") if isinstance(summary, dict) else None
    throughput = summary.get("throughput", {}).get("avg_output_tps") if isinstance(summary, dict) else None
    samples = payload.get("num")
    if samples is None and isinstance(summary, dict):
        samples = summary.get("n_samples")

    return {
        "variant": variant,
        "model_name": model_name,
        "dataset": dataset,
        "metrics": _extract_metrics(payload),
        "samples": samples,
        "latency": latency,
        "throughput": throughput,
        "path": str(path),
    }


def discover_reports(results_root: Path) -> list[dict[str, Any]]:
    if not results_root.is_dir():
        return []

    reports: list[dict[str, Any]] = []
    for variant_dir in sorted(
        (path for path in results_root.iterdir() if path.is_dir()),
        key=lambda path: _variant_sort_key(path.name),
    ):
        reports_dir = variant_dir / "reports"
        if not reports_dir.is_dir():
            continue
        for path in sorted(reports_dir.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Warning: skipped unreadable report {path}: {exc}", file=sys.stderr)
                continue
            if not isinstance(payload, dict):
                print(f"Warning: skipped non-object report {path}", file=sys.stderr)
                continue
            reports.append(_extract_report(variant_dir.name, path, payload))
    return reports


def _format_score_cell(report: dict[str, Any] | None) -> str:
    if report is None:
        return "-"
    metrics = report["metrics"]
    if not metrics:
        return "-"
    if len(metrics) == 1:
        return _format_number(metrics[0][1])
    return "<br>".join(f"{name}={_format_number(score)}" for name, score in metrics)


def _format_samples(value: Any) -> str:
    number = _number(value)
    if isinstance(number, int):
        return str(number)
    if isinstance(number, float) and number.is_integer():
        return str(int(number))
    return _format_number(value)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join('---' for _ in headers)} |"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return lines


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment manifest must contain a JSON object: {path}")
    return payload


def _manifest_methods(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if manifest is None:
        return []
    methods = manifest.get("methods", [])
    if not isinstance(methods, list) or not all(isinstance(method, dict) for method in methods):
        raise ValueError("manifest methods must be a list of objects.")
    return methods


def _manifest_datasets(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if manifest is None:
        return []
    evaluation = manifest.get("evaluation", {})
    datasets = evaluation.get("active_datasets", []) if isinstance(evaluation, dict) else []
    if not isinstance(datasets, list) or not all(isinstance(dataset, dict) for dataset in datasets):
        raise ValueError("manifest evaluation.active_datasets must be a list of objects.")
    return datasets


def _format_setting(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, dict)):
        return f"`{json.dumps(value, ensure_ascii=True, sort_keys=True)}`"
    return str(value)


def _render_experiment_setup(manifest: dict[str, Any]) -> list[str]:
    model = manifest.get("model", {})
    calibration = manifest.get("calibration", {})
    geometry = manifest.get("sparse_geometry", {})
    runtime = manifest.get("runtime", {})
    evaluation = manifest.get("evaluation", {})
    methods = _manifest_methods(manifest)
    datasets = _manifest_datasets(manifest)

    lines = [
        "## Experimental Setup",
        "",
        f"- Experiment ID: `{manifest.get('experiment_id', '-')}`",
        f"- Manifest status: `{manifest.get('status', '-')}`",
        f"- Model: `{model.get('path', '-')}`",
        f"- Model geometry: {_format_setting(model.get('architecture'))}",
        f"- Sparse retained budget: `{geometry.get('retained_blocks', '-')}` / `{geometry.get('total_blocks', '-')}` blocks",
        f"- Sparse retained channels: `{geometry.get('retained_channels', '-')}`",
        f"- Allocation policy: {_format_setting(geometry.get('allocation_policy'))}",
        "",
        "### Shared Calibration",
        "",
        f"- Status: `{calibration.get('status', '-')}`",
        f"- Protocol: `{calibration.get('protocol_name', '-')}`",
        f"- Artifact: `{calibration.get('output_cache', '-')}`",
        f"- Shape: `{calibration.get('sequences', '-')}` sequences x `{calibration.get('sequence_length', '-')}` tokens",
        f"- Total tokens: `{calibration.get('tokens', '-')}`",
        f"- Mixing policy: {_format_setting(calibration.get('mixing_policy'))}",
        f"- Source quotas: {_format_setting(calibration.get('source_quotas'))}",
        f"- Token SHA256: `{calibration.get('input_ids_sha256') or 'pending'}`",
        "",
        "### Methods and GPUs",
        "",
    ]
    method_rows = []
    for method in methods:
        method_rows.append([
            str(method.get("name", "-")),
            str(method.get("gpu", "-")),
            str(method.get("retained_blocks", "-")),
            str(method.get("allocation", "-")),
            str(method.get("profile", "-")),
            str(method.get("status", "pending")),
        ])
    lines.extend(_markdown_table(["Method", "GPU", "Blocks", "Allocation", "Profile", "Status"], method_rows))
    lines.extend([
        "",
        "### Runtime and Decoding",
        "",
        f"- Eval batch size: `{evaluation.get('eval_batch_size', '-')}`",
        f"- Seed: `{evaluation.get('seed', '-')}`",
        f"- Generation config: {_format_setting(evaluation.get('generation_config'))}",
        f"- Thinking enabled: `{evaluation.get('enable_thinking', '-')}`",
        f"- Runtime backend: `{runtime.get('moe_backend', '-')}`",
        f"- Correction mode: `{runtime.get('correction_mode', '-')}`",
        f"- Performance collection: `{runtime.get('collect_performance', '-')}`",
        f"- Sandbox: {_format_setting(evaluation.get('sandbox'))}",
        "",
        "### Active Dataset Subsets",
        "",
    ])
    dataset_rows = []
    for dataset in datasets:
        dataset_rows.append([
            str(dataset.get("label", dataset.get("name", "-"))),
            str(dataset.get("expected_samples", "-")),
            _format_setting(dataset.get("subsets")),
            str(dataset.get("split", "-")),
            _format_setting(dataset.get("limit")),
            _format_setting(dataset.get("few_shot_num")),
            str(dataset.get("local_path", "-")),
        ])
    lines.extend(
        _markdown_table(
            ["Dataset", "Expected samples", "Subsets", "Split", "Limit", "Few-shot", "Local source"],
            dataset_rows,
        )
    )

    deferred = manifest.get("deferred_datasets", [])
    if isinstance(deferred, list) and deferred:
        lines.extend(["", "### Deferred Datasets", ""])
        deferred_rows = []
        for dataset in deferred:
            if not isinstance(dataset, dict):
                continue
            deferred_rows.append([
                str(dataset.get("label", dataset.get("name", "-"))),
                str(dataset.get("status", "deferred")),
                str(dataset.get("target_population", "-")),
                str(dataset.get("planned_samples", "-")),
                str(dataset.get("reason", "-")),
            ])
        lines.extend(_markdown_table(["Dataset", "Status", "Target population", "Planned samples", "Reason"], deferred_rows))
    return lines


def render_markdown(
    reports: list[dict[str, Any]],
    results_root: Path,
    manifest: dict[str, Any] | None = None,
) -> str:
    manifest_methods = _manifest_methods(manifest)
    manifest_datasets = _manifest_datasets(manifest)
    variants = [str(method["name"]) for method in manifest_methods if method.get("name")]
    datasets = [str(dataset["name"]) for dataset in manifest_datasets if dataset.get("name")]
    if not variants:
        variants = sorted({report["variant"] for report in reports}, key=_variant_sort_key)
    if not datasets:
        datasets = sorted({report["dataset"] for report in reports}, key=_dataset_sort_key)
    dataset_labels = {
        str(dataset["name"]): str(dataset.get("label", dataset["name"]))
        for dataset in manifest_datasets
        if dataset.get("name")
    }
    report_by_key = {(report["variant"], report["dataset"]): report for report in reports}

    completed_by_dataset = {
        dataset: sum((variant, dataset) in report_by_key for variant in variants)
        for dataset in datasets
    }
    completed_datasets = [dataset for dataset, count in completed_by_dataset.items() if count == len(variants)]
    partial_datasets = [dataset for dataset, count in completed_by_dataset.items() if 0 < count < len(variants)]

    lines = [
        f"# {manifest.get('title', 'Qwen3 50% Downstream Comparison') if manifest else 'Qwen3 50% Downstream Comparison'}",
        "",
        f"- Results root: `{results_root}`",
        f"- Completed report files: `{len(reports)}`",
        f"- Completed by all methods: `{', '.join(completed_datasets) if completed_datasets else 'none'}`",
        f"- Partially completed: `{', '.join(partial_datasets) if partial_datasets else 'none'}`",
        "",
        "## Scores",
        "",
    ]
    score_rows = []
    for variant in variants:
        score_rows.append(
            [
                variant,
                *[
                    _format_score_cell(report_by_key.get((variant, dataset)))
                    if (variant, dataset) in report_by_key
                    else "pending"
                    for dataset in datasets
                ],
            ]
        )
    lines.extend(
        _markdown_table(
            ["Variant", *(dataset_labels.get(dataset, DATASET_LABELS.get(dataset, dataset)) for dataset in datasets)],
            score_rows,
        )
    )

    lines.extend(["", "## Dataset Completion", ""])
    completion_rows = []
    for dataset in datasets:
        completed = completed_by_dataset[dataset]
        if completed == len(variants) and variants:
            status = "complete"
        elif completed:
            status = "partial"
        else:
            status = "pending"
        completion_rows.append([
            dataset_labels.get(dataset, DATASET_LABELS.get(dataset, dataset)),
            f"{completed}/{len(variants)}",
            status,
        ])
    lines.extend(_markdown_table(["Dataset", "Completed methods", "Status"], completion_rows))

    lines.extend(["", "## Performance", ""])
    performance_rows = []
    for report in sorted(
        reports,
        key=lambda item: (_variant_sort_key(item["variant"]), _dataset_sort_key(item["dataset"])),
    ):
        performance_rows.append(
            [
                report["variant"],
                dataset_labels.get(report["dataset"], DATASET_LABELS.get(report["dataset"], report["dataset"])),
                _format_samples(report["samples"]),
                _format_number(report["latency"]),
                _format_number(report["throughput"]),
                f"`{report['path']}`",
            ]
        )
    lines.extend(
        _markdown_table(
            ["Variant", "Dataset", "Samples", "Avg latency (s)", "Output tok/s", "Report"],
            performance_rows,
        )
    )
    if not reports:
        lines.extend(["", "No completed report JSON files were found yet."])
    if manifest is not None:
        lines.extend(["", *_render_experiment_setup(manifest)])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest.expanduser().resolve()) if args.manifest is not None else None
    manifest_results_root = manifest.get("paths", {}).get("results_root") if manifest is not None else None
    selected_results_root = args.results_root or manifest_results_root or DEFAULT_RESULTS_ROOT
    results_root = Path(selected_results_root).expanduser().resolve()
    reports = discover_reports(results_root)
    markdown = render_markdown(reports, results_root, manifest)
    print(markdown, end="")
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(f"Wrote comparison table to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()