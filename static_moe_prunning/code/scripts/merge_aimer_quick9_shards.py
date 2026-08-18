from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


MODEL_ID = "qwen3-mixed-512x1024-global-quick9-50pct-aimer"
PROFILE_SHA256 = "93092a6200ce6f8427664404bac8c1aa06e1f368f8c9253bc9943dda9c9755b8"
CHANNEL_SHA256 = "2c31a313d8f049f94dc475b3a2bde044842afcd8f558f31d24c7b397a17c6148"
EXPECTED_DATASETS = {
    "arc": {"limit": 300, "samples": 600},
    "hellaswag": {"limit": 1000, "samples": 1000},
    "mmlu": {"limit": 10, "samples": 570},
    "winogrande": {"limit": 400, "samples": 400},
    "gsm8k": {"limit": 128, "samples": 128},
    "math_500": {"limit": 20, "samples": 100},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and merge completed AIMER quick9 dataset shards.")
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _validate_manifest(path: Path, dataset: str) -> dict[str, Any]:
    manifest = _load_json(path)
    task = manifest.get("task_config")
    if not isinstance(task, dict):
        raise ValueError(f"task_config is missing from {path}")
    expected = EXPECTED_DATASETS[dataset]
    checks = {
        "profile method": manifest.get("profile_method") == "aimer",
        "profile SHA256": manifest.get("profile_file_sha256") == PROFILE_SHA256,
        "channel SHA256": manifest.get("channel_file_sha256") == CHANNEL_SHA256,
        "model ID": task.get("model_id") == MODEL_ID,
        "dataset": task.get("datasets") == [dataset],
        "seed": task.get("seed") == 42,
        "eval batch size": task.get("eval_batch_size") == 1,
        "thinking disabled": task.get("model_args", {}).get("enable_thinking") is False,
        "correction disabled": task.get("model_args", {}).get("correction_mode") == "none",
        "MoE backend": task.get("model_args", {}).get("moe_backend") == "torch_index_add",
        "generation config": task.get("generation_config")
        == {"do_sample": False, "temperature": 0.0, "max_tokens": 1024},
        "dataset limit": task.get("dataset_args", {}).get(dataset, {}).get("limit") == expected["limit"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"AIMER shard manifest mismatch for {dataset}: {', '.join(failed)}")
    return manifest


def _find_dataset_report(shard: Path, dataset: str) -> Path:
    candidates = []
    for path in (shard / "reports").rglob("*.json"):
        payload = _load_json(path)
        if payload.get("dataset_name") == dataset:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one aggregate {dataset} report in {shard}, found {len(candidates)}.")
    report_path = candidates[0]
    report = _load_json(report_path)
    if report.get("num") != EXPECTED_DATASETS[dataset]["samples"]:
        raise ValueError(
            f"Unexpected {dataset} report sample count: {report.get('num')} != "
            f"{EXPECTED_DATASETS[dataset]['samples']}"
        )
    if dataset == "math_500":
        _validate_math_level_counts(report)
    return report_path


def _validate_math_level_counts(report: dict[str, Any]) -> None:
    expected = {f"Level {level}": 20 for level in range(1, 6)}
    observed: dict[str, int] = {}
    for metric in report.get("metrics", []):
        for category in metric.get("categories", []):
            for subset in category.get("subsets", []):
                name = subset.get("name")
                if name in expected:
                    observed[str(name)] = int(subset.get("num", -1))
    if observed != expected:
        raise ValueError(
            "AIMER math_500 report must contain exactly 20 samples from each difficulty level; "
            f"observed {observed}."
        )


def _copy_tree_contents(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError(f"Refusing to overwrite a different merged result file: {target}")
        shutil.copy2(path, target)


def merge_shards(results_root: Path) -> Path:
    root = results_root.expanduser().resolve()
    shard_root = root / "parallel_shards"
    destination = root / "aimer"
    manifests: dict[str, dict[str, Any]] = {}
    shards: dict[str, Path] = {}
    for dataset in EXPECTED_DATASETS:
        consolidated = shard_root / f"aimer_{dataset}_consolidated"
        matches = [consolidated] if consolidated.is_dir() else sorted(shard_root.glob(f"aimer_{dataset}_gpu*"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one AIMER shard for {dataset}, found {len(matches)}.")
        shard = matches[0]
        manifests[dataset] = _validate_manifest(shard / "evalscope_static_profile_manifest.json", dataset)
        _find_dataset_report(shard, dataset)
        shards[dataset] = shard

    destination.mkdir(parents=True, exist_ok=True)
    for dataset, shard in shards.items():
        for directory in ("predictions", "reviews"):
            _copy_tree_contents(shard / directory, destination / directory)
        _copy_tree_contents(
            shard / "reports" / MODEL_ID,
            destination / "reports" / MODEL_ID,
        )
        _copy_tree_contents(shard / "logs", destination / "logs" / dataset)
    merged_manifest = {
        "schema_version": 1,
        "method": "aimer",
        "model_id": MODEL_ID,
        "profile_file_sha256": PROFILE_SHA256,
        "channel_file_sha256": CHANNEL_SHA256,
        "datasets": list(EXPECTED_DATASETS),
        "expected_samples": {dataset: values["samples"] for dataset, values in EXPECTED_DATASETS.items()},
        "shards": {
            dataset: {
                "path": str(shards[dataset]),
                "cuda_visible_devices": manifests[dataset].get("cuda_visible_devices"),
                "source_identity": manifests[dataset].get("source_identity"),
            }
            for dataset in EXPECTED_DATASETS
        },
    }
    manifest_path = destination / "aimer_quick9_merged_manifest.json"
    manifest_path.write_text(json.dumps(merged_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    print(merge_shards(args.results_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())