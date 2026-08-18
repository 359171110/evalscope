from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_aimer_quick9_shards import (
    CHANNEL_SHA256,
    EXPECTED_DATASETS,
    MODEL_ID,
    PROFILE_SHA256,
    merge_shards,
)


def _write_shard(root: Path, dataset: str, gpu: int, math_counts: list[int] | None = None) -> None:
    shard = root / "parallel_shards" / f"aimer_{dataset}_gpu{gpu}"
    manifest = {
        "profile_method": "aimer",
        "profile_file_sha256": PROFILE_SHA256,
        "channel_file_sha256": CHANNEL_SHA256,
        "cuda_visible_devices": [str(gpu)],
        "source_identity": {"runtime": dataset},
        "task_config": {
            "model_id": MODEL_ID,
            "datasets": [dataset],
            "seed": 42,
            "eval_batch_size": 1,
            "generation_config": {"do_sample": False, "temperature": 0.0, "max_tokens": 1024},
            "dataset_args": {dataset: {"limit": EXPECTED_DATASETS[dataset]["limit"]}},
            "model_args": {
                "enable_thinking": False,
                "correction_mode": "none",
                "moe_backend": "torch_index_add",
            },
        },
    }
    shard.mkdir(parents=True)
    (shard / "evalscope_static_profile_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = shard / "reports" / MODEL_ID / f"{dataset}.json"
    report.parent.mkdir(parents=True)
    report_payload = {"dataset_name": dataset, "num": EXPECTED_DATASETS[dataset]["samples"], "score": 0.5}
    if dataset == "math_500":
        counts = math_counts or [20, 20, 20, 20, 20]
        report_payload["metrics"] = [
            {
                "name": "mean_acc",
                "categories": [
                    {
                        "name": ["default"],
                        "subsets": [
                            {"name": f"Level {level}", "num": count, "score": 0.5, "is_aggregate": False}
                            for level, count in enumerate(counts, start=1)
                        ],
                    }
                ],
            }
        ]
    report.write_text(json.dumps(report_payload), encoding="utf-8")
    (shard / "reports" / "report.html").write_text(dataset, encoding="utf-8")
    prediction = shard / "predictions" / MODEL_ID / f"{dataset}.jsonl"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("{}\n", encoding="utf-8")


def test_merge_aimer_quick9_shards_validates_and_publishes_results(tmp_path: Path) -> None:
    for gpu, dataset in enumerate(EXPECTED_DATASETS):
        _write_shard(tmp_path, dataset, gpu)

    manifest_path = merge_shards(tmp_path)

    assert manifest_path == tmp_path / "aimer" / "aimer_quick9_merged_manifest.json"
    merged = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert merged["datasets"] == list(EXPECTED_DATASETS)
    for dataset in EXPECTED_DATASETS:
        assert (tmp_path / "aimer" / "reports" / MODEL_ID / f"{dataset}.json").is_file()


def test_merge_aimer_quick9_shards_accepts_consolidated_mmlu(tmp_path: Path) -> None:
    for gpu, dataset in enumerate(EXPECTED_DATASETS):
        _write_shard(tmp_path, dataset, gpu)

    mmlu_shard = tmp_path / "parallel_shards" / "aimer_mmlu_gpu2"
    mmlu_shard.rename(tmp_path / "parallel_shards" / "aimer_mmlu_consolidated")

    manifest_path = merge_shards(tmp_path)

    merged = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert merged["shards"]["mmlu"]["path"].endswith("aimer_mmlu_consolidated")


def test_merge_aimer_quick9_shards_rejects_unbalanced_math_levels(tmp_path: Path) -> None:
    for gpu, dataset in enumerate(EXPECTED_DATASETS):
        math_counts = [20, 20, 20, 19, 21] if dataset == "math_500" else None
        _write_shard(tmp_path, dataset, gpu, math_counts=math_counts)

    with pytest.raises(ValueError, match="20 samples from each difficulty level"):
        merge_shards(tmp_path)