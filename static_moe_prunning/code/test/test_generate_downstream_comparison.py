from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_downstream_comparison import discover_reports, render_markdown


def test_comparison_uses_completed_reports_and_marks_missing_datasets(tmp_path: Path) -> None:
    report_path = tmp_path / "dense" / "reports" / "qwen3-dense" / "arc.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "dataset_name": "arc",
                "model_name": "qwen3-dense",
                "score": 0.9488,
                "metrics": [{"name": "mean_acc", "score": 0.9488}],
                "num": 1172,
                "perf_metrics": {
                    "summary": {
                        "latency": {"mean": 1.25},
                        "throughput": {"avg_output_tps": 4.5},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    reports = discover_reports(tmp_path)
    markdown = render_markdown(reports, tmp_path)

    assert "| dense | 0.9488 |" in markdown
    assert "| Variant | ARC |" in markdown
    assert "| dense | ARC | 1172 | 1.2500 | 4.5000 |" in markdown
    assert "HellaSwag" not in markdown


def test_comparison_accepts_null_performance_metrics(tmp_path: Path) -> None:
    report_path = tmp_path / "aimer" / "reports" / "qwen3-aimer" / "mmlu.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "dataset_name": "mmlu",
                "model_name": "qwen3-aimer",
                "score": 0.5105,
                "num": 570,
                "perf_metrics": None,
            }
        ),
        encoding="utf-8",
    )

    markdown = render_markdown(discover_reports(tmp_path), tmp_path)

    assert "| aimer | 0.5105 |" in markdown
    assert "| aimer | MMLU | 570 | - | - |" in markdown


def test_manifest_drives_zero_and_partial_result_matrix(tmp_path: Path) -> None:
    manifest = {
        "experiment_id": "four-arm-test",
        "title": "Four-arm comparison",
        "status": "calibration_pending",
        "model": {"path": "/models/qwen3", "architecture": "48 MoE layers"},
        "calibration": {
            "status": "pending",
            "protocol_name": "mixed",
            "output_cache": "/cache/mixed.pt",
            "sequences": 128,
            "sequence_length": 2048,
            "tokens": 262144,
            "mixing_policy": "round_robin",
            "source_quotas": {"wikitext": 64, "gsm8k": 32, "arc": 16, "mbpp": 16},
            "input_ids_sha256": None,
        },
        "sparse_geometry": {
            "total_blocks": 73728,
            "retained_blocks": 36864,
            "retained_channels": 2359296,
            "allocation_policy": "method native",
        },
        "methods": [
            {"name": "dense", "gpu": 0, "retained_blocks": 73728, "allocation": "full", "profile": "dense.pt"},
            {"name": "official_reap", "gpu": 1, "retained_blocks": 36864, "allocation": "per layer", "profile": "reap.pt"},
        ],
        "runtime": {"moe_backend": "torch_index_add", "correction_mode": "none", "collect_performance": True},
        "evaluation": {
            "eval_batch_size": 1,
            "seed": 42,
            "generation_config": {"max_tokens": 1024},
            "enable_thinking": False,
            "sandbox": {"enabled": True},
            "active_datasets": [
                {
                    "name": "arc",
                    "label": "ARC-Easy + ARC-Challenge",
                    "expected_samples": 600,
                    "subsets": ["ARC-Easy", "ARC-Challenge"],
                    "split": "test",
                    "limit": 300,
                    "few_shot_num": 0,
                    "local_path": "/data/arc",
                },
                {
                    "name": "gsm8k",
                    "label": "GSM8K",
                    "expected_samples": 128,
                    "subsets": ["main"],
                    "split": "test",
                    "limit": 128,
                    "few_shot_num": 0,
                    "local_path": "/data/gsm8k",
                },
            ],
        },
        "deferred_datasets": [
            {
                "name": "live_code_bench",
                "label": "LiveCodeBench",
                "status": "deferred_downloading",
                "target_population": 182,
                "planned_samples": 40,
                "reason": "source reconstruction pending",
            }
        ],
    }

    empty_markdown = render_markdown([], tmp_path, manifest)
    assert "| dense | pending | pending |" in empty_markdown
    assert "| ARC-Easy + ARC-Challenge | 0/2 | pending |" in empty_markdown
    assert "| LiveCodeBench | deferred_downloading | 182 | 40 |" in empty_markdown
    assert "`262144`" in empty_markdown

    report_path = tmp_path / "dense" / "reports" / "qwen3-dense" / "arc.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({"dataset_name": "arc", "score": 0.75, "num": 600}), encoding="utf-8")
    partial_markdown = render_markdown(discover_reports(tmp_path), tmp_path, manifest)
    assert "| dense | 0.7500 | pending |" in partial_markdown
    assert "| ARC-Easy + ARC-Challenge | 1/2 | partial |" in partial_markdown