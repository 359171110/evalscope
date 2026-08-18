from __future__ import annotations

from pathlib import Path

import pytest

from scripts.consolidate_aimer_subset_predictions import MODEL_ID, consolidate_predictions


def _write_prediction(work_dir: Path, name: str, content: str) -> Path:
    path = work_dir / "predictions" / MODEL_ID / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_consolidate_predictions_copies_non_overlapping_subset_files(tmp_path: Path) -> None:
    destination = tmp_path / "canonical"
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    _write_prediction(source_a, "math_500_Level 4.jsonl", "{}\n")
    _write_prediction(source_b, "math_500_Level 5.jsonl", "{\"index\": 1}\n")

    copied = consolidate_predictions(destination, [source_a, source_b])

    assert len(copied) == 2
    assert (destination / "predictions" / MODEL_ID / "math_500_Level 4.jsonl").is_file()
    assert (destination / "predictions" / MODEL_ID / "math_500_Level 5.jsonl").is_file()


def test_consolidate_predictions_rejects_conflicting_subset_files(tmp_path: Path) -> None:
    destination = tmp_path / "canonical"
    source = tmp_path / "source"
    _write_prediction(destination, "mmlu_anatomy.jsonl", "{}\n")
    _write_prediction(source, "mmlu_anatomy.jsonl", "{\"different\": true}\n")

    with pytest.raises(ValueError, match="Conflicting prediction shard"):
        consolidate_predictions(destination, [source])


def test_consolidate_predictions_skips_partial_files_when_complete_source_exists(tmp_path: Path) -> None:
    destination = tmp_path / "canonical"
    partial = tmp_path / "partial"
    complete = tmp_path / "complete"
    _write_prediction(partial, "mmlu_anatomy.jsonl", "{}\n" * 4)
    _write_prediction(complete, "mmlu_anatomy.jsonl", "{}\n" * 10)

    copied = consolidate_predictions(
        destination,
        [partial, complete],
        expected_records_per_file=10,
    )

    assert len(copied) == 1
    target = destination / "predictions" / MODEL_ID / "mmlu_anatomy.jsonl"
    assert len(target.read_text(encoding="utf-8").splitlines()) == 10


def test_consolidate_predictions_rejects_conflicting_complete_files(tmp_path: Path) -> None:
    destination = tmp_path / "canonical"
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    _write_prediction(source_a, "mmlu_anatomy.jsonl", "{}\n" * 10)
    _write_prediction(source_b, "mmlu_anatomy.jsonl", '{"different": true}\n' * 10)

    with pytest.raises(ValueError, match="Conflicting complete prediction shards"):
        consolidate_predictions(
            destination,
            [source_a, source_b],
            expected_records_per_file=10,
        )