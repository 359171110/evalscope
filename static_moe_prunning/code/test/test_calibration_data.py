from __future__ import annotations

from pathlib import Path

import pytest
import torch
from datasets import Dataset

from src.calibration_data import (
    build_model_cache_identity,
    calibration_batches_from_payload,
    collect_contiguous_text_tokens,
    token_tensor_sha256,
    load_calibration_text_dataset,
    validate_calibration_token_cache_payload,
    validate_model_cache_compatibility,
)


class CharacterTokenizer:
    """Tiny deterministic tokenizer used to test stream boundaries."""

    model_max_length = 0

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [ord(character) for character in text]}


def _arrow_file(tmp_path: Path, name: str, texts: list[str]) -> Path:
    directory = tmp_path / name
    Dataset.from_dict({"text": texts}).save_to_disk(directory)
    return next(directory.glob("*.arrow"))


def test_explicit_arrow_calibration_source_is_ordered_and_auditable(
    tmp_path: Path,
) -> None:
    first = _arrow_file(tmp_path, "first", ["alpha", "beta"])
    second = _arrow_file(tmp_path, "second", ["gamma"])

    dataset, provenance = load_calibration_text_dataset(
        dataset_name="allenai/c4",
        dataset_config="en",
        split="train",
        text_field="text",
        arrow_files=[first, second],
    )

    assert dataset[0]["text"] == "alpha"
    assert dataset[-1]["text"] == "gamma"
    assert provenance["dataset"] == "allenai/c4"
    assert provenance["config"] == "en"
    assert provenance["split"] == "train"
    assert provenance["text_field"] == "text"
    assert provenance["source_type"] == "arrow_files"
    assert provenance["num_rows"] == 3
    assert [entry["path"] for entry in provenance["arrow_files"]] == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert all(len(entry["sha256"]) == 64 for entry in provenance["arrow_files"])


def test_calibration_source_rejects_non_train_split(tmp_path: Path) -> None:
    arrow = _arrow_file(tmp_path, "validation", ["held out"])

    with pytest.raises(ValueError, match="train split"):
        load_calibration_text_dataset(
            dataset_name="allenai/c4",
            dataset_config="en",
            split="validation",
            text_field="text",
            arrow_files=[arrow],
        )


def test_explicit_evaluation_source_can_use_validation_split(tmp_path: Path) -> None:
    arrow = _arrow_file(tmp_path, "validation-eval", ["held out"])

    dataset, provenance = load_calibration_text_dataset(
        dataset_name="allenai/c4",
        dataset_config="en-local-validation",
        split="validation",
        text_field="text",
        arrow_files=[arrow],
        require_train=False,
    )

    assert dataset[0]["text"] == "held out"
    assert provenance["split"] == "validation"
    assert provenance["source_type"] == "arrow_files"


def test_contiguous_document_stream_is_reproducible_across_offsets() -> None:
    dataset = Dataset.from_dict({"text": ["abc", "def", "ghi"]})
    tokenizer = CharacterTokenizer()

    whole, whole_metadata = collect_contiguous_text_tokens(
        tokenizer,
        dataset,
        text_field="text",
        total_tokens=8,
        token_offset=0,
        separator="\n",
        row_batch_size=2,
    )
    shifted, shifted_metadata = collect_contiguous_text_tokens(
        tokenizer,
        dataset,
        text_field="text",
        total_tokens=4,
        token_offset=4,
        separator="\n",
        row_batch_size=2,
    )

    assert whole.tolist() == [[ord(c) for c in "abc\ndef\n"]]
    assert shifted.tolist() == [[ord(c) for c in "def\n"]]
    assert shifted.tolist() == [whole.tolist()[0][4:8]]
    assert whole_metadata["tokenization_strategy"] == "joined_documents"
    assert whole_metadata["separator"] == "\\n"
    assert whole_metadata["rows_consumed"] == 3
    assert shifted_metadata["token_offset"] == 4
    assert shifted_metadata["token_end"] == 8


def test_contiguous_document_stream_rejects_missing_text_field() -> None:
    dataset = Dataset.from_dict({"body": ["abc"]})

    with pytest.raises(ValueError, match="text field"):
        collect_contiguous_text_tokens(
            CharacterTokenizer(),
            dataset,
            text_field="text",
            total_tokens=1,
        )


def test_model_cache_identity_is_path_independent_for_same_tokenizer(tmp_path: Path) -> None:
    base = tmp_path / "base"
    pruned = tmp_path / "pruned"
    for directory in (base, pruned):
        directory.mkdir()
        (directory / "config.json").write_text(
            '{"model_type":"qwen3_moe","vocab_size":10,"eos_token_id":2,"num_experts":128}',
            encoding="utf-8",
        )
        (directory / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
        (directory / "tokenizer_config.json").write_text(
            '{"eos_token":"</s>"}', encoding="utf-8"
        )
    (pruned / "config.json").write_text(
        '{"model_type":"qwen3_moe","vocab_size":10,"eos_token_id":2,"num_experts":64}',
        encoding="utf-8",
    )

    identity = build_model_cache_identity(base)
    validate_model_cache_compatibility(identity, pruned)

    assert len(identity["tokenizer_sha256"]) == 64
    assert len(identity["tokenization_config_sha256"]) == 64
    assert identity["checkpoint_path"] == str(base.resolve())


def test_model_cache_identity_rejects_tokenizer_change(tmp_path: Path) -> None:
    base = tmp_path / "base"
    changed = tmp_path / "changed"
    for directory, version in ((base, "1.0"), (changed, "2.0")):
        directory.mkdir()
        (directory / "config.json").write_text(
            '{"model_type":"qwen3_moe","vocab_size":10,"eos_token_id":2}',
            encoding="utf-8",
        )
        (directory / "tokenizer.json").write_text(
            f'{{"version":"{version}"}}', encoding="utf-8"
        )

    with pytest.raises(ValueError, match="tokenizer"):
        validate_model_cache_compatibility(build_model_cache_identity(base), changed)


def test_token_tensor_sha256_changes_with_token_content() -> None:
    first = token_tensor_sha256(torch.tensor([[1, 2, 3]], dtype=torch.long))
    second = token_tensor_sha256(torch.tensor([[1, 2, 4]], dtype=torch.long))

    assert len(first) == 64
    assert first != second


def test_shared_calibration_cache_is_train_only_and_splits_exact_batches() -> None:
    tokens = torch.arange(16, dtype=torch.long).view(1, -1)
    payload = {
        "schema_version": 1,
        "purpose": "shared_moe_pruning_calibration",
        "split": "train",
        "sequence_length": 4,
        "calibration_sequences": 4,
        "calibration_tokens": 16,
        "input_ids": tokens,
        "input_ids_sha256": token_tensor_sha256(tokens),
        "attention_mask_semantics": "all_ones_no_padding",
        "frozen_before_profile": True,
        "test_metrics_used": False,
        "source": {"dataset": "wikitext", "split": "train", "arrow_files": []},
    }

    validated = validate_calibration_token_cache_payload(
        payload,
        required_sequence_length=4,
    )
    batches = calibration_batches_from_payload(payload, required_sequence_length=4)

    assert torch.equal(validated, tokens)
    assert len(batches) == 4
    assert batches[0]["input_ids"].tolist() == [[0, 1, 2, 3]]
    assert batches[-1]["input_ids"].tolist() == [[12, 13, 14, 15]]
    assert bool(torch.stack([batch["attention_mask"].bool().all() for batch in batches]).all())


def test_shared_calibration_cache_rejects_non_train_or_unfrozen_payload() -> None:
    tokens = torch.arange(4, dtype=torch.long).view(1, -1)
    base = {
        "schema_version": 1,
        "purpose": "shared_moe_pruning_calibration",
        "split": "train",
        "sequence_length": 4,
        "calibration_sequences": 1,
        "calibration_tokens": 4,
        "input_ids": tokens,
        "input_ids_sha256": token_tensor_sha256(tokens),
        "attention_mask_semantics": "all_ones_no_padding",
        "frozen_before_profile": True,
        "test_metrics_used": False,
        "source": {"dataset": "wikitext", "split": "train", "arrow_files": []},
    }

    with pytest.raises(ValueError, match="train split"):
        validate_calibration_token_cache_payload({**base, "split": "test"}, required_sequence_length=4)
    with pytest.raises(ValueError, match="frozen"):
        validate_calibration_token_cache_payload(
            {**base, "frozen_before_profile": False},
            required_sequence_length=4,
        )
