from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
import torch

from scripts.calibrate_hessian_channels import continuous_calibration_tokens as hessian_calibration_tokens
from scripts.collect_dynamic_regret_teacher import continuous_calibration_tokens as teacher_calibration_tokens
from src.calibration_data import token_tensor_sha256


class CharacterTokenizer:
    model_max_length = 0

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [ord(character) for character in text]}


def test_teacher_uses_explicit_auditable_calibration_source(tmp_path: Path) -> None:
    directory = tmp_path / "c4"
    Dataset.from_dict({"text": ["abc", "def", "ghi"]}).save_to_disk(directory)
    arrow = next(directory.glob("*.arrow"))
    args = SimpleNamespace(
        calibration_dataset="allenai/c4",
        calibration_config="en-local-first-shard",
        calibration_split="train",
        calibration_text_field="text",
        calibration_arrow_file=[arrow],
        calibration_token_offset=4,
        calibration_row_batch_size=2,
    )

    tokens, source = teacher_calibration_tokens(
        CharacterTokenizer(), args, total_tokens=4, device="cpu"
    )

    assert tokens.tolist() == [[ord(character) for character in "def\n"]]
    assert source["dataset"] == "allenai/c4"
    assert source["split"] == "train"
    assert source["arrow_files"][0]["path"] == str(arrow.resolve())
    assert len(source["arrow_files"][0]["sha256"]) == 64
    assert source["token_stream"]["token_offset"] == 4
    assert source["token_stream"]["token_end"] == 8


def test_collectors_use_exact_shared_calibration_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "shared.pt"
    tokens = torch.arange(8, dtype=torch.long).view(1, -1)
    torch.save(
        {
            "schema_version": 1,
            "purpose": "shared_moe_pruning_calibration",
            "protocol_name": "c1_test",
            "split": "train",
            "sequence_length": 4,
            "calibration_sequences": 2,
            "calibration_tokens": 8,
            "input_ids": tokens,
            "input_ids_sha256": token_tensor_sha256(tokens),
            "attention_mask_semantics": "all_ones_no_padding",
            "frozen_before_profile": True,
            "test_metrics_used": False,
            "source": {"dataset": "wikitext", "split": "train", "arrow_files": []},
        },
        cache_path,
    )
    args = SimpleNamespace(
        model_path=None,
        calibration_token_cache=cache_path,
        sequence_length=4,
        calibration_dataset="unused",
        calibration_config=None,
        calibration_split="train",
        calibration_text_field="text",
        calibration_arrow_file=[],
        calibration_token_offset=0,
        calibration_row_batch_size=1,
    )

    for loader in (hessian_calibration_tokens, teacher_calibration_tokens):
        actual, source = loader(None, args, total_tokens=8, device="cpu")

        assert torch.equal(actual, tokens)
        assert source["source_type"] == "shared_calibration_token_cache"
        assert source["input_ids_sha256"] == token_tensor_sha256(tokens)
        assert len(source["cache_file_sha256"]) == 64
