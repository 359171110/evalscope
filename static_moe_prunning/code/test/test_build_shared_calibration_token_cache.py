from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import torch
from datasets import Dataset

from scripts import build_shared_calibration_token_cache
from src.calibration_data import validate_calibration_token_cache_payload


class _Tokenizer:
    model_max_length = 0

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": list(range(len(text)))}


def test_builder_freezes_shared_train_only_calibration(monkeypatch, tmp_path) -> None:
    output = tmp_path / "calibration.pt"
    args = Namespace(
        model_path="/models/test",
        output_cache=output,
        dataset="wikitext",
        config="wikitext-2-raw-v1",
        split="train",
        text_field="text",
        arrow_file=[],
        sequence_length=4,
        calibration_sequences=2,
        token_offset=0,
        row_batch_size=1024,
        protocol_name="c1_test",
    )
    monkeypatch.setattr(build_shared_calibration_token_cache, "parse_args", lambda: args)
    monkeypatch.setattr(
        build_shared_calibration_token_cache,
        "load_calibration_text_dataset",
        lambda **_kwargs: (
            Dataset.from_dict({"text": ["abcd", "efgh", "ijkl"]}),
            {"dataset": "wikitext", "split": "train", "arrow_files": []},
        ),
    )
    monkeypatch.setattr(
        build_shared_calibration_token_cache.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    monkeypatch.setattr(
        build_shared_calibration_token_cache,
        "build_model_cache_identity",
        lambda *_args, **_kwargs: {
            "tokenizer_sha256": "a" * 64,
            "tokenization_config_sha256": "b" * 64,
        },
    )

    assert build_shared_calibration_token_cache.main() == 0
    payload = torch.load(output, map_location="cpu", weights_only=True)
    tokens = validate_calibration_token_cache_payload(
        payload,
        required_sequence_length=4,
    )

    assert tokens.shape == (1, 8)
    assert payload["purpose"] == "shared_moe_pruning_calibration"
    assert payload["protocol_name"] == "c1_test"
    assert payload["frozen_before_profile"] is True
    assert payload["test_metrics_used"] is False
