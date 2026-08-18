from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import torch

from scripts import build_wikitext_full_token_cache
from src.corpus_ppl import validate_token_cache_payload


class _Tokenizer:
    model_max_length = 0

    def __call__(self, text, **kwargs):
        del kwargs
        return SimpleNamespace(
            input_ids=torch.arange(len(text), dtype=torch.long).view(1, -1)
        )


def test_builder_freezes_model_tokenized_full_corpus(monkeypatch, tmp_path) -> None:
    output = tmp_path / "tokens.pt"
    args = Namespace(
        model_path="/models/test",
        output_cache=output,
        arrow_file=[],
        sequence_length=2048,
        expected_windows=1,
        min_text_length=3,
        protocol_name="test_full_wikitext",
    )
    monkeypatch.setattr(build_wikitext_full_token_cache, "parse_args", lambda: args)
    monkeypatch.setattr(
        build_wikitext_full_token_cache,
        "load_calibration_text_dataset",
        lambda **_kwargs: (
            {"text": ["ab", "cde", "fghi"]},
            {"arrow_files": [{"sha256": "a" * 64}]},
        ),
    )
    monkeypatch.setattr(
        build_wikitext_full_token_cache.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    monkeypatch.setattr(
        build_wikitext_full_token_cache,
        "build_model_cache_identity",
        lambda *_args, **_kwargs: {
            "tokenizer_sha256": "b" * 64,
            "tokenization_config_sha256": "c" * 64,
        },
    )

    assert build_wikitext_full_token_cache.main() == 0
    payload = torch.load(output, map_location="cpu", weights_only=True)
    tokens = validate_token_cache_payload(payload, required_sequence_length=2048)

    assert tokens.shape == (1, 7)
    assert payload["evaluation_windows"] == 1
    assert payload["evaluation_tokens"] == 7
    assert payload["token_stream"]["rows_selected"] == 2
    assert len(payload["input_ids_sha256"]) == 64
    assert payload["attention_mask_semantics"] == "all_ones_no_padding"
    assert payload["model_identity"]["tokenizer_sha256"] == "b" * 64
    assert payload["frozen_before_evaluation"] is True
    assert payload["test_metrics_used"] is False
