from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from src.corpus_ppl import (
    FrozenTokenCorpusPerplexity,
    frozen_protocol_matches,
    validate_token_cache_payload,
)


class ConstantLossModel(torch.nn.Module):
    def __init__(self, loss: float):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.loss_value = float(loss)

    def forward(self, input_ids, labels, use_cache=False):
        del input_ids, labels, use_cache
        return SimpleNamespace(loss=self.anchor.new_tensor(self.loss_value))


def test_frozen_token_corpus_reports_exact_windows_tokens_and_ppl() -> None:
    evaluator = FrozenTokenCorpusPerplexity(
        ConstantLossModel(math.log(2.0)),
        torch.arange(10).view(1, 10),
    )

    metrics = evaluator.calculate_corpus_ppl(n_ctx=4)

    assert metrics["ppl"] == pytest.approx(2.0)
    assert metrics["windows"] == 3
    assert metrics["tokens"] == 10


def test_token_cache_payload_requires_frozen_single_sequence() -> None:
    payload = {
        "schema_version": 1,
        "input_ids": torch.arange(8).view(1, 8),
        "dataset": "allenai/c4",
        "split": "validation",
        "sequence_length": 4,
        "evaluation_tokens": 8,
        "frozen_before_evaluation": True,
        "test_metrics_used": False,
        "source": {"arrow_files": [{"sha256": "a" * 64}]},
    }

    tokens = validate_token_cache_payload(payload, required_sequence_length=4)

    assert tokens.tolist() == [list(range(8))]
    with pytest.raises(ValueError, match="frozen"):
        validate_token_cache_payload(
            {**payload, "frozen_before_evaluation": False},
            required_sequence_length=4,
        )
    with pytest.raises(ValueError, match="evaluation_tokens"):
        validate_token_cache_payload(
            {**payload, "evaluation_tokens": 7},
            required_sequence_length=4,
        )


def test_frozen_protocol_requires_exact_windows_and_tokens() -> None:
    payload = {"evaluation_windows": 2, "evaluation_tokens": 8}

    assert frozen_protocol_matches(
        {"windows": 2, "tokens": 8}, payload, max_windows=None
    )
    assert not frozen_protocol_matches(
        {"windows": 2, "tokens": 7}, payload, max_windows=None
    )
    assert not frozen_protocol_matches(
        {"windows": 2, "tokens": 8}, payload, max_windows=2
    )
