from __future__ import annotations

import torch

from scripts.build_mixed_calibration_token_cache import _render_code_instruction
from scripts.build_mixed_calibration_token_cache import _render_math
from scripts.build_mixed_calibration_token_cache import _round_robin_sequences
from scripts.build_mixed_calibration_token_cache import _tokenize_source


def test_round_robin_sequences_preserves_quotas_and_source_order() -> None:
    mixed, order = _round_robin_sequences([
        ("general", torch.tensor([[1, 2], [3, 4], [5, 6]])),
        ("math", torch.tensor([[7, 8], [9, 10]])),
        ("code", torch.tensor([[11, 12]])),
    ])

    assert order == ["general", "math", "code", "general", "math", "general"]
    assert mixed.tolist() == [[1, 2, 7, 8, 11, 12, 3, 4, 9, 10, 5, 6]]


def test_tokenize_source_can_repeat_train_text_explicitly() -> None:
    class Tokenizer:
        model_max_length = 0

        def __call__(self, text, **kwargs):
            del kwargs
            return {"input_ids": list(range(len(text)))}

    dataset = [{"text": "abc"}]
    tokens, metadata = _tokenize_source(
        Tokenizer(),
        dataset,
        lambda record: record["text"],
        sequences=2,
        sequence_length=4,
        row_batch_size=1,
        allow_source_repetition=True,
    )

    assert tokens.shape == (2, 4)
    assert metadata["selected_tokens"] == 8


def test_render_math_includes_problem_and_solution() -> None:
    assert _render_math({"problem": "Compute 1+1.", "solution": "2"}) == (
        "Problem: Compute 1+1.\nSolution: 2"
    )


def test_render_code_instruction_includes_optional_input() -> None:
    assert _render_code_instruction({
        "instruction": "Write a sum function.",
        "input": "[1, 2]",
        "output": "def add(values): return sum(values)",
    }) == (
        "Task: Write a sum function.\nInput:\n[1, 2]\n"
        "Solution:\ndef add(values): return sum(values)"
    )