from __future__ import annotations

import math

from src.multiple_choice_eval import aggregate_truthfulqa_metrics


def test_truthfulqa_metrics_match_mc1_and_mc2_definitions() -> None:
    rows = [
        {
            "mc1_scores": [2.0, 0.0, -1.0],
            "mc1_labels": [1, 0, 0],
            "mc2_scores": [2.0, 0.0, -1.0],
            "mc2_labels": [1, 0, 0],
        },
        {
            "mc1_scores": [0.0, 1.0],
            "mc1_labels": [1, 0],
            "mc2_scores": [0.0, 1.0, 2.0],
            "mc2_labels": [1, 0, 1],
        },
    ]

    metrics = aggregate_truthfulqa_metrics(rows)

    assert metrics["examples"] == 2
    assert metrics["mc1_accuracy"] == 0.5
    first_mc2 = math.exp(2.0) / (math.exp(2.0) + 1.0 + math.exp(-1.0))
    second_mc2 = (1.0 + math.exp(2.0)) / (
        1.0 + math.exp(1.0) + math.exp(2.0)
    )
    assert math.isclose(
        metrics["mc2_true_probability"],
        0.5 * (first_mc2 + second_mc2),
        rel_tol=1.0e-12,
    )


def test_truthfulqa_metrics_reject_malformed_rows() -> None:
    malformed = [
        {
            "mc1_scores": [1.0],
            "mc1_labels": [1, 0],
            "mc2_scores": [1.0],
            "mc2_labels": [1],
        }
    ]

    try:
        aggregate_truthfulqa_metrics(malformed)
    except ValueError as error:
        assert "length" in str(error)
    else:
        raise AssertionError("malformed TruthfulQA rows must be rejected")
