from __future__ import annotations

import pytest
import torch

from NAPS_v2.analyze_layer_drift import (
    is_before_divergent_token,
    token_divergence_metrics,
    vector_drift_metrics,
)


def test_vector_drift_separates_direction_and_scale() -> None:
    dense = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    scaled = dense * 2.0
    orthogonal = torch.tensor([[0.0, 1.0], [2.0, 0.0]])

    scaled_metrics = vector_drift_metrics(dense, scaled)
    orthogonal_metrics = vector_drift_metrics(dense, orthogonal)

    assert scaled_metrics["cosine_drift"] == pytest.approx(0.0)
    assert scaled_metrics["relative_l2"] == pytest.approx(1.0)
    assert scaled_metrics["rms_ratio"] == pytest.approx(2.0)
    assert orthogonal_metrics["cosine_drift"] == pytest.approx(1.0)


def test_token_divergence_reports_one_based_position() -> None:
    dense = torch.tensor([[4, 5, 6, 7]])
    pruned = torch.tensor([[4, 5, 9, 7]])

    metrics = token_divergence_metrics(dense, pruned)

    assert metrics["prediction_match_rate"] == pytest.approx(0.75)
    assert metrics["first_token_divergence"] == 3


def test_token_divergence_handles_identical_empty_sequences() -> None:
    empty = torch.empty((1, 0), dtype=torch.long)

    metrics = token_divergence_metrics(empty, empty)

    assert metrics["prediction_match_rate"] == 1.0
    assert metrics["first_token_divergence"] is None


def test_before_divergent_token_includes_predictor_state() -> None:
    assert is_before_divergent_token(2, 3)
    assert is_before_divergent_token(3, 3)
    assert not is_before_divergent_token(4, 3)
    assert is_before_divergent_token(100, None)