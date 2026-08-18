from __future__ import annotations

import pytest
import torch

from src.tail_risk import (
    blend_typical_and_tail_score,
    build_consensus_rare_event_risk_floors,
    build_rare_event_risk_floors,
    expert_tail_risk_from_channels,
    slice_contiguous_token_span,
)


def test_tail_blend_endpoints_preserve_source_scores() -> None:
    typical = torch.tensor([1.0, 4.0])
    tail = torch.tensor([9.0, 16.0])

    assert torch.equal(
        blend_typical_and_tail_score(typical, tail, tail_lambda=0.0), typical
    )
    assert torch.equal(
        blend_typical_and_tail_score(typical, tail, tail_lambda=1.0), tail
    )


def test_tail_blend_uses_geometric_interpolation() -> None:
    typical = torch.tensor([1.0, 4.0])
    tail = torch.tensor([9.0, 16.0])

    blended = blend_typical_and_tail_score(typical, tail, tail_lambda=0.5)

    assert torch.allclose(blended, torch.tensor([3.0, 8.0]))


def test_tail_blend_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError, match="tail_lambda"):
        blend_typical_and_tail_score(torch.ones(2), torch.ones(2), tail_lambda=1.1)


def test_expert_tail_risk_uses_largest_channel_contribution() -> None:
    max_abs = torch.tensor([[2.0, 10.0, 4.0], [3.0, 2.0, 1.0]])
    down_norm = torch.tensor([[5.0, 0.5, 1.0], [1.0, 4.0, 2.0]])

    risk = expert_tail_risk_from_channels(max_abs, down_norm)

    assert torch.equal(risk, torch.tensor([10.0, 8.0]))


def test_rare_event_floors_use_global_tail_threshold_but_only_early_layers() -> None:
    risk = torch.tensor(
        [
            [1.0, 50.0, 2.0],
            [3.0, 80.0, 4.0],
            [100.0, 90.0, 5.0],
        ]
    )

    floors, metadata = build_rare_event_risk_floors(
        risk,
        early_layer_count=2,
        global_quantile=0.5,
        relative_to_global_max=0.4,
        minimum_width=2,
        num_blocks=12,
    )

    assert floors.tolist() == [[0, 2, 0], [0, 2, 0], [0, 0, 0]]
    assert metadata["selected_experts"] == [
        {"layer": 0, "expert": 1, "risk": 50.0, "min_width": 2},
        {"layer": 1, "expert": 1, "risk": 80.0, "min_width": 2},
    ]
    assert metadata["threshold"] == pytest.approx(40.0)


def test_rare_event_floors_reject_invalid_protocol_parameters() -> None:
    risk = torch.ones(2, 3)

    with pytest.raises(ValueError, match="global_quantile"):
        build_rare_event_risk_floors(
            risk,
            early_layer_count=1,
            global_quantile=1.1,
            relative_to_global_max=0.1,
            minimum_width=1,
            num_blocks=12,
        )
    with pytest.raises(ValueError, match="minimum_width"):
        build_rare_event_risk_floors(
            risk,
            early_layer_count=1,
            global_quantile=0.995,
            relative_to_global_max=0.1,
            minimum_width=13,
            num_blocks=12,
        )


def test_calibration_token_span_honors_nonoverlapping_offset() -> None:
    tokens = torch.arange(20).view(1, 20)

    selected = slice_contiguous_token_span(tokens, total_tokens=6, token_offset=8)

    assert selected.tolist() == [[8, 9, 10, 11, 12, 13]]


def test_calibration_token_span_rejects_out_of_range_request() -> None:
    with pytest.raises(ValueError, match="enough calibration tokens"):
        slice_contiguous_token_span(
            torch.arange(10).view(1, 10), total_tokens=6, token_offset=5
        )


def test_consensus_risk_floors_require_cross_interval_votes() -> None:
    risks = torch.tensor(
        [
            [[10.0, 1.0, 8.0]],
            [[9.0, 7.0, 1.0]],
            [[8.0, 6.0, 2.0]],
        ]
    )

    floors, metadata = build_consensus_rare_event_risk_floors(
        risks,
        early_layer_count=1,
        global_quantile=0.5,
        relative_to_global_max=0.0,
        minimum_width=2,
        num_blocks=12,
        minimum_votes=2,
    )

    assert floors.tolist() == [[2, 2, 0]]
    assert metadata["fold_count"] == 3
    assert metadata["minimum_votes"] == 2
    assert metadata["selected_experts"] == [
        {"layer": 0, "expert": 0, "votes": 3, "min_width": 2},
        {"layer": 0, "expert": 1, "votes": 2, "min_width": 2},
    ]


def test_consensus_risk_floors_reject_impossible_vote_count() -> None:
    with pytest.raises(ValueError, match="minimum_votes"):
        build_consensus_rare_event_risk_floors(
            torch.ones(2, 1, 3),
            early_layer_count=1,
            global_quantile=0.5,
            relative_to_global_max=0.0,
            minimum_width=1,
            num_blocks=12,
            minimum_votes=3,
        )
