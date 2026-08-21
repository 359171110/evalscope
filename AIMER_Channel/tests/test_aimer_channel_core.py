from __future__ import annotations

import torch

from AIMER_Channel.aimer_channel_core import (
    coupled_channel_aimer_importance,
    packed_channel_aimer_importance,
    rank_channels_by_aimer,
    ranking_table,
    retained_prefix,
    validate_rankings,
)


def test_channel_aimer_prefers_lower_meanabs_over_rms() -> None:
    gate = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    up = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    down = torch.tensor([[1.0, 2.0], [1.0, 0.0]])

    importance = coupled_channel_aimer_importance(gate, up, down)

    assert importance.dtype == torch.float32
    assert importance.shape == (2, )
    assert importance[1] > importance[0]


def test_packed_aimer_matches_split_gate_up() -> None:
    width = 8
    gate = torch.arange(width * 3, dtype=torch.float32).reshape(width, 3) + 1
    up = torch.arange(width * 3, dtype=torch.float32).reshape(width, 3) + 10
    down = torch.arange(3 * width, dtype=torch.float32).reshape(3, width) + 20
    packed = torch.cat((gate, up), dim=0)
    assert torch.allclose(
        packed_channel_aimer_importance(packed, down),
        coupled_channel_aimer_importance(gate, up, down),
    )


def test_near_zero_channel_cannot_outrank_a_finite_channel() -> None:
    gate = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    up = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    down = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    scores = coupled_channel_aimer_importance(gate, up, down)
    ranked = rank_channels_by_aimer(scores)
    assert torch.isneginf(scores[1])
    assert torch.equal(ranked, torch.tensor([0, 1]))


def test_ranking_is_descending_and_ties_keep_lower_index() -> None:
    scores = torch.tensor([1.0, 3.0, 3.0, 2.0])
    ranked = rank_channels_by_aimer(scores)
    assert torch.equal(ranked, torch.tensor([1, 2, 3, 0]))


def test_retained_prefix_keeps_the_highest_aimer_channels() -> None:
    scores = torch.tensor([0.1, 4.0, 2.0, 3.0])
    ranked = rank_channels_by_aimer(scores)
    kept = retained_prefix(ranked, 2)
    assert torch.equal(kept, torch.tensor([1, 3]))


def test_ranking_table_is_a_complete_permutation() -> None:
    scores = torch.tensor([[0.2, 0.9, 0.1, 0.4], [1.0, 0.0, 0.5, 0.2]])
    table = {0: ranking_table(scores, 2)}
    validate_rankings(table, 1, 2, 4, layer_ids=(0, ))
    assert torch.equal(table[0]["ranked_indices"][0], torch.tensor([1, 3, 0, 2]))
