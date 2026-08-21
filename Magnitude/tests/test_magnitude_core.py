from __future__ import annotations

import torch

from Magnitude.magnitude_core import (
    coupled_channel_magnitude,
    packed_channel_magnitude,
    rank_channels_by_magnitude,
    ranking_table,
    retained_prefix,
    validate_rankings,
)


def test_coupled_magnitude_uses_fp32_l2_of_gate_up_down() -> None:
    gate = torch.zeros(4, 2, dtype=torch.bfloat16)
    up = torch.zeros(4, 2, dtype=torch.bfloat16)
    down = torch.zeros(2, 4, dtype=torch.bfloat16)
    gate[1, 0] = 3
    up[1, 1] = 4
    down[0, 1] = 12
    scores = coupled_channel_magnitude(gate, up, down)
    assert scores.dtype == torch.float32
    assert torch.isclose(scores[1], torch.tensor(13.0))
    assert torch.isclose(scores[0], torch.tensor(0.0))


def test_packed_magnitude_matches_split_gate_up() -> None:
    width = 8
    gate = torch.arange(width * 3, dtype=torch.float32).reshape(width, 3)
    up = torch.arange(width * 3, dtype=torch.float32).reshape(width, 3) + 10
    down = torch.arange(3 * width, dtype=torch.float32).reshape(3, width) + 20
    packed = torch.cat((gate, up), dim=0)
    assert torch.allclose(packed_channel_magnitude(packed, down), coupled_channel_magnitude(gate, up, down))


def test_ranking_is_descending_and_ties_keep_lower_index() -> None:
    scores = torch.tensor([1.0, 3.0, 3.0, 2.0])
    ranked = rank_channels_by_magnitude(scores)
    assert torch.equal(ranked, torch.tensor([1, 2, 3, 0]))


def test_retained_prefix_keeps_the_highest_magnitude_channels() -> None:
    scores = torch.tensor([0.1, 4.0, 2.0, 3.0])
    ranked = rank_channels_by_magnitude(scores)
    kept = retained_prefix(ranked, 2)
    assert torch.equal(kept, torch.tensor([1, 3]))


def test_ranking_table_is_a_complete_permutation() -> None:
    scores = torch.tensor([[0.2, 0.9, 0.1, 0.4], [1.0, 0.0, 0.5, 0.2]])
    table = {0: ranking_table(scores, 2)}
    validate_rankings(table, 1, 2, 4, layer_ids=(0, ))
    assert torch.equal(table[0]["ranked_indices"][0], torch.tensor([1, 3, 0, 2]))
