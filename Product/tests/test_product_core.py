from __future__ import annotations

import torch

from Product.product_core import (
    coupled_channel_product,
    packed_channel_product,
    rank_channels_by_product,
    ranking_table,
    retained_prefix,
    validate_rankings,
)


def test_coupled_product_is_gate_l2_times_up_l2_and_ignores_down() -> None:
    gate = torch.zeros(4, 2, dtype=torch.bfloat16)
    up = torch.zeros(4, 2, dtype=torch.bfloat16)
    down = torch.zeros(2, 4, dtype=torch.bfloat16)
    gate[1, 0] = 3
    up[1, 1] = 4
    down[0, 1] = 12
    scores = coupled_channel_product(gate, up, down)
    assert scores.dtype == torch.float32
    assert torch.isclose(scores[1], torch.tensor(12.0))
    assert torch.isclose(scores[0], torch.tensor(0.0))


def test_zero_up_zeros_the_product_even_if_gate_is_large() -> None:
    gate = torch.ones(2, 3)
    up = torch.zeros(2, 3)
    down = torch.ones(3, 2)
    scores = coupled_channel_product(gate, up, down)
    assert torch.allclose(scores, torch.zeros(2))


def test_packed_product_matches_split_gate_up() -> None:
    width = 8
    gate = torch.arange(width * 3, dtype=torch.float32).reshape(width, 3)
    up = torch.arange(width * 3, dtype=torch.float32).reshape(width, 3) + 10
    down = torch.arange(3 * width, dtype=torch.float32).reshape(3, width) + 20
    packed = torch.cat((gate, up), dim=0)
    assert torch.allclose(packed_channel_product(packed, down), coupled_channel_product(gate, up, down))


def test_ranking_is_descending_and_ties_keep_lower_index() -> None:
    scores = torch.tensor([1.0, 3.0, 3.0, 2.0])
    ranked = rank_channels_by_product(scores)
    assert torch.equal(ranked, torch.tensor([1, 2, 3, 0]))


def test_retained_prefix_keeps_the_highest_product_channels() -> None:
    scores = torch.tensor([0.1, 4.0, 2.0, 3.0])
    ranked = rank_channels_by_product(scores)
    kept = retained_prefix(ranked, 2)
    assert torch.equal(kept, torch.tensor([1, 3]))


def test_ranking_table_is_a_complete_permutation() -> None:
    scores = torch.tensor([[0.2, 0.9, 0.1, 0.4], [1.0, 0.0, 0.5, 0.2]])
    table = {0: ranking_table(scores, 2)}
    validate_rankings(table, 1, 2, 4, layer_ids=(0, ))
    assert torch.equal(table[0]["ranked_indices"][0], torch.tensor([1, 3, 0, 2]))
