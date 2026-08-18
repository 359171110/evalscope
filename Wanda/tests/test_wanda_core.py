from __future__ import annotations

import torch

from Wanda.wanda_core import (
    WandaStatistics,
    build_channel_table,
    grouped_wanda_score,
    route_sample_weights,
    validate_rankings,
)


class TinyPackedExperts(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]]))
        self.down_proj = torch.nn.Parameter(torch.ones(1, 2, 2))
        self.act_fn = torch.nn.Identity()


def test_grouped_wanda_score_couples_all_three_projection_branches() -> None:
    gate = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    up = torch.tensor([[0.0, 3.0], [4.0, 0.0]])
    down = torch.tensor([[5.0, 0.0], [0.0, 6.0]])

    score = grouped_wanda_score(
        gate,
        up,
        down,
        input_square_sum=torch.tensor([4.0, 9.0]),
        middle_square_sum=torch.tensor([16.0, 25.0]),
        normalizer=1.0,
    )

    expected = torch.tensor([
        (1.0**2 * 2.0**2 + 3.0**2 * 3.0**2 + 5.0**2 * 4.0**2)**0.5,
        (2.0**2 * 3.0**2 + 4.0**2 * 2.0**2 + 6.0**2 * 5.0**2)**0.5,
    ])
    assert torch.allclose(score, expected)


def test_statistics_apply_native_router_mass_to_both_moments() -> None:
    experts = TinyPackedExperts()
    statistics = WandaStatistics((0, ), 1, 2, 2, route_weighting="mass")
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    statistics.update(
        0,
        inputs,
        torch.tensor([[0], [0]]),
        torch.tensor([[0.25], [0.75]]),
        experts,
    )
    payload = statistics.payload()

    assert torch.allclose(payload["input_square_sums"][0][0], torch.tensor([7.0, 13.0]))
    assert torch.allclose(payload["middle_square_sums"][0][0], torch.tensor([244.0, 784.0]))
    assert payload["weight_sums"][0][0].item() == 1.0
    assert payload["route_counts"][0][0].item() == 2
    assert torch.equal(route_sample_weights(torch.tensor([0.5]), "square"), torch.tensor([0.25]))


def test_channel_table_is_a_deterministic_complete_permutation() -> None:
    raw_scores = torch.tensor([[1.0, 3.0, 3.0, 2.0], [4.0, 1.0, 2.0, 3.0]])
    table = {0: build_channel_table(raw_scores, block_size=2)}

    validate_rankings(table, num_layers=1, num_experts=2, width=4)

    assert torch.equal(table[0]["ranked_indices"][0], torch.tensor([1, 2, 3, 0]))
    assert torch.equal(table[0]["block_sizes"], torch.tensor([2, 2]))