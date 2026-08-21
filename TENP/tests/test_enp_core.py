from __future__ import annotations

import torch

from TENP.enp_core import (
    EnpStatistics,
    build_channel_table,
    enp_cos_token_scores,
    expert_channel_response,
    expert_down_weight,
    expert_importance_scores,
    validate_rankings,
    weight_only_group_score,
)


class TinyPackedExperts(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]]))
        self.down_proj = torch.nn.Parameter(torch.ones(1, 2, 2))
        self.act_fn = torch.nn.Identity()


def test_enp_cos_matches_explicit_per_channel_projection() -> None:
    middle = torch.tensor([[1.0, 2.0, 0.5], [0.0, 1.0, 3.0]])
    down = torch.tensor([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 1.0],
        [0.0, 0.5, 0.0],
    ])
    output = middle @ down.T
    expected = []
    for channel in range(middle.shape[1]):
        token_scores = []
        for token in range(middle.shape[0]):
            contribution = middle[token, channel] * down[:, channel]
            token_scores.append(torch.dot(contribution, output[token]) / (output[token].norm() + 1.0e-8))
        expected.append(sum(token_scores))

    scores = enp_cos_token_scores(middle, down)

    assert torch.allclose(scores, torch.stack(expected), atol=1.0e-6)


def test_mean_enp_score_divides_unique_token_sum() -> None:
    score_sum = torch.tensor([4.0, 6.0])
    gate = torch.ones(2, 3)
    up = torch.ones(2, 3)
    down = torch.ones(3, 2)

    mean_score, used_fallback = expert_importance_scores(score_sum, 2, gate, up, down)

    assert used_fallback is False
    assert torch.allclose(mean_score, torch.tensor([2.0, 3.0]))


def test_zero_token_expert_uses_weight_only_l2_fallback() -> None:
    score_sum = torch.zeros(2)
    gate = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    up = torch.zeros(2, 2)
    down = torch.zeros(2, 2)

    score, used_fallback = expert_importance_scores(score_sum, 0, gate, up, down)

    assert used_fallback is True
    assert torch.allclose(score, weight_only_group_score(gate, up, down))
    assert torch.allclose(score, torch.tensor([3.0, 4.0]))


def test_statistics_count_unique_routed_tokens_only() -> None:
    experts = TinyPackedExperts()
    statistics = EnpStatistics((0, ), 1, 2, 2)
    inputs = torch.tensor([[1.0, 2.0]])

    statistics.update(
        0,
        inputs,
        torch.tensor([[0, 0]]),
        torch.tensor([[0.4, 0.6]]),
        experts,
    )
    payload = statistics.payload()

    assert payload["score_mode"] == "enp_cos"
    assert payload["token_aggregation"] == "unique_routed_mean"
    assert payload["route_counts"][0][0].item() == 1
    middle = expert_channel_response(experts, inputs, 0)
    down = expert_down_weight(experts, 0)
    assert torch.allclose(payload["channel_score_sums"][0][0], enp_cos_token_scores(middle, down))


def test_channel_table_is_a_deterministic_complete_permutation() -> None:
    raw_scores = torch.tensor([[1.0, 3.0, 3.0, 2.0], [4.0, 1.0, 2.0, 3.0]])
    table = {0: build_channel_table(raw_scores, block_size=2)}

    validate_rankings(table, num_layers=1, num_experts=2, width=4)

    assert torch.equal(table[0]["ranked_indices"][0], torch.tensor([1, 2, 3, 0]))
    assert torch.equal(table[0]["block_sizes"], torch.tensor([2, 2]))
