from __future__ import annotations

import torch

from scripts.build_enp_tenp_profiles import EnpTenpAccumulator, apply_enp_zero_token_policy
from src.enp_tenp import (
    build_enp_widths,
    build_signed_projection_channel_table,
    build_tenp_widths,
    signed_projection_scores,
    trapezoid_counts,
)


class _ToyExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(3, 4, bias=False)
        self.up_proj = torch.nn.Linear(3, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 3, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        middle = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(middle)


def test_signed_projection_scores_match_explicit_channel_outputs() -> None:
    generator = torch.Generator().manual_seed(7)
    middle = torch.randn(5, 6, generator=generator)
    down_weight = torch.randn(4, 6, generator=generator)
    output = middle @ down_weight.T

    explicit = []
    for channel_idx in range(middle.shape[1]):
        contribution = middle[:, channel_idx, None] * down_weight[:, channel_idx]
        explicit.append(
            (contribution * output).sum(dim=-1) / output.norm(dim=-1).clamp_min(1.0e-8)
        )
    expected = torch.stack(explicit, dim=1).sum(dim=0)

    actual = signed_projection_scores(middle, down_weight)

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_signed_projection_channel_table_preserves_negative_score_ranking() -> None:
    raw_scores = {3: torch.tensor([[2.0, -1.0, 1.0, -3.0]])}

    table = build_signed_projection_channel_table(raw_scores, block_size=2)

    assert table[3].ranked_indices.tolist() == [[0, 2, 1, 3]]
    assert table[3].block_sizes.tolist() == [2, 2]
    assert bool((table[3].block_relative_scores >= 0).all())
    assert bool((table[3].block_coverage_scores >= 0).all())


def test_accumulator_preserves_dense_equivalent_weighted_expert_output() -> None:
    torch.manual_seed(11)
    experts = torch.nn.ModuleList([_ToyExpert(), _ToyExpert()])
    hidden_states = torch.randn(3, 3)
    selected = torch.tensor([[0, 1], [1, 0], [0, 1]])
    routing_weights = torch.tensor([[0.7, 0.3], [0.8, 0.2], [0.6, 0.4]])
    expected = torch.zeros_like(hidden_states)
    for token_idx in range(hidden_states.shape[0]):
        for slot_idx in range(selected.shape[1]):
            expert_idx = int(selected[token_idx, slot_idx].item())
            expected[token_idx] += (
                routing_weights[token_idx, slot_idx]
                * experts[expert_idx](hidden_states[token_idx : token_idx + 1]).squeeze(0)
            )

    accumulator = EnpTenpAccumulator()
    actual = accumulator.update_and_compute_output(
        2,
        hidden_states,
        experts,
        selected,
        routing_weights,
    )

    torch.testing.assert_close(actual, expected)
    assert accumulator.route_counts["all"][2].tolist() == [3, 3]
    assert accumulator.channel_score_sums["all"][2].shape == (2, 4)


def test_enp_widths_are_uniform_and_match_exact_budget() -> None:
    widths = build_enp_widths(
        num_layers=3,
        num_experts=4,
        num_blocks=12,
        routed_param_retention=0.5,
    )

    assert widths.tolist() == [[6] * 4] * 3
    assert int(widths.sum().item()) == 72


def test_enp_widths_match_wikitext_25_and_50_percent_targets() -> None:
    widths_25 = build_enp_widths(
        num_layers=48,
        num_experts=128,
        num_blocks=12,
        routed_param_retention=0.75,
    )
    widths_50 = build_enp_widths(
        num_layers=48,
        num_experts=128,
        num_blocks=12,
        routed_param_retention=0.50,
    )

    assert bool((widths_25 == 9).all())
    assert bool((widths_50 == 6).all())
    assert int(widths_25[0, 0].item()) * 64 == 576
    assert int(widths_50[0, 0].item()) * 64 == 384


def test_enp_prune_uniform_keeps_zero_token_experts_at_common_width() -> None:
    widths = torch.full((2, 3), 6, dtype=torch.long)
    zero_token_mask = torch.tensor(
        [
            [False, True, False],
            [True, False, True],
        ]
    )

    adjusted = apply_enp_zero_token_policy(
        widths,
        zero_token_mask=zero_token_mask,
        num_blocks=12,
        policy="prune_uniform",
    )

    assert bool((adjusted == 6).all())


def test_enp_error_policy_is_noop_after_zero_coverage_check_passes() -> None:
    widths = torch.full((2, 3), 6, dtype=torch.long)

    adjusted = apply_enp_zero_token_policy(
        widths,
        zero_token_mask=torch.zeros_like(widths, dtype=torch.bool),
        num_blocks=12,
        policy="error",
    )

    torch.testing.assert_close(adjusted, widths)


def test_trapezoid_counts_are_deterministic_and_deeper_layers_receive_more() -> None:
    counts = trapezoid_counts(
        [8, 8, 8, 8],
        important_expert_ratio=0.25,
        shallow_weight=1.0,
        deep_weight=2.0,
    )

    assert sum(counts) == 8
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_tenp_widths_select_important_experts_and_match_exact_budget() -> None:
    expert_scores = torch.tensor(
        [
            [9.0, 8.0, 1.0, 0.0],
            [0.0, 1.0, 8.0, 9.0],
        ]
    )

    widths, important_mask, audit = build_tenp_widths(
        expert_scores,
        num_blocks=12,
        routed_param_retention=0.5,
        important_expert_ratio=0.25,
        shallow_weight=1.0,
        deep_weight=2.0,
    )

    assert int(widths.sum().item()) == 48
    assert int(important_mask.sum().item()) == 2
    assert bool((widths[important_mask] == 12).all())
    assert bool((widths[~important_mask] < 12).all())
    assert audit["total_blocks"] == 48
    assert audit["full_experts_by_layer"] == important_mask.sum(dim=1).tolist()


def test_tenp_widths_keep_forced_experts_full_and_rebalance_layer_counts() -> None:
    expert_scores = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    forced_full_mask = torch.zeros_like(expert_scores, dtype=torch.bool)
    forced_full_mask[0, :3] = True

    widths, important_mask, audit = build_tenp_widths(
        expert_scores,
        num_blocks=10,
        routed_param_retention=0.5,
        important_expert_ratio=0.25,
        shallow_weight=1.0,
        deep_weight=2.0,
        forced_full_mask=forced_full_mask,
    )

    assert bool(important_mask[forced_full_mask].all())
    assert bool((widths[forced_full_mask] == 10).all())
    assert int(widths.sum().item()) == 80
    assert int(important_mask.sum().item()) == 4
    assert audit["forced_full_expert_count"] == 3