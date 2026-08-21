from __future__ import annotations

import torch

from AIMER_Channel.aimer_channel_core import coupled_channel_aimer_importance, rank_channels_by_aimer
from AIMER_Mix.mix_core import (
    descending_unit_ranks,
    energy_balance_alpha,
    geom_channel_energy,
    l2_channel_energy,
    mix_channel_importance,
    packed_mix_channel_importance,
    path_mean_energies,
    rank_channels_by_mix,
    ranking_table,
    retained_prefix,
    validate_rankings,
)
from Magnitude.magnitude_core import coupled_channel_magnitude


def _balanced_sparse_vs_dense() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Channel 0 is dense/high-energy; channel 1 is sparse/high-AIMER."""

    gate = torch.tensor(
        [
            [2.0, 2.0, 2.0, 2.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    up = gate.clone()
    down = gate.T.clone()
    return gate, up, down


def test_balanced_path_energies_give_alpha_one() -> None:
    gate, up, down = _balanced_sparse_vs_dense()
    mean_gate, mean_up, mean_down = path_mean_energies(gate, up, down)
    alpha = energy_balance_alpha(mean_gate, mean_up, mean_down)
    assert alpha == 1.0


def test_unbalanced_path_energy_sets_alpha_to_min_over_max() -> None:
    alpha = energy_balance_alpha(1.0, 1.0, 10.0)
    assert alpha == 0.1


def test_degenerate_zero_energies_fall_back_to_alpha_one() -> None:
    assert energy_balance_alpha(0.0, 0.0, 0.0) == 1.0
    assert energy_balance_alpha(1e-12, 1e-12, 1e-12) == 1.0


def test_mix_equals_aimer_when_alpha_is_one() -> None:
    gate, up, down = _balanced_sparse_vs_dense()
    mix, alpha = mix_channel_importance(gate, up, down)
    aimer = coupled_channel_aimer_importance(gate, up, down)
    assert alpha == 1.0
    assert torch.equal(rank_channels_by_mix(mix), rank_channels_by_aimer(aimer))
    assert int(rank_channels_by_mix(mix)[0].item()) == 1


def test_low_alpha_tilts_mix_toward_geom_energy() -> None:
    gate, up, _down = _balanced_sparse_vs_dense()
    down = 100.0 * gate.T.clone()
    mix, alpha = mix_channel_importance(gate, up, down)
    aimer = coupled_channel_aimer_importance(gate, up, down)
    geom = geom_channel_energy(gate, up, down)
    assert alpha < 0.05
    assert int(rank_channels_by_aimer(aimer)[0].item()) == 1
    assert int(rank_channels_by_mix(geom)[0].item()) == 0
    assert int(rank_channels_by_mix(mix)[0].item()) == 0


def test_alpha_is_not_hardcoded_half() -> None:
    assert energy_balance_alpha(0.43, 1.0, 1.0) == 0.43
    assert energy_balance_alpha(0.87, 0.90, 1.0) == 0.87


def test_l2_energy_matches_magnitude_baseline() -> None:
    gate, up, down = _balanced_sparse_vs_dense()
    assert torch.allclose(l2_channel_energy(gate, up, down), coupled_channel_magnitude(gate, up, down))


def test_packed_mix_matches_split_gate_up() -> None:
    width = 8
    gate = torch.arange(width * 3, dtype=torch.float32).reshape(width, 3) + 1
    up = torch.arange(width * 3, dtype=torch.float32).reshape(width, 3) + 10
    down = torch.arange(3 * width, dtype=torch.float32).reshape(3, width) + 20
    packed = torch.cat((gate, up), dim=0)
    split_mix, split_alpha = mix_channel_importance(gate, up, down)
    packed_mix, packed_alpha = packed_mix_channel_importance(packed, down)
    assert split_alpha == packed_alpha
    assert torch.allclose(split_mix, packed_mix)


def test_near_zero_aimer_channel_gets_rank_zero() -> None:
    gate = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    up = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    down = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    aimer = coupled_channel_aimer_importance(gate, up, down)
    ranks = descending_unit_ranks(aimer)
    assert torch.isneginf(aimer[1])
    assert float(ranks[1].item()) == 0.0
    assert float(ranks[0].item()) == 1.0


def test_tie_ranks_are_averaged() -> None:
    ranks = descending_unit_ranks(torch.tensor([1.0, 3.0, 3.0, 0.0]))
    # descending order: two 3.0s share ordinal 0.5, then 1.0, then 0.0
    assert torch.allclose(ranks, torch.tensor([1.0 - 2.0 / 3.0, 1.0 - 0.5 / 3.0, 1.0 - 0.5 / 3.0, 0.0]))


def test_retained_prefix_keeps_highest_mix_channels() -> None:
    scores = torch.tensor([0.1, 4.0, 2.0, 3.0])
    ranked = rank_channels_by_mix(scores)
    kept = retained_prefix(ranked, 2)
    assert torch.equal(kept, torch.tensor([1, 3]))


def test_ranking_table_records_per_expert_alpha() -> None:
    scores = torch.tensor([[0.2, 0.9, 0.1, 0.4], [1.0, 0.0, 0.5, 0.2]])
    alphas = torch.tensor([0.43, 0.87])
    table = {0: ranking_table(scores, 2, expert_alpha=alphas)}
    validate_rankings(table, 1, 2, 4, layer_ids=(0, ))
    assert torch.equal(table[0]["ranked_indices"][0], torch.tensor([1, 3, 0, 2]))
    assert torch.allclose(table[0]["expert_alpha"], alphas)
    assert abs(float(table[0]["mean_alpha"]) - 0.65) < 1.0e-6
