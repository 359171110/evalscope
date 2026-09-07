from __future__ import annotations

import torch

from HARP.rank_analysis_core import (
    adjacent_gaps,
    build_layer_rows,
    cohens_d_prefix,
    decide_allocation,
    jaccard,
    one_change_point,
    ordinary_z,
    participation_mass,
    pearson,
    ranks_descending,
    robust_z,
    spearman,
)


def test_ranks_are_stable_and_one_based() -> None:
    values = torch.tensor([0.2, 0.9, 0.9, 0.1])
    ranks = ranks_descending(values)
    assert ranks.tolist() == [3, 1, 2, 4]


def test_robust_z_is_large_for_a_single_spike() -> None:
    values = torch.tensor([1.0] * 20 + [8.0])
    rz = robust_z(values.unsqueeze(0)).reshape(-1)
    assert float(rz[-1].item()) > 3.0
    assert float(ordinary_z(values.unsqueeze(0)).reshape(-1)[-1].item()) > 2.0


def test_one_change_point_finds_a_head_not_only_the_first_item() -> None:
    head = torch.tensor([2.0, 1.9, 1.85, 1.8])
    bulk = torch.full((20,), 1.0)
    ranked = torch.cat([head, bulk])
    found = one_change_point(ranked)
    assert found["rank"] is not None
    assert 3 <= int(found["rank"]) <= 6
    assert found["endpoint_only"] is False


def test_endpoint_spike_is_not_a_natural_boundary() -> None:
    ranked = torch.tensor([10.0] + [1.0] * 20)
    found = one_change_point(ranked)
    assert found["endpoint_only"] is True
    assert found["natural_boundary"] is False


def test_participation_mass_is_nonnegative() -> None:
    scores = torch.tensor([-0.2, 0.0, 0.5, 1.0])
    mass = participation_mass(scores)
    assert bool((mass >= 0).all())
    assert float(mass[-1].item()) > float(mass[0].item())


def test_correlations_and_cohens_d() -> None:
    left = torch.arange(10.0)
    assert pearson(left, left) > 0.99
    assert spearman(left, -left) < -0.99
    ranked = torch.linspace(2.0, 0.0, 20)
    assert cohens_d_prefix(ranked, 0.25) > 0.5
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5


def test_adjacent_gaps_match_descending_differences() -> None:
    ranked = torch.tensor([3.0, 2.0, 1.5])
    assert adjacent_gaps(ranked).tolist() == [1.0, 0.5]


def test_build_layer_rows_preserve_ids_and_ranks() -> None:
    payload = {
        "model_family": "toy",
        "layer_scores": [0.5, 1.0, 0.4],
        "table": {
            7: {"expert_structural_scores": torch.tensor([0.4, 0.6])},
            8: {"expert_structural_scores": torch.tensor([1.0, 0.9])},
            9: {"expert_structural_scores": torch.tensor([0.3, 0.3])},
        },
    }
    rows = build_layer_rows(payload)
    assert [row["layer_id"] for row in rows] == [8, 7, 9]
    assert rows[0]["rank"] == 1
    assert rows[0]["score"] == 1.0


def test_quantile_decision_without_natural_gap() -> None:
    layer_seg = {"natural_boundary": False, "within_two": 0.9}
    nulls = {"expert_mean_to_layer": {"excess": 0.9}}
    links: dict[str, object] = {}
    assert decide_allocation(layer_seg, nulls, links) == "use quantile only"
