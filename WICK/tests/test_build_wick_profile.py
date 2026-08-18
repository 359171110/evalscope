from __future__ import annotations

from pathlib import Path

import torch

from WICK.build_wick_profile import (
    build_wick_artifacts,
    combine_wick_priority,
    pseudo_protection_importance,
    router_gram_neighbors,
    weight_path_importance,
)
from src.static_expert_pruning import validate_static_profile_payload


def test_weight_path_importance_uses_all_three_swiglu_projections() -> None:
    gate = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    up = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    down = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    scores = weight_path_importance(gate, up, down)

    assert scores[1] > scores[0]


def test_router_gram_neighbors_keeps_self_and_selects_closest_direction() -> None:
    router = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])

    neighbors = router_gram_neighbors(router, neighbor_count=1)

    assert neighbors.tolist() == [[0, 1], [1, 0], [2, 1]]


def test_pseudo_protection_overrides_weight_only_pruning_order() -> None:
    priority, protected = combine_wick_priority(
        torch.tensor([1.0, 4.0, 3.0, 2.0]),
        torch.tensor([100.0, 0.0, 0.0, 0.0]),
        protection_ratio=0.25,
        retained_channels=2,
    )

    assert protected.tolist() == [True, False, False, False]
    assert torch.argsort(priority, descending=True).tolist()[:2] == [0, 1]


def test_pseudo_protection_importance_uses_strongest_probe_contributions() -> None:
    probes = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    gate = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    up = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    down = torch.eye(2)

    scores = pseudo_protection_importance(probes, gate, up, down, top_q=1)

    assert scores[0] > 0.0
    assert scores[1] == 0.0


def test_pseudo_protection_importance_can_disable_down_proj_norm() -> None:
    probes = torch.tensor([[1.0, 0.0]])
    gate = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    up = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    down = torch.tensor([[1.0, 3.0], [0.0, 4.0]])

    weighted = pseudo_protection_importance(probes, gate, up, down, top_q=1)
    unweighted = pseudo_protection_importance(
        probes,
        gate,
        up,
        down,
        top_q=1,
        use_down_proj_norm=False,
    )

    assert torch.isclose(weighted[1], weighted[0] * 5.0)
    assert torch.isclose(unweighted[0], unweighted[1])


def test_wick_artifacts_keep_an_exact_fixed_width_per_expert() -> None:
    priorities = {0: torch.arange(8, 0, -1, dtype=torch.float32).repeat(2, 1)}
    protected = {0: torch.tensor([[True] + [False] * 7, [False, True] + [False] * 6])}

    channel, profile = build_wick_artifacts(
        model_path=Path("/models/qwen3"),
        priorities_by_layer=priorities,
        protected_by_layer=protected,
        target_pruning_ratio=0.5,
        protection_ratio=0.125,
        router_neighbors=1,
        top_q=2,
        block_size=2,
        checkpoint_identity={"weight_index_sha256": "test"},
    )

    widths = validate_static_profile_payload(profile)
    assert widths.tolist() == [[2, 2]]
    assert profile["total_blocks"] == 4
    assert profile["maximum_blocks"] == 8
    assert profile["actual_structural_pruning_ratio"] == 0.5
    assert channel["purpose"] == "wick_gram_protected_channel_ranking"
    assert channel["wick"]["protected_counts"].tolist() == [[1, 1]]