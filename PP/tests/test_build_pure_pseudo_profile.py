from __future__ import annotations

from pathlib import Path

import torch

from PP.build_pure_pseudo_profile import build_pure_pseudo_artifacts, expand_probe_signs, pure_pseudo_priority
from PP.derive_fixed_width_profile import derive_fixed_width_profile
from src.channel_runtime import channel_table_from_payload
from src.static_expert_pruning import validate_static_profile_payload


def test_pure_pseudo_priority_uses_only_pseudo_scores() -> None:
    pseudo_scores = torch.tensor([0.25, 4.0, 1.5, 3.0])

    priority = pure_pseudo_priority(pseudo_scores)

    assert torch.argsort(priority, descending=True).tolist() == [1, 3, 2, 0]


def test_expand_probe_signs_appends_negative_directions() -> None:
    probes = torch.tensor([[1.0, -2.0], [3.0, 4.0]])

    expanded = expand_probe_signs(probes, "positive-negative")

    assert torch.equal(expanded, torch.cat((probes, -probes), dim=0))


def test_pure_pseudo_artifacts_keep_an_exact_fixed_width_per_expert() -> None:
    priorities = {0: torch.tensor([[8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0]]).repeat(2, 1)}

    channel, profile = build_pure_pseudo_artifacts(
        model_path=Path("/models/qwen3"),
        priorities_by_layer=priorities,
        target_pruning_ratio=0.5,
        router_neighbors=1,
        top_q=2,
        probe_signs="positive",
        block_size=2,
        checkpoint_identity={"weight_index_sha256": "test"},
    )

    widths = validate_static_profile_payload(profile)
    table = channel_table_from_payload(channel["table"])

    assert widths.tolist() == [[2, 2]]
    assert profile["total_blocks"] == 4
    assert profile["maximum_blocks"] == 8
    assert profile["actual_structural_pruning_ratio"] == 0.5
    assert profile["profile_construction"] == "calibration_free"
    assert profile["test_metrics_used_for_profile"] is False
    assert profile["method"] == "pure_pseudo"
    assert channel["purpose"] == "pure_pseudo_channel_ranking"
    assert table[0].ranked_indices[0].tolist() == [0, 2, 4, 6, 7, 5, 3, 1]


def test_derive_fixed_width_profile_preserves_ranking_metadata() -> None:
    _, source = build_pure_pseudo_artifacts(
        model_path=Path("/models/qwen3"),
        priorities_by_layer={0: torch.ones(2, 8)},
        target_pruning_ratio=0.5,
        router_neighbors=1,
        top_q=2,
        probe_signs="positive",
        block_size=2,
        checkpoint_identity={"weight_index_sha256": "test"},
    )

    profile = derive_fixed_width_profile(source, retained_blocks=3)
    widths = validate_static_profile_payload(profile)

    assert widths.tolist() == [[3, 3]]
    assert profile["target_pruning_ratio"] == 0.25
    assert profile["total_blocks"] == 6
    assert profile["maximum_blocks"] == 8
    assert profile["pure_pseudo"] == source["pure_pseudo"]
    assert profile["width_sweep"]["pruned_blocks"] == 1