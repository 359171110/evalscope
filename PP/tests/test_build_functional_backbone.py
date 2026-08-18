from __future__ import annotations

from pathlib import Path

import torch

from PP.build_functional_backbone import (
    build_functional_artifacts,
    mean_functional_energy,
    swiglu_probe_responses,
    weight_functional_moment,
)
from src.channel_runtime import channel_table_from_payload
from src.static_expert_pruning import validate_static_profile_payload


def test_swiglu_probe_responses_and_mfe_match_manual_values() -> None:
    probes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gate = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    up = torch.tensor([[2.0, 0.0], [0.0, 3.0]])

    responses = swiglu_probe_responses(probes, gate, up)
    scores = mean_functional_energy(responses)

    expected = responses.square().mean(dim=0)
    assert torch.allclose(scores, expected)
    assert responses[0, 1] == 0.0
    assert responses[1, 0] == 0.0


def test_weight_functional_moment_matches_explicit_covariance() -> None:
    gate = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
    up = torch.tensor([[3.0, -1.0], [1.0, 4.0]])
    gamma = torch.tensor([2.0, 0.5])
    covariance = torch.diag(gamma.square())
    expected = []
    for gate_row, up_row in zip(gate, up):
        gate_energy = gate_row @ covariance @ gate_row
        up_energy = up_row @ covariance @ up_row
        cross = gate_row @ covariance @ up_row
        expected.append(0.25 * (gate_energy * up_energy + 2.0 * cross.square()))

    scores = weight_functional_moment(gate, up, gamma)

    assert torch.allclose(scores, torch.stack(expected))


def test_functional_artifacts_produce_ranked_fixed_width_payload() -> None:
    priorities = {0: torch.tensor([[8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0]])}

    channel, profile = build_functional_artifacts(
        model_path=Path("/models/qwen3"),
        priorities_by_layer=priorities,
        importance="mfe",
        target_pruning_ratio=0.5,
        router_neighbors=8,
        block_size=2,
        checkpoint_identity={"weight_index_sha256": "test"},
    )

    widths = validate_static_profile_payload(profile)
    table = channel_table_from_payload(channel["table"])
    assert widths.tolist() == [[2]]
    assert table[0].ranked_indices[0].tolist() == [0, 2, 4, 6, 7, 5, 3, 1]
    assert profile["functional_backbone"]["importance"] == "mfe"
    assert profile["test_metrics_used_for_profile"] is False