from __future__ import annotations

import torch

from AIMER_MIX_PLUS.build_pseudo_source_artifacts import (
    activation_response,
    aggregate_channel_scores,
    select_previous_write_probes,
)


def test_native_activation_dispatch_changes_gemma_response() -> None:
    probes = torch.tensor([[1.0, -0.5], [-0.25, 0.75]])
    gate = torch.tensor([[1.0, 0.5], [-0.5, 1.0]])
    up = torch.tensor([[0.5, 1.0], [1.0, -0.5]])
    silu = activation_response(probes, gate, up, "silu")
    gelu = activation_response(probes, gate, up, "gelu_pytorch_tanh")
    assert not torch.allclose(silu, gelu)


def test_output_score_includes_down_column_norm() -> None:
    probes = torch.ones(2, 2)
    gate = torch.ones(2, 2)
    up = torch.ones(2, 2)
    down = torch.tensor([[1.0, 10.0], [0.0, 0.0]])
    activation = aggregate_channel_scores(
        probes,
        gate,
        up,
        down,
        activation="silu",
        top_q=1,
        score_mode="activation",
    )
    output = aggregate_channel_scores(
        probes,
        gate,
        up,
        down,
        activation="silu",
        top_q=1,
        score_mode="output",
    )
    assert torch.allclose(activation[0], activation[1])
    assert output[1] > output[0]


def test_previous_write_selection_orients_negative_directions() -> None:
    router = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    chunks = [torch.tensor([[-1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])]
    probes, affinities = select_previous_write_probes(
        router,
        chunks,
        previous_gamma=torch.ones(2),
        probe_count=1,
    )
    assert torch.allclose(probes[0, 0], torch.tensor([1.0, 0.0]))
    assert float(affinities[0, 0].item()) == 1.0