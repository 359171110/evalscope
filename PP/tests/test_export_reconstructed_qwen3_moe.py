from __future__ import annotations

import torch

from PP.export_reconstructed_qwen3_moe import reconstruct_down_proj, swiglu_responses


def test_swiglu_responses_match_manual_computation() -> None:
    probes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gate = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    up = torch.tensor([[2.0, 0.0], [0.0, 3.0]])

    responses = swiglu_responses(probes, gate, up)

    expected = torch.nn.functional.silu(probes @ gate.transpose(0, 1)) * (probes @ up.transpose(0, 1))
    assert torch.allclose(responses, expected)


def test_dual_reconstruction_recovers_pruned_output_without_changing_mask() -> None:
    responses = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    down = torch.tensor([[1.0, 3.0], [2.0, -1.0]])
    retained = torch.tensor([0])

    effective, diagnostics = reconstruct_down_proj(
        responses,
        down,
        retained,
        ridge_relative=0.0,
    )

    original_output = responses @ down.transpose(0, 1)
    reconstructed_output = responses[:, retained] @ effective.transpose(0, 1)
    assert effective.shape == (2, 1)
    assert torch.allclose(reconstructed_output, original_output, atol=1.0e-5)
    assert diagnostics["error_after"] < 1.0e-6
    assert diagnostics["recovery_ratio"] > 0.99999


def test_ridge_scale_uses_retained_response_gram_trace() -> None:
    responses = torch.tensor([[1.0, 2.0, 0.0], [2.0, 1.0, 1.0]])
    down = torch.eye(3)
    retained = torch.tensor([0, 2])

    _, diagnostics = reconstruct_down_proj(
        responses,
        down,
        retained,
        ridge_relative=1.0e-4,
    )

    responses_keep = responses[:, retained]
    expected = 1.0e-4 * float(responses_keep.square().sum().item()) / responses.shape[0]
    assert diagnostics["regularization"] == expected