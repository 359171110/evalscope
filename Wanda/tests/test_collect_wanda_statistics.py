from __future__ import annotations

import torch

from Wanda.collect_wanda_statistics import native_route_from_gate_output


def test_native_route_accepts_topk_router_tuple() -> None:
    indices = torch.tensor([[1, 0]])
    weights = torch.tensor([[0.7, 0.3]])
    output = (torch.zeros(1, 4), weights, indices)

    parsed_indices, parsed_weights = native_route_from_gate_output(
        output,
        top_k=2,
        norm_topk_prob=True,
        weight_dtype=torch.float32,
    )

    assert torch.equal(parsed_indices, indices)
    assert torch.equal(parsed_weights, weights)


def test_native_route_reconstructs_topk_from_linear_gate_logits() -> None:
    logits = torch.tensor([[0.0, 4.0, 1.0, 3.0]])

    indices, weights = native_route_from_gate_output(
        logits,
        top_k=2,
        norm_topk_prob=True,
        weight_dtype=torch.float32,
    )

    assert torch.equal(indices, torch.tensor([[1, 3]]))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))
    assert weights[0, 0] > weights[0, 1]
