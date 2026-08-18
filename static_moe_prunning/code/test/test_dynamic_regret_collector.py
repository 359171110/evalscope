from __future__ import annotations

import torch

from scripts.collect_dynamic_regret_teacher import DynamicRegretAccumulator


def test_output_collection_accumulates_gate_weighted_co_route_context() -> None:
    accumulator = DynamicRegretAccumulator()
    selected = torch.tensor([[0, 2], [1, 2]])
    weights = torch.tensor([[0.75, 0.25], [0.60, 0.40]])
    outputs = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 2.0]],
            [[0.0, 1.0], [6.0, 8.0]],
        ]
    )

    accumulator.update_output_saliency(
        4,
        selected,
        weights,
        outputs,
        num_experts=3,
    )

    expected_context = torch.tensor(
        [
            [0.5625, 0.0, 0.1875],
            [0.0, 0.36, 0.24],
            [0.1875, 0.24, 0.2225],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(accumulator.co_route_context_sums[4], expected_context)
    assert torch.allclose(
        accumulator.output_saliency_sums[4],
        torch.tensor([3.75, 0.6, 4.5], dtype=torch.float64),
    )
