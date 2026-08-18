from __future__ import annotations

import torch

from PP.build_response_coverage import output_direction_coverage_order, response_coverage_order


def test_response_coverage_keeps_pp_prefix_and_selects_novel_channel() -> None:
    responses = torch.tensor(
        [
            [1.0, 0.99, 0.0, 0.7],
            [0.0, 0.01, 1.0, 0.7],
        ]
    )
    importance_order = torch.tensor([0, 1, 3, 2])
    pseudo_order = torch.tensor([0, 1, 2, 3])

    order = response_coverage_order(
        responses,
        importance_order,
        pseudo_order,
        retained_channels=2,
        protected_channels=1,
    )

    assert order[:2].tolist() == [0, 2]
    assert sorted(order.tolist()) == [0, 1, 2, 3]


def test_response_coverage_uses_importance_to_break_ties() -> None:
    responses = torch.eye(3)
    importance_order = torch.tensor([2, 1, 0])
    pseudo_order = torch.tensor([0, 1, 2])

    order = response_coverage_order(
        responses,
        importance_order,
        pseudo_order,
        retained_channels=1,
        protected_channels=0,
    )

    assert order[0].item() == 2


def test_output_direction_coverage_distinguishes_write_directions() -> None:
    responses = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    down = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    importance_order = torch.tensor([0, 1, 2])
    pseudo_order = torch.tensor([0, 1, 2])

    order = output_direction_coverage_order(
        responses,
        down,
        importance_order,
        pseudo_order,
        retained_channels=2,
        protected_channels=1,
    )

    assert order[:2].tolist() == [0, 1]
    assert sorted(order.tolist()) == [0, 1, 2]