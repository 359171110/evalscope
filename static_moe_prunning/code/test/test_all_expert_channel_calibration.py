from __future__ import annotations

import torch

from scripts.calibrate_hessian_channels import DownInputHessianAccumulator


class FakeExpert(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(2, 2, bias=False)
        self.up_proj = torch.nn.Linear(2, 2, bias=False)
        self.down_proj = torch.nn.Linear(2, 2, bias=False)
        self.act_fn = torch.nn.SiLU()


def test_all_expert_statistics_preserve_actual_route_counts() -> None:
    accumulator = DownInputHessianAccumulator()
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    experts = torch.nn.ModuleList([FakeExpert(), FakeExpert(), FakeExpert()])
    all_experts = torch.tensor([[0, 1, 2], [0, 1, 2]])
    routed_experts = torch.tensor([[0], [1]])

    accumulator.update(
        0,
        hidden_states,
        experts,
        all_experts,
        route_selected_experts=routed_experts,
    )

    assert accumulator.counts[0].tolist() == [2, 2, 2]
    assert accumulator.route_counts[0].tolist() == [1, 1, 0]
    assert all(value.shape == (3, 2) for value in (
        accumulator.square_sums[0],
        accumulator.abs_sums[0],
        accumulator.max_abs[0],
    ))