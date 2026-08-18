from __future__ import annotations

import torch

from ramp_statistics import RoutedExpertCovarianceAccumulator


class _Expert(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(2, 2, bias=False)
        self.up_proj = torch.nn.Linear(2, 2, bias=False)
        self.down_proj = torch.nn.Linear(2, 2, bias=False)
        self.act_fn = torch.nn.Identity()
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.eye(2))
            self.up_proj.weight.copy_(torch.eye(2))
            self.down_proj.weight.copy_(torch.eye(2))


def test_accumulator_matches_gate_weighted_covariance_by_routed_slot() -> None:
    accumulator = RoutedExpertCovarianceAccumulator({0: (1,)}, accumulation_dtype=torch.float64)
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    selected_experts = torch.tensor([[1, 0], [1, 0]])
    routing_weights = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
    experts = torch.nn.ModuleList([_Expert(), _Expert()])

    accumulator.update(0, hidden_states, experts, selected_experts, routing_weights)
    payload = accumulator.to_payload()
    stats = payload[0][1]

    assert stats["route_count"] == 2
    assert torch.allclose(stats["gate_square_sum"], torch.tensor(0.3125, dtype=torch.float64))
    assert torch.allclose(stats["gate_fourth_sum"], torch.tensor(0.06640625, dtype=torch.float64))
    assert torch.allclose(
        stats["covariance"],
        torch.tensor([[5.3125, 10.0], [10.0, 20.0]], dtype=torch.float64),
    )
    assert torch.allclose(stats["unweighted_square_sum"], torch.tensor([82.0, 272.0], dtype=torch.float64))
    assert torch.allclose(stats["max_abs"], torch.tensor([9.0, 16.0], dtype=torch.float64))
    assert torch.equal(stats["down_proj"], torch.eye(2))