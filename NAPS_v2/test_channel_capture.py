from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from NAPS_v2.capture_routed_tokens import (
    Gemma4Capture,
    RoutedTokenAccumulator,
    validate_weights_only_payload,
    weights_only_safe,
)
from NAPS_v2.model_adapter import (
    BranchTopology,
    ChannelArchitecture,
    ExpertInputSemantics,
    ExpertTensorCodec,
)


class _Experts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.tensor([
            [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 3.0], [3.0, 0.0]],
        ]))
        self.act_fn = torch.nn.Identity()

    def forward(
        self,
        expert_inputs: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return expert_inputs


class _GemmaLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _Experts()


def test_accumulator_pairs_token_rows_with_exact_route_slots() -> None:
    accumulator = RoutedTokenAccumulator([0], num_experts=2, hidden_size=2, limit=2)
    expert_inputs = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    top_k_index = torch.tensor([[1, 0], [0, 1], [1, 0]])
    top_k_weights = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])

    accumulator.add(0, expert_inputs, top_k_index, top_k_weights)
    experts = accumulator.payload()["layers"][0]

    assert torch.equal(experts[0]["inputs"], expert_inputs[:2])
    assert experts[0]["inputs"].dtype is torch.bfloat16
    assert torch.allclose(experts[0]["route_weights"], torch.tensor([0.9, 0.8]))
    assert experts[0]["captured_token_count"] == 2
    assert experts[0]["captured_route_mass"] == pytest.approx(1.7)
    assert experts[0]["total_route_count"] == 3
    assert experts[0]["total_route_mass"] == pytest.approx(2.4)
    assert torch.equal(experts[1]["inputs"], expert_inputs[:2])
    assert torch.allclose(experts[1]["route_weights"], torch.tensor([0.1, 0.2]))


def test_gemma_capture_reads_native_expert_call_tuple() -> None:
    layer = _GemmaLayer()
    accumulator = RoutedTokenAccumulator(
        [0], num_experts=2, hidden_size=2, limit=4, intermediate_size=2
    )
    capture = Gemma4Capture({0: layer}, accumulator)
    expert_inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    top_k_index = torch.tensor([[0, 1], [1, 0]])
    top_k_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]])

    layer.experts(expert_inputs, top_k_index, top_k_weights)
    capture.close()
    experts = accumulator.payload()["layers"][0]
    response_energy = accumulator.payload()["route_weighted_response_energy"]

    assert torch.equal(experts[0]["inputs"], expert_inputs)
    assert experts[0]["inputs"].dtype is torch.bfloat16
    assert torch.allclose(experts[0]["route_weights"], torch.tensor([0.75, 0.4]))
    assert torch.equal(experts[1]["inputs"], expert_inputs)
    assert torch.allclose(experts[1]["route_weights"], torch.tensor([0.25, 0.6]))
    expected_expert_0 = torch.tensor([
        (0.75 * 2.0) ** 2 + (0.4 * 18.0) ** 2,
        (0.75 * 8.0) ** 2 + (0.4 * 32.0) ** 2,
    ])
    expected_expert_1 = torch.tensor([
        (0.25 * 12.0) ** 2 + (0.6 * 48.0) ** 2,
        (0.25 * 3.0) ** 2 + (0.6 * 27.0) ** 2,
    ])
    assert torch.allclose(response_energy[0, 0], expected_expert_0)
    assert torch.allclose(response_energy[0, 1], expected_expert_1)


def test_capture_metadata_is_weights_only_safe() -> None:
    architecture = ChannelArchitecture(
        model_family="gemma4",
        hidden_size=2,
        source_intermediate_size=4,
        num_layers=1,
        num_experts=2,
        router_top_k=1,
        activation="gelu_pytorch_tanh",
        expert_tensor_codec=ExpertTensorCodec.PACKED,
        branch_topology=BranchTopology.DENSE_PLUS_SPARSE,
        expert_input_semantics=ExpertInputSemantics.INDEPENDENT_FROM_RAW_RESIDUAL,
        channel_alignment=1,
        dense_intermediate_size=4,
        shared_expert_intermediate_size=None,
        router_has_global_scale=True,
        router_has_per_expert_scale=True,
    )
    payload = {"architecture": weights_only_safe(asdict(architecture))}

    validate_weights_only_payload(payload)

    assert payload["architecture"]["expert_tensor_codec"] == "packed"
    assert payload["architecture"]["branch_topology"] == "dense_plus_sparse"