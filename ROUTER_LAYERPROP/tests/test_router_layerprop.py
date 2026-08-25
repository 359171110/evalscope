"""Focused tests for the data-free Router LayerProp implementation."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from ROUTER_LAYERPROP.adapters import Gemma4MoeAdapter, Qwen35MoeAdapter, Qwen3MoeAdapter, adapter_for_model
from ROUTER_LAYERPROP.config import LayerPropConfig
from ROUTER_LAYERPROP.core import (
    build_router_probes,
    fit_ridge_down,
    output_energy_scores,
    recoverability_swap_refinement,
)
from ROUTER_LAYERPROP.planner import build_expert_plan, build_layer_plan
from ROUTER_LAYERPROP.propagation import (
    pack_refresh_probes,
    refresh_source_targets,
    run_refresh_propagation,
)


class FakeRMSNorm(nn.Module):
    def __init__(self, hidden: int, *, add_one: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden) * (0.2 if add_one else 1.0))
        self.eps = 1.0e-6
        self.add_one = add_one

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gamma = self.weight + 1.0 if self.add_one else self.weight
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps) * gamma


def _separate_expert(hidden: int, channels: int) -> nn.Module:
    expert = nn.Module()
    expert.gate_proj = nn.Linear(hidden, channels, bias=False)
    expert.up_proj = nn.Linear(hidden, channels, bias=False)
    expert.down_proj = nn.Linear(channels, hidden, bias=False)
    return expert


def _packed_experts(experts: int, hidden: int, channels: int) -> nn.Module:
    module = nn.Module()
    module.gate_up_proj = nn.Parameter(torch.randn(experts, 2 * channels, hidden))
    module.down_proj = nn.Parameter(torch.randn(experts, hidden, channels))
    return module


def _fake_qwen3() -> nn.Module:
    hidden, channels, experts = 6, 5, 4
    layer = nn.Module()
    layer.post_attention_layernorm = FakeRMSNorm(hidden)
    layer.mlp = nn.Module()
    layer.mlp.gate = nn.Linear(hidden, experts, bias=False)
    layer.mlp.gate.norm_topk_prob = True
    layer.mlp.gate.routed_scaling_factor = 1.0
    layer.mlp.experts = nn.ModuleList([_separate_expert(hidden, channels) for _ in range(experts)])
    root = nn.Module()
    root.layers = nn.ModuleList([layer])
    model = nn.Module()
    model.model = root
    model.embed_tokens = nn.Embedding(16, hidden)
    model.config = SimpleNamespace(model_type="qwen3_moe", num_experts_per_tok=2)
    model.get_input_embeddings = lambda: model.embed_tokens
    return model


def _fake_qwen35() -> nn.Module:
    hidden, channels, experts = 6, 5, 4
    layer = nn.Module()
    layer.post_attention_layernorm = FakeRMSNorm(hidden, add_one=True)
    layer.mlp = nn.Module()
    layer.mlp.gate = nn.Linear(hidden, experts, bias=False)
    layer.mlp.gate.norm_topk_prob = True
    layer.mlp.gate.routed_scaling_factor = 1.0
    layer.mlp.experts = _packed_experts(experts, hidden, channels)
    root = nn.Module()
    root.layers = nn.ModuleList([layer])
    model = nn.Module()
    model.model = root
    model.embed_tokens = nn.Embedding(16, hidden)
    model.config = SimpleNamespace(model_type="qwen3_5_moe_text", num_experts_per_tok=2)
    model.get_input_embeddings = lambda: model.embed_tokens
    return model


def _fake_gemma4() -> nn.Module:
    hidden, channels, experts = 6, 5, 4
    layer = nn.Module()
    layer.pre_feedforward_layernorm_2 = FakeRMSNorm(hidden)
    layer.router = nn.Module()
    layer.router.proj = nn.Linear(hidden, experts, bias=False)
    layer.router.scale = nn.Parameter(torch.ones(hidden))

    def route(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = layer.router.proj(value * layer.router.scale)
        weights, indices = torch.topk(torch.sigmoid(logits), 2, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights, indices

    layer.router.forward = route
    layer.experts = _packed_experts(experts, hidden, channels)
    root = nn.Module()
    root.layers = nn.ModuleList([layer])
    model = nn.Module()
    model.model = root
    model.embed_tokens = nn.Embedding(16, hidden)
    model.config = SimpleNamespace(model_type="gemma4_text", top_k_experts=2)
    model.get_input_embeddings = lambda: model.embed_tokens
    return model


def test_core_selection_and_compensation_shapes() -> None:
    torch.manual_seed(11)
    activation = torch.randn(32, 6)
    down = torch.randn(4, 6)
    scores = output_energy_scores(activation, down)
    keep = torch.topk(scores, 4).indices
    refined = recoverability_swap_refinement(activation, down, keep, max_swaps=1)
    result = fit_ridge_down(activation[:16], activation[16:], down, refined)
    assert refined.shape == (4,)
    assert result.down.shape == (4, 4)


def test_probe_variants_never_use_self_partner() -> None:
    directions = torch.eye(4)
    probes = build_router_probes(directions, variants=3)
    for expert in range(4):
        assert torch.allclose(probes[expert, 0], probes[expert, 0])
        assert torch.isfinite(probes[expert]).all()


def test_refresh_schedule_uses_nearest_source_within_horizon() -> None:
    schedule = refresh_source_targets(tuple(range(13)), stride=4, horizon=8)
    assert schedule == {
        4: (4, 5, 6, 7),
        8: (8, 9, 10, 11),
        12: (12,),
    }


def test_refresh_probe_pack_uses_fixed_interleaved_budget() -> None:
    adapter = adapter_for_model(_fake_qwen3())
    config = LayerPropConfig(
        num_pseudo_tokens=16,
        sequence_length=4,
        probe_variants=2,
        channel_multiple=2,
    )
    bank = pack_refresh_probes(adapter, adapter.layers()[0], config, scale=1.0)
    assert bank.shape == (4, 4, 6)
    assert torch.isfinite(bank).all()


def test_refresh_propagation_runs_each_origin_once() -> None:
    adapter = adapter_for_model(_fake_qwen3())
    adapter.model.model.layers = nn.ModuleList([adapter.layers()[0] for _ in range(6)])
    calls = []

    def run_window(
        bank: torch.Tensor,
        source_layer_id: int,
        target_layer_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        calls.append((source_layer_id, target_layer_ids))
        return bank, {layer_id: bank + float(layer_id) for layer_id in target_layer_ids}

    adapter.run_refresh_window = run_window
    config = LayerPropConfig(
        num_pseudo_tokens=16,
        sequence_length=4,
        probe_variants=2,
        refresh_stride=2,
        refresh_horizon=2,
        max_rows_per_expert_per_origin=8,
        channel_multiple=2,
    )
    rows = run_refresh_propagation(
        adapter,
        tuple(range(6)),
        config,
        scales={layer_id: 1.0 for layer_id in range(6)},
        fallback_scale=1.0,
    )
    assert calls == [(2, (2, 3)), (4, (4, 5))]
    assert set(rows) == {"refresh_2", "refresh_4"}
    assert set(rows["refresh_2"]) == {2, 3}
    assert set(rows["refresh_4"]) == {4, 5}


def test_qwen3_adapter_uses_raw_residual_norm_coordinate() -> None:
    model = _fake_qwen3()
    adapter = adapter_for_model(model)
    assert isinstance(adapter, Qwen3MoeAdapter)
    layer = adapter.layers()[0]
    raw = torch.randn(3, 6)
    indices, weights = adapter.native_route(layer, raw)
    assert indices.shape == (3, 2)
    assert weights.shape == (3, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1.0e-5)


def test_qwen35_and_gemma4_adapters_are_selectable() -> None:
    qwen35 = adapter_for_model(_fake_qwen35())
    gemma4 = adapter_for_model(_fake_gemma4())
    assert isinstance(qwen35, Qwen35MoeAdapter)
    assert isinstance(gemma4, Gemma4MoeAdapter)
    for adapter in (qwen35, gemma4):
        layer = adapter.layers()[0]
        probes = adapter.router_probe_bank(layer, variants=2, scale=1.0)
        assert probes.shape == (4, 2, 6)
        rows = adapter.local_rows(layer, variants=2, scale=1.0, max_rows_per_expert=8)
        assert set(rows) == {0, 1, 2, 3}


def test_plan_falls_back_for_uncovered_expert() -> None:
    config = LayerPropConfig(min_train_rows=4, min_valid_rows=2, channel_multiple=2)
    config.validate()
    down = torch.randn(4, 6)
    plan = build_expert_plan(
        source_rows={},
        down_proj=down,
        retained_channels=2,
        config=config,
    )
    assert plan["retained_width"] == 2
    assert plan["compensation_accepted"] is False


def test_local_rows_only_fill_insufficient_long_coverage() -> None:
    config = LayerPropConfig(min_train_rows=3, min_valid_rows=2, channel_multiple=2)
    down = torch.randn(2, 4, 6)
    long = {0: torch.randn(4, 6), 1: torch.randn(1, 6)}
    local = {0: torch.randn(4, 6), 1: torch.randn(4, 6)}
    valid = {0: torch.randn(3, 6), 1: torch.randn(3, 6)}
    plans = build_layer_plan(
        source_rows={"source0_long": long},
        validation_source_rows={"source0_long": valid, "target_local": local},
        fallback_source_rows={"target_local": local},
        down_proj=down,
        retained_channels=2,
        config=config,
    )
    assert plans[0]["train_source"] == "source0_long"
    assert plans[1]["train_source"] == "source0_long+target_local"
