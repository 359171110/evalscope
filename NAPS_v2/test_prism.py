from __future__ import annotations

import json
from pathlib import Path

import torch

from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.prism import (
    RouteNCRConfig,
    build_router_conditioned_probes,
    native_probe_spaces,
    native_router_forward,
    synthetic_channel_score,
)


def _qwen_adapter(tmp_path: Path) -> PurePseudoModelAdapter:
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_moe",
        "hidden_size": 4,
        "hidden_act": "silu",
        "rms_norm_eps": 1.0e-6,
        "moe_intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_experts": 4,
        "num_experts_per_tok": 1,
    }), encoding="utf-8")
    return PurePseudoModelAdapter.from_checkpoint(tmp_path, {
        "model.layers.0.mlp.gate.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.gate_proj.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.up_proj.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.down_proj.weight": "model.safetensors",
    })


def _gemma_adapter(tmp_path: Path) -> PurePseudoModelAdapter:
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 4,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1.0e-6,
            "intermediate_size": 8,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_experts": 4,
            "top_k_experts": 2,
        },
    }), encoding="utf-8")
    return PurePseudoModelAdapter.from_checkpoint(tmp_path, {
        "model.language_model.layers.0.router.proj.weight": "model.safetensors",
        "model.language_model.layers.0.experts.gate_up_proj": "model.safetensors",
        "model.language_model.layers.0.experts.down_proj": "model.safetensors",
    })


def test_native_probe_spaces_use_distinct_gemma_expert_norm(tmp_path: Path) -> None:
    adapter = _gemma_adapter(tmp_path)
    latent = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expert_norm = torch.tensor([1.0, 2.0, 3.0, 4.0])

    route_rows, expert_rows = native_probe_spaces(adapter, latent, None, expert_norm)

    assert torch.allclose(expert_rows, route_rows * expert_norm)
    assert not torch.allclose(route_rows, expert_rows)


def test_gemma_native_router_applies_per_expert_scale(tmp_path: Path) -> None:
    adapter = _gemma_adapter(tmp_path)
    route_rows = torch.eye(4)
    router = torch.eye(4)
    per_expert_scale = torch.tensor([1.0, 2.0, 3.0, 4.0])

    _, selected, weights = native_router_forward(
        adapter,
        route_rows,
        router,
        torch.tensor(2.0),
        per_expert_scale,
    )

    expected_scale = per_expert_scale[selected]
    assert torch.allclose(weights.sum(1), (weights / expected_scale).sum(1) * 0 + weights.sum(1))
    assert torch.all(weights > 0)
    assert torch.allclose(
        weights / expected_scale,
        (weights / expected_scale) / (weights / expected_scale).sum(1, keepdim=True),
    )


def test_router_conditioned_probes_are_complete_and_deterministic(tmp_path: Path) -> None:
    adapter = _qwen_adapter(tmp_path)
    router = torch.eye(4)
    norm = torch.ones(4)
    config = RouteNCRConfig(
        probes_per_expert=3,
        candidates_per_attempt=8,
        max_attempts=4,
    )

    first = build_router_conditioned_probes(adapter, router, norm, None, None, None, config)
    second = build_router_conditioned_probes(adapter, router, norm, None, None, None, config)

    assert first[0].shape == (4, 3, 4)
    assert first[1].shape == (4, 3)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2]["prior_coverage"].all()
    assert torch.all(first[2]["routed_candidate_counts"] >= 3)
    assert torch.equal(first[2]["accepted_counts"], torch.full((4,), 3))


def test_router_conditioned_probes_fail_without_prior_coverage(tmp_path: Path) -> None:
    adapter = _qwen_adapter(tmp_path)
    router = torch.zeros(4, 4)
    norm = torch.ones(4)
    config = RouteNCRConfig(
        probes_per_expert=1,
        candidates_per_attempt=1,
        max_attempts=1,
    )

    try:
        build_router_conditioned_probes(adapter, router, norm, None, None, None, config)
    except RuntimeError as error:
        assert "did not fill every expert budget" in str(error)
    else:
        raise AssertionError("RouteNCR must not fill missing experts with non-prior fallback probes")


def test_synthetic_channel_score_uses_route_weighted_nonlinear_response() -> None:
    probes = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    gate = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    up = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    down = torch.tensor([[2.0, 3.0], [0.0, 0.0]])
    weights = torch.tensor([0.5, 2.0])

    unweighted = synthetic_channel_score(probes, gate, up, down, "silu")
    weighted = synthetic_channel_score(probes, gate, up, down, "silu", weights)
    responses = torch.nn.functional.silu(probes @ gate.transpose(0, 1)) * (probes @ up.transpose(0, 1))
    expected_unweighted = responses.square().mean(0) * down.square().sum(0)
    expected_weighted = (
        (responses * weights.unsqueeze(1)).square().sum(0) / weights.square().sum()
        * down.square().sum(0)
    )

    assert torch.allclose(unweighted, expected_unweighted)
    assert torch.allclose(weighted, expected_weighted)
    assert weighted[1] == 0.0