from __future__ import annotations

import pytest
import torch

from src.aimer_selector import build_aimer_keep_table_for_model
from src.amp_proxy import build_amp_table_for_model
from src.expert_priors import build_prior_payload


class _FusedExperts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 8, 3))
        self.down_proj = torch.nn.Parameter(torch.randn(2, 3, 4))


class _FusedMoeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = torch.nn.Module()
        layer.input_layernorm = torch.nn.LayerNorm(3)
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = _FusedExperts()
        layer.mlp.gate = torch.nn.Linear(3, 2, bias=False)
        layer.mlp.top_k = 1
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer])


def test_build_prior_payload_preserves_layer_expert_shape() -> None:
    payload = build_prior_payload(
        method="top_p_method1",
        model_path="/models/qwen2-moe",
        table={0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])},
    )

    assert payload["method"] == "top_p_method1"
    assert payload["model_path"] == "/models/qwen2-moe"
    assert payload["num_layers"] == 2
    assert payload["num_experts"] == 2
    assert set(payload["table"]) == {0, 1}


def test_build_prior_payload_rejects_inconsistent_expert_count() -> None:
    with pytest.raises(ValueError, match="expert count"):
        build_prior_payload(
            method="top_p_aimer",
            model_path="/models/qwen2-moe",
            table={0: torch.ones(2), 1: torch.ones(3)},
        )


def test_prior_builders_support_fused_expert_container_without_len() -> None:
    model = _FusedMoeModel()

    amp = build_amp_table_for_model(model)
    aimer = build_aimer_keep_table_for_model(model)

    assert amp[0].shape == (2,)
    assert aimer[0].shape == (2,)
    assert torch.isfinite(amp[0]).all()
    assert torch.isfinite(aimer[0]).all()
