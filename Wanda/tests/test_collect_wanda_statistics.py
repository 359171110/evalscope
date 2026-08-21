from __future__ import annotations

from pathlib import Path

import torch

from Wanda.collect_wanda_statistics import load_causal_or_conditional_model, native_route_from_gate_output


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


def test_load_conditional_generation_uses_image_text_to_text(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    class FakeConfig:
        architectures = ["Qwen3_5MoeForConditionalGeneration"]

    class FakeModel:
        pass

    def fake_config_from_pretrained(path, **kwargs):
        return FakeConfig()

    def fake_image_text_from_pretrained(path, **kwargs):
        calls.append("image_text")
        return FakeModel()

    def fake_causal_from_pretrained(path, **kwargs):
        calls.append("causal")
        return FakeModel()

    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        fake_config_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForImageTextToText.from_pretrained",
        fake_image_text_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        fake_causal_from_pretrained,
    )

    model = load_causal_or_conditional_model(tmp_path, {"trust_remote_code": True})

    assert isinstance(model, FakeModel)
    assert calls == ["image_text"]
