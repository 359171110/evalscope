from __future__ import annotations

from pathlib import Path

import pytest

from CSP.model_adapter import CSPModelAdapter
from CSP.tests.helpers import write_checkpoint


@pytest.mark.parametrize(
    ("family", "expected_family", "codec", "first_layer", "alignment"),
    [
        ("qwen3", "qwen3", "separate", 0, 64),
        ("gemma4", "gemma4", "packed", 0, 32),
        ("qwen3.6", "qwen3.6", "packed", 0, 64),
        ("deepseek", "deepseek_v2", "separate", 1, 32),
        ("olmoe", "olmoe", "separate", 0, 64),
        ("mixtral", "mixtral", "separate", 0, 64),
    ],
)
def test_adapter_supports_requested_model_families(
    tmp_path: Path,
    family: str,
    expected_family: str,
    codec: str,
    first_layer: int,
    alignment: int,
) -> None:
    model_path = tmp_path / family
    tensors = write_checkpoint(model_path, family)
    adapter = CSPModelAdapter.from_checkpoint(model_path, {name: "model.safetensors" for name in tensors})
    assert adapter.architecture.model_family == expected_family
    assert adapter.architecture.tensor_codec == codec
    assert adapter.architecture.moe_layer_ids()[0] == first_layer
    assert adapter.architecture.channel_alignment == alignment


def test_only_gemma4_declares_architecture_aware_input_scale(tmp_path: Path) -> None:
    for family in ("qwen3", "qwen3.6", "deepseek", "olmoe", "mixtral"):
        model_path = tmp_path / family
        tensors = write_checkpoint(model_path, family)
        adapter = CSPModelAdapter.from_checkpoint(model_path, {name: "model.safetensors" for name in tensors})
        assert adapter.input_scale_name(adapter.architecture.moe_layer_ids()[0]) is None
    model_path = tmp_path / "gemma4"
    tensors = write_checkpoint(model_path, "gemma4")
    adapter = CSPModelAdapter.from_checkpoint(model_path, {name: "model.safetensors" for name in tensors})
    assert adapter.input_scale_name(0) == "model.language_model.layers.0.pre_feedforward_layernorm_2.weight"
