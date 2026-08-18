from __future__ import annotations

import json
from pathlib import Path

from PP.pure_pseudo_model_adapter import PurePseudoModelAdapter


DATASETS = Path("/data01/datasets")


def test_gemma4_adapter_uses_nested_text_config_and_fused_experts() -> None:
    model_path = DATASETS / "gemma-4-26B-A4B-it"
    index = json.loads(next(model_path.glob("*.index.json")).read_text())
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, index["weight_map"])

    assert adapter.model_family == "gemma4"
    assert (adapter.num_layers, adapter.num_experts, adapter.intermediate_size) == (30, 128, 704)
    assert adapter.router_top_k == 8
    assert adapter.router_name(0).endswith("router.proj.weight")
    assert adapter.expert_gate_up_name(0).endswith("experts.gate_up_proj")
    assert adapter.expert_down_name(0).endswith("experts.down_proj")


def test_qwen36_adapter_uses_nested_text_config_and_fused_experts() -> None:
    model_path = DATASETS / "Qwen3.6-35B-A3B"
    index = json.loads(next(model_path.glob("*.index.json")).read_text())
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, index["weight_map"])

    assert adapter.model_family == "qwen3.6"
    assert (adapter.num_layers, adapter.num_experts, adapter.intermediate_size) == (40, 256, 512)
    assert adapter.router_top_k == 8
    assert adapter.router_name(0).endswith("mlp.gate.weight")
    assert adapter.expert_gate_up_name(0).endswith("mlp.experts.gate_up_proj")
    assert adapter.expert_down_name(0).endswith("mlp.experts.down_proj")
