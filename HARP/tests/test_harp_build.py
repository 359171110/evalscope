from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from CSP.export_csp_checkpoint import main as export_main
from CSP.tests.helpers import write_checkpoint
from HARP.build_harp_artifacts import build_profile


def test_harp_build_and_export_preserves_exact_layer_budget(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "qwen3-model"
    write_checkpoint(model, "qwen3", large=True)
    artifact = tmp_path / "harp-artifact"
    channel = artifact / "channel.pt"
    profile = artifact / "profile.pt"
    build_profile(model, channel, profile, budget_width=128, low_width=64, high_width=192)
    payload = torch.load(profile, map_location="cpu", weights_only=True)
    widths = payload["profile_widths"] * payload["channel_block_size"]
    assert payload["method"] == "harp"
    assert payload["allocation_scope"] == "per_layer_expert_harp_layer_expert_channel_sp"
    assert widths.sum(dim=1).tolist() == [payload["num_experts"] * 128] * payload["num_layers"]
    assert bool(((widths >= 64) & (widths <= 192)).all())

    output = tmp_path / "harp-output"
    monkeypatch.setattr("sys.argv", [
        "export_harp_checkpoint", "--model-path", str(model), "--profile", str(profile),
        "--channel-cache", str(channel), "--output-dir", str(output),
    ])
    assert export_main() == 0
    manifest = json.loads((output / "pruning_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "harp"
    assert manifest["allocation_scope"] == payload["allocation_scope"]
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
        assert handle.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight").shape[0] == 192
