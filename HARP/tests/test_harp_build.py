from __future__ import annotations

import json
from pathlib import Path

import pytest
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
    assert set(payload["csp"]["harp"]["layer_order_descending"]) == set(payload["layer_ids"])

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


@pytest.mark.parametrize(
    ("family", "low_width", "budget", "high_width"),
    [
        ("qwen3", 320, 384, 448),
        ("qwen3.6", 192, 256, 320),
        ("gemma4", 288, 352, 416),
        ("deepseek", 640, 704, 768),
        ("olmoe", 448, 512, 576),
    ],
)
def test_harp_profile_supports_eval_families(
    tmp_path: Path,
    family: str,
    low_width: int,
    budget: int,
    high_width: int,
) -> None:
    model = tmp_path / f"{family}-harp-model"
    write_checkpoint(model, family, large=True)
    artifact = tmp_path / f"{family}-harp-artifact"
    channel = artifact / "channel.pt"
    profile = artifact / "profile.pt"
    build_profile(model, channel, profile, budget_width=budget, low_width=low_width, high_width=high_width)
    payload = torch.load(profile, map_location="cpu", weights_only=True)
    logical = payload["profile_widths"] * payload["channel_block_size"]
    assert payload["method"] == "harp"
    assert payload["padded_intermediate_size"] == high_width
    assert logical.sum().item() == payload["num_layers"] * payload["num_experts"] * budget
    assert bool(((logical >= low_width) & (logical <= high_width)).all())
    assert set(payload["csp"]["harp"]["layer_order_descending"]) == set(payload["layer_ids"])
    second = artifact / "profile_25.pt"
    build_profile(model, channel, second, budget_width=budget, low_width=low_width, high_width=high_width)
    again = torch.load(second, map_location="cpu", weights_only=True)
    assert again["profile_sha256"] == payload["profile_sha256"]


def test_harp_combo_profile_uses_direct_tiers(tmp_path: Path) -> None:
    model = tmp_path / "gemma4-combo-model"
    write_checkpoint(model, "gemma4", large=True)
    artifact = tmp_path / "gemma4-combo-artifact"
    channel = artifact / "channel.pt"
    profile = artifact / "profile.pt"
    build_profile(
        model, channel, profile, budget_width=352, low_width=288, high_width=416,
        allocator="combo", gamma=2.0, min_fraction=0.15,
    )
    payload = torch.load(profile, map_location="cpu", weights_only=True)
    logical = payload["profile_widths"] * payload["channel_block_size"]
    unique = set(int(x) for x in logical.reshape(-1).tolist())
    assert payload["csp"]["harp"]["allocator"] == "combo"
    assert unique <= {288, 352, 416}
    assert unique == {288, 352, 416} or len(unique) >= 2
    means = logical.float().mean(dim=1)
    assert float(means.max()) >= float(means.min())


def test_harp_combo_two_profile_uses_low_and_high_only(tmp_path: Path) -> None:
    model = tmp_path / "gemma4-combo-two-model"
    write_checkpoint(model, "gemma4", large=True)
    artifact = tmp_path / "gemma4-combo-two-artifact"
    channel = artifact / "channel.pt"
    profile = artifact / "profile.pt"
    build_profile(
        model, channel, profile, budget_width=352, low_width=288, high_width=416,
        allocator="combo_two", gamma=2.0, min_fraction=0.15,
    )
    payload = torch.load(profile, map_location="cpu", weights_only=True)
    logical = payload["profile_widths"] * payload["channel_block_size"]
    unique = set(int(x) for x in logical.reshape(-1).tolist())
    assert payload["csp"]["harp"]["allocator"] == "combo_two"
    assert payload["width_options"] == [288, 416]
    assert unique <= {288, 416}
    assert 352 not in unique


def test_harp_quantile_layer_profile_uses_rank_split(tmp_path: Path) -> None:
    model = tmp_path / "gemma4-quantile-model"
    write_checkpoint(model, "gemma4", large=True)
    artifact = tmp_path / "gemma4-quantile-artifact"
    channel = artifact / "channel.pt"
    profile = artifact / "profile.pt"
    build_profile(
        model, channel, profile, budget_width=352, low_width=288, high_width=416,
        allocator="quantile_layer", min_fraction=0.15,
    )
    payload = torch.load(profile, map_location="cpu", weights_only=True)
    logical = payload["profile_widths"] * payload["channel_block_size"]
    unique = set(int(x) for x in logical.reshape(-1).tolist())
    harp = payload["csp"]["harp"]
    assert harp["allocator"] == "quantile_layer"
    assert payload["mode"] == "harp_quantile_layer_expert_channel_sp"
    assert payload["width_options"] == [288, 416]
    assert unique <= {288, 416}
    assert 352 not in unique
    assert int(logical.sum().item()) == payload["num_layers"] * payload["num_experts"] * 352
    assert "layer_high_counts" in harp
    assert int(sum(harp["layer_high_counts"])) * (416 - 288) + payload["num_layers"] * payload["num_experts"] * 288 == (
        payload["num_layers"] * payload["num_experts"] * 352
    )


