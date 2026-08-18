from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.export_uniform_enp_qwen3_moe import file_sha256, main as export_enp, validate_enp_artifacts


def _artifacts(tmp_path: Path, widths: torch.Tensor, zero_token_policy: str = "prune_uniform") -> tuple[dict, dict, Path]:
    channel_path = tmp_path / "channels.pt"
    channel_path.write_bytes(b"frozen-enp-channel-cache")
    channel_cache = {
        "purpose": "enp_tenp_signed_projection_channel_ranking",
        "split": "train",
        "test_metrics_used": False,
    }
    profile = {
        "method": "enp",
        "mode": "uniform_expert_neuron_pruning",
        "profile_construction": "calibrated",
        "test_metrics_used_for_profile": False,
        "channel_block_size": 64,
        "profile_widths": widths,
        "cache_provenance": {
            "calibration": {"protocol_name": "c1_wikitext_train_128x2048_seed42_screening_v1"},
            "channel": {"sha256": file_sha256(channel_path)},
        },
        "enp": {"zero_token_policy": zero_token_policy},
    }
    return profile, channel_cache, channel_path


@pytest.mark.parametrize(("retained_channels", "width"), [(576, 9), (384, 6)])
def test_validate_enp_artifacts_accepts_wikitext_uniform_targets(
    tmp_path: Path,
    retained_channels: int,
    width: int,
) -> None:
    profile, channel_cache, channel_path = _artifacts(tmp_path, torch.full((48, 128), width))

    actual = validate_enp_artifacts(
        profile,
        channel_cache,
        retained_channels=retained_channels,
        expected_protocol_name="c1_wikitext_train_128x2048_seed42_screening_v1",
        channel_cache_path=channel_path,
    )

    assert bool((actual == width).all())


def test_validate_enp_artifacts_rejects_keep_full_profile(tmp_path: Path) -> None:
    widths = torch.full((2, 3), 6)
    widths[0, 0] = 12
    profile, channel_cache, channel_path = _artifacts(tmp_path, widths, zero_token_policy="keep_full")

    with pytest.raises(ValueError, match="same requested width"):
        validate_enp_artifacts(
            profile,
            channel_cache,
            retained_channels=384,
            expected_protocol_name="c1_wikitext_train_128x2048_seed42_screening_v1",
            channel_cache_path=channel_path,
        )


def test_export_gemma4_slices_packed_experts_and_preserves_dense_mlp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = 64
    retained = 32
    model_path = tmp_path / "gemma4-model"
    output = tmp_path / "gemma4-output"
    model_path.mkdir()
    prefix = "model.language_model.layers.0"
    tensors = {
        f"{prefix}.router.proj.weight": torch.ones(2, 4),
        f"{prefix}.mlp.down_proj.weight": torch.full((4, 8), 9.0),
        f"{prefix}.experts.gate_up_proj": torch.arange(2 * width * 2 * 4).reshape(2, width * 2, 4).float(),
        f"{prefix}.experts.down_proj": torch.arange(2 * 4 * width).reshape(2, 4, width).float(),
    }
    save_file(tensors, model_path / "model.safetensors")
    config = {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "moe_intermediate_size": width,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "top_k_experts": 1,
            "hidden_size": 4,
        },
    }
    index = {
        "metadata": {"total_size": sum(value.numel() * value.element_size() for value in tensors.values())},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    ranking = torch.stack((torch.arange(width - 1, -1, -1), torch.arange(width)))
    channel = {
        "purpose": "enp_tenp_signed_projection_channel_ranking",
        "split": "train",
        "test_metrics_used": False,
        "table": {
            0: {
                "ranked_indices": ranking,
                "block_relative_scores": torch.ones(2, 2),
                "block_coverage_scores": torch.full((2, 2), 0.5),
                "block_sizes": torch.tensor([32, 32]),
                "intermediate_size": width,
            }
        },
    }
    channel_path = tmp_path / "channels.pt"
    torch.save(channel, channel_path)
    profile = {
        "method": "enp",
        "mode": "uniform_expert_neuron_pruning",
        "profile_construction": "calibrated",
        "test_metrics_used_for_profile": False,
        "channel_block_size": 32,
        "profile_widths": torch.ones((1, 2), dtype=torch.long),
        "routed_param_retention": 0.5,
        "target_pruning_ratio": 0.5,
        "cache_provenance": {
            "calibration": {
                "protocol_name": "gemma4_wikitext128x2048_v1",
                "sha256": "unused",
                "input_ids_sha256": "unused",
            },
            "channel": {"sha256": file_sha256(channel_path)},
        },
        "enp": {"zero_token_policy": "prune_uniform"},
    }
    profile_path = tmp_path / "profile.pt"
    torch.save(profile, profile_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_uniform_enp_qwen3_moe",
            "--model-path",
            str(model_path),
            "--profile",
            str(profile_path),
            "--channel-cache",
            str(channel_path),
            "--output-dir",
            str(output),
            "--retained-channels",
            str(retained),
            "--expected-protocol-name",
            "gemma4_wikitext128x2048_v1",
        ],
    )

    assert export_enp() == 0

    kept = torch.arange(width - 1, retained - 1, -1)
    gate_up = f"{prefix}.experts.gate_up_proj"
    down = f"{prefix}.experts.down_proj"
    dense = f"{prefix}.mlp.down_proj.weight"
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
        exported_gate = handle.get_tensor(gate_up)
        assert exported_gate.shape == (2, retained * 2, 4)
        assert torch.equal(exported_gate[0, :retained], tensors[gate_up][0, :width].index_select(0, kept))
        assert torch.equal(exported_gate[0, retained:], tensors[gate_up][0, width:].index_select(0, kept))
        assert torch.equal(
            handle.get_tensor(down)[0],
            tensors[down][0].index_select(1, kept),
        )
        assert torch.equal(handle.get_tensor(dense), tensors[dense])
    exported_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert exported_config["text_config"]["moe_intermediate_size"] == retained
    manifest = json.loads((output / "pruning_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "enp"
    assert manifest["retained_channels"] == retained