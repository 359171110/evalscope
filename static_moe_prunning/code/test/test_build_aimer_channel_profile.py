from __future__ import annotations

from pathlib import Path

import torch

from scripts.build_aimer_channel_profile import (
    _load_model_config,
    build_aimer_channel_artifacts,
    channel_aimer_importance,
)
from src.static_expert_pruning import validate_static_profile_payload


def test_channel_aimer_importance_prefers_lower_aimer_ratio() -> None:
    gate = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    up = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    down = torch.tensor([[1.0, 2.0], [1.0, 0.0]])

    importance = channel_aimer_importance(gate, up, down)

    assert importance.shape == (2,)
    assert importance[1] > importance[0]


def test_gauge_balanced_aimer_is_invariant_to_up_down_rescaling() -> None:
    gate = torch.tensor([[1.0, -2.0, 0.5], [0.2, 0.3, -0.4]])
    up = torch.tensor([[2.0, -1.0, 3.0], [0.5, 1.5, -2.0]])
    down = torch.tensor([[1.0, -0.5], [2.0, 1.0], [-1.0, 0.25], [0.5, -2.0]])
    channel_scale = torch.tensor([10.0, 0.1])

    baseline = channel_aimer_importance(gate, up, down, score_variant="gauge_balanced")
    rescaled = channel_aimer_importance(
        gate,
        up * channel_scale[:, None],
        down / channel_scale[None, :],
        score_variant="gauge_balanced",
    )

    torch.testing.assert_close(rescaled, baseline)


def test_channel_aimer_profile_allocates_exact_per_layer_budget() -> None:
    raw_scores = {
        0: torch.tensor(
            [
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                [4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0],
            ]
        )
    }

    channel, profile = build_aimer_channel_artifacts(
        model_path=Path("/models/qwen3"),
        raw_scores_by_layer=raw_scores,
        target_pruning_ratio=0.5,
        block_size=2,
        source_identity={"commit": "test"},
    )

    widths = validate_static_profile_payload(profile)
    assert widths.shape == (1, 2)
    assert widths.sum(dim=1).tolist() == [4]
    assert profile["mode"] == "aimer_weight_only_channel"
    assert profile["profile_construction"] == "calibration_free"
    assert profile["allocation_scope"] == "per_layer"
    assert profile["retained_expert_mask"] is None
    ranked = channel["table"][0]["ranked_indices"]
    assert ranked[0].tolist() == list(range(8))


def test_qwen36_channel_aimer_uses_nested_text_config() -> None:
    config = _load_model_config(Path("/data01/datasets/Qwen3.6-35B-A3B"))

    assert config["model_type"] == "qwen3_5_moe_text"
    assert (config["num_hidden_layers"], config["num_experts"], config["moe_intermediate_size"]) == (40, 256, 512)
