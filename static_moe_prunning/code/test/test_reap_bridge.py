from __future__ import annotations

import torch

from src.reap_bridge import build_reap_profile_payload
from src.static_expert_pruning import validate_static_profile_payload


def test_reap_profile_prunes_lowest_saliency_per_layer_at_exact_budget() -> None:
    observer_data = {
        0: {"reap": torch.tensor([0.4, 0.1, 0.3, 0.2])},
        1: {"reap": torch.tensor([0.2, 0.4, 0.1, 0.3])},
    }
    calibration = {
        "split": "train",
        "sequence_length": 2048,
        "calibration_sequences": 2,
        "calibration_tokens": 4096,
        "input_ids_sha256": "a" * 64,
        "frozen_before_profile": True,
        "test_metrics_used": False,
    }

    profile = build_reap_profile_payload(
        observer_data=observer_data,
        model_path="/models/qwen3",
        calibration_payload=calibration,
        calibration_file_sha256="b" * 64,
        channel_file_sha256="c" * 64,
        official_reap_commit="1970473c51ca3caeb98c10392f15b3a08a672974",
        num_blocks=12,
        experts_to_prune_per_layer=2,
        top_k=2,
        renormalize_router_weights=True,
    )

    assert profile["method"] == "official_reap"
    assert profile["allocation_scope"] == "per_layer"
    assert profile["retained_experts_by_layer"] == [2, 2]
    assert profile["retained_expert_mask"].tolist() == [
        [True, False, True, False],
        [False, True, False, True],
    ]
    assert profile["profile_widths"].tolist() == [
        [12, 0, 12, 0],
        [0, 12, 0, 12],
    ]
    assert profile["actual_blocks_by_layer"] == [24, 24]
    assert profile["observer"]["renormalize_router_weights"] is True
    assert profile["test_metrics_used_for_profile"] is False
    assert validate_static_profile_payload(profile).tolist() == profile["profile_widths"].tolist()