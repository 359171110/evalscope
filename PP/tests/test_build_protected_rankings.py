from __future__ import annotations

from pathlib import Path

import torch

from PP.build_protected_rankings import build_protected_artifacts, build_protected_orders
from src.channel_runtime import channel_table_from_payload
from src.static_expert_pruning import validate_static_profile_payload


def test_build_protected_orders_places_pseudo_prefix_before_backbone() -> None:
    backbone = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]])
    pseudo = torch.tensor([[[5, 2, 7, 1, 0, 3, 4, 6]]])

    combined = build_protected_orders(backbone, pseudo, protected_channels=2)

    assert combined.tolist() == [[[5, 2, 0, 1, 3, 4, 6, 7]]]


def test_build_protected_artifacts_uses_fixed_expert_width() -> None:
    orders = torch.arange(8).repeat(1, 2, 1)

    channel, profile = build_protected_artifacts(
        model_path=Path("/models/qwen3"),
        orders=orders,
        method="aimer_pp_g10_b2of4",
        backbone="aimer",
        retained_blocks=2,
        protection_ratio=0.125,
        block_size=2,
        backbone_cache_sha256="backbone",
        pseudo_cache_sha256="pseudo",
    )

    widths = validate_static_profile_payload(profile)
    table = channel_table_from_payload(channel["table"])
    assert widths.tolist() == [[2, 2]]
    assert table[0].ranked_indices.shape == (2, 8)
    assert profile["pseudo_protection"]["protected_channels"] == 1