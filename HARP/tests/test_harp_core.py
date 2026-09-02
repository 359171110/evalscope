from __future__ import annotations

import torch

from HARP.harp_core import allocate_expert_widths, allocate_layer_upgrade_units, detect_anchor_layer


def test_detect_anchor_layer_uses_two_sigma_rule() -> None:
    assert detect_anchor_layer(torch.tensor([10.0] + [1.0] * 47)) == 0
    assert detect_anchor_layer(torch.tensor([1.0] * 48)) is None


def test_layer_and_expert_allocations_are_aligned_and_exact() -> None:
    layers = allocate_layer_upgrade_units(
        torch.tensor([3.0, 2.0, 1.0]), total_units=6, max_units_per_layer=4,
    )
    assert layers.tolist() == [2, 2, 2]
    experts = allocate_expert_widths(torch.tensor([1.0, 3.0, 2.0]), low_blocks=5, target_units=3)
    assert experts.tolist() == [5, 7, 6]
    assert int((experts - 5).sum()) == 3
