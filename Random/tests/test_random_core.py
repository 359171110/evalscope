from __future__ import annotations

import torch

from Random.random_core import build_layer_orders, random_channel_order, retained_prefix, validate_rankings


def test_random_orders_are_deterministic_for_seed() -> None:
    first = random_channel_order(16, seed=42, layer_id=3, expert_id=7)
    second = random_channel_order(16, seed=42, layer_id=3, expert_id=7)
    assert torch.equal(first, second)
    assert torch.equal(torch.sort(first).values, torch.arange(16))


def test_random_orders_do_not_depend_on_call_order() -> None:
    first = random_channel_order(32, seed=7, layer_id=2, expert_id=5)
    random_channel_order(32, seed=7, layer_id=0, expert_id=0)
    random_channel_order(32, seed=7, layer_id=2, expert_id=4)
    second = random_channel_order(32, seed=7, layer_id=2, expert_id=5)
    assert torch.equal(first, second)


def test_different_experts_draw_independent_permutations() -> None:
    left = random_channel_order(64, seed=42, layer_id=0, expert_id=0)
    right = random_channel_order(64, seed=42, layer_id=0, expert_id=1)
    assert not torch.equal(left, right)


def test_retained_prefix_keeps_the_leading_k_channels() -> None:
    order = random_channel_order(8, seed=42, layer_id=0, expert_id=0)
    kept = retained_prefix(order, 3)
    assert torch.equal(kept, order[:3])


def test_build_layer_orders_covers_requested_moe_layers() -> None:
    orders = build_layer_orders((1, 2), 3, 8, seed=42)
    table = {
        layer_id: {
            "ranked_indices": ranking,
        }
        for layer_id, ranking in orders.items()
    }
    validate_rankings(table, 2, 3, 8, layer_ids=(1, 2))
    assert set(orders) == {1, 2}
    assert tuple(orders[1].shape) == (3, 8)
