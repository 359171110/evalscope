from __future__ import annotations

import torch

from WICK.build_random_profiles import build_random_orders


def test_random_protection_preserves_protected_prefix_and_budget() -> None:
    pseudo_order = torch.tensor([[[5, 2, 7, 1, 0, 3, 4, 6]]])
    random_orders, protected_orders = build_random_orders(
        pseudo_order,
        retained_channels=4,
        protected_channels=2,
        seed=42,
    )

    assert set(random_orders[0, 0].tolist()) == set(range(8))
    assert protected_orders[0, 0, :2].tolist() == [5, 2]
    assert len(set(protected_orders[0, 0, :4].tolist())) == 4
    assert set(protected_orders[0, 0].tolist()) == set(range(8))


def test_random_orders_are_deterministic_for_seed() -> None:
    pseudo_order = torch.arange(8).repeat(1, 2, 1)
    first = build_random_orders(pseudo_order, retained_channels=4, protected_channels=2, seed=7)
    second = build_random_orders(pseudo_order, retained_channels=4, protected_channels=2, seed=7)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])