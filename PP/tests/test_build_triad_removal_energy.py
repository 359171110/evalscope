from __future__ import annotations

import torch

from PP.build_triad_removal_energy import triad_boundary_order, triad_removal_energy


def test_triad_removal_energy_matches_closed_form() -> None:
    gate = torch.tensor([[1.0, 2.0], [0.0, 3.0]])
    up = torch.tensor([[2.0, -1.0], [4.0, 0.0]])
    down = torch.tensor([[1.0, 2.0], [2.0, -1.0]])

    energy = triad_removal_energy(gate, up, down)

    expected = torch.tensor(
        [
            5.0 * (5.0 * 5.0 + 0.0**2),
            5.0 * (9.0 * 16.0 + 0.0**2),
        ]
    )
    assert torch.allclose(energy, expected)


def test_triad_boundary_freezes_pp_and_high_confidence_aimer_channels() -> None:
    aimer_order = torch.arange(10)
    pseudo_order = torch.tensor([9, *range(9)])
    energy = torch.arange(10, dtype=torch.float32)

    order, diagnostics = triad_boundary_order(
        aimer_order,
        pseudo_order,
        energy,
        retained_channels=6,
        protected_channels=1,
        boundary_channels=2,
    )

    assert order[:6].tolist() == [9, 0, 1, 2, 6, 5]
    assert diagnostics["replacements"] == 2.0
    assert sorted(order.tolist()) == list(range(10))


def test_triad_boundary_selects_only_from_cutoff_pool() -> None:
    aimer_order = torch.tensor([7, 0, 1, 2, 3, 4, 5, 6])
    pseudo_order = torch.tensor([6, 0, 1, 2, 3, 4, 5, 7])
    energy = torch.tensor([1.0, 1.0, 100.0, 90.0, 80.0, 70.0, 1000.0, 2000.0])

    order, _ = triad_boundary_order(
        aimer_order,
        pseudo_order,
        energy,
        retained_channels=5,
        protected_channels=1,
        boundary_channels=1,
    )

    assert order[:5].tolist() == [6, 7, 0, 1, 2]
    assert 4 not in order[:5]


def test_triad_boundary_uses_aimer_order_for_energy_ties() -> None:
    aimer_order = torch.arange(6)
    pseudo_order = torch.arange(6)
    energy = torch.ones(6)

    order, diagnostics = triad_boundary_order(
        aimer_order,
        pseudo_order,
        energy,
        retained_channels=3,
        protected_channels=0,
        boundary_channels=1,
    )

    assert order[:3].tolist() == [0, 1, 2]
    assert diagnostics["overlap_with_aimer"] == 1.0