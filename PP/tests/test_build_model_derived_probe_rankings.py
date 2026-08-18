import torch

from PP.build_model_derived_probe_rankings import (
    aimer_filled_order,
    expert_spectral_probes,
    order_to_scores,
    previous_write_probes,
    protection_overlap,
    swiglu_probe_importance,
)


def test_swiglu_probe_importance_uses_absolute_top_q_without_down_norm() -> None:
    probes = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    gate = torch.eye(2)
    up = torch.eye(2)

    scores = swiglu_probe_importance(probes, gate, up, top_q=2)

    expected_first = (
        torch.nn.functional.silu(torch.tensor(1.0))
        + torch.nn.functional.silu(torch.tensor(-1.0)).abs()
    ) / 2
    expected_second = torch.nn.functional.silu(torch.tensor(1.0)) / 2
    assert torch.allclose(scores, torch.tensor([expected_first, expected_second]))


def test_expert_spectral_probes_find_dominant_joint_input_directions() -> None:
    gate = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    up = torch.diag(torch.tensor([4.0, 1.0, 0.5]))
    router = torch.tensor([-1.0, 0.0, 0.0])
    norm_weight = torch.ones(3)

    probes, eigenvalues, concentration = expert_spectral_probes(
        gate,
        up,
        router,
        norm_weight,
        probe_count=2,
        oversample=1,
        power_iterations=2,
        seed=7,
        eps=1.0e-6,
    )

    assert torch.allclose(eigenvalues, torch.tensor([25.0, 5.0]), atol=1.0e-4)
    assert torch.dot(probes[0], router) >= 0.0
    assert torch.allclose(torch.sort(probes.abs(), dim=1).values[:, -1], torch.full((2,), 3**0.5), atol=1.0e-4)
    assert abs(concentration - 30.0 / 31.25) < 1.0e-5


def test_previous_write_probes_stream_top_affinity_and_orient_signs() -> None:
    router = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    norm_weight = torch.ones(2)
    previous_down = [
        torch.tensor([[2.0, 0.0], [0.0, -3.0]]),
        torch.tensor([[1.0, 1.0], [1.0, -1.0]]),
    ]

    probes, scores = previous_write_probes(
        router,
        norm_weight,
        previous_down,
        probe_count=1,
        eps=1.0e-6,
    )

    assert probes.shape == (2, 1, 2)
    assert torch.all((router.unsqueeze(1) * probes).sum(dim=2) >= 0.0)
    assert torch.allclose(scores, torch.full((2, 1), 2**0.5), atol=1.0e-4)


def test_order_scores_and_overlap_preserve_frozen_pp_fallback() -> None:
    order = torch.tensor([2, 0, 3, 1])
    scores = order_to_scores(order)

    assert torch.argsort(scores, descending=True, stable=True).tolist() == order.tolist()
    assert protection_overlap(order, scores, protected_channels=2) == 1.0


def test_aimer_fill_places_probe_protection_before_backbone() -> None:
    aimer_order = torch.tensor([3, 0, 5, 1, 4, 2])
    probe_scores = torch.tensor([0.0, 4.0, 1.0, 2.0, 3.0, 5.0])

    order = aimer_filled_order(aimer_order, probe_scores, protected_channels=2)

    assert order.tolist() == [5, 1, 3, 0, 4, 2]
    assert torch.equal(torch.sort(order).values, torch.arange(6))