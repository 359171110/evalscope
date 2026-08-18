import torch

from PP.build_gate_hybrid_protection import (
    build_aimer_filled_order,
    gate_accessibility,
    select_protection_sets,
)


def test_gate_accessibility_uses_positive_cosine_top_q_mean() -> None:
    probes = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    gate = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])

    scores = gate_accessibility(probes, gate, top_q=2)

    assert torch.allclose(scores, torch.tensor([0.5, 0.5, 0.5]))


def test_gate_protection_uses_top_77_and_reports_overlap() -> None:
    channel_count = 768
    pp_order = torch.arange(channel_count)
    ga_scores = torch.arange(channel_count, dtype=torch.float32)

    protected, diagnostics = select_protection_sets(
        pp_order,
        ga_scores,
        method="GateGA",
        total_protected=77,
    )

    assert protected.tolist() == list(range(767, 690, -1))
    assert diagnostics["protected_channels"] == 77.0
    assert diagnostics["pp10_ga10_overlap"] == 0.0


def test_hybrid_uses_pp38_and_disjoint_ga39() -> None:
    channel_count = 768
    pp_order = torch.arange(channel_count)
    ga_scores = torch.arange(channel_count, dtype=torch.float32)

    protected, diagnostics = select_protection_sets(
        pp_order,
        ga_scores,
        method="Hybrid",
        total_protected=77,
    )

    assert protected.numel() == 77
    assert protected[:38].tolist() == list(range(38))
    assert protected[38:].tolist() == list(range(767, 728, -1))
    assert protected.unique().numel() == 77
    assert diagnostics["pp5_channels"] == 38.0
    assert diagnostics["ga5_channels"] == 39.0


def test_aimer_fill_preserves_protected_prefix_and_permutation() -> None:
    aimer_order = torch.tensor([3, 0, 7, 2, 6, 1, 5, 4])
    protected = torch.tensor([5, 2])

    order = build_aimer_filled_order(aimer_order, protected)

    assert order.tolist() == [5, 2, 3, 0, 7, 6, 1, 4]
    assert torch.equal(torch.sort(order).values, torch.arange(8))


def test_gate_accessibility_rejects_nonfinite_input() -> None:
    probes = torch.tensor([[1.0, float("nan")]])
    gate = torch.tensor([[1.0, 0.0]])

    try:
        gate_accessibility(probes, gate, top_q=1)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite inputs must be rejected")
