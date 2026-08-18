from __future__ import annotations

import pytest
import torch

from src.budgeted_micro_expert import (
    allocate_prefix_blocks,
    apply_hierarchical_completion,
    prefix_mask_from_widths,
)


def test_allocator_enforces_exact_block_budget_and_prefixes() -> None:
    parent = torch.tensor([[0.8, 0.2]])
    marginal = torch.tensor(
        [[[0.40, 0.30, 0.20, 0.10], [0.05, 0.90, 0.04, 0.01]]]
    )

    allocation = allocate_prefix_blocks(parent, marginal, total_blocks=5)

    assert torch.equal(allocation.widths, torch.tensor([[3, 2]]))
    assert int(allocation.block_mask.sum()) == 5
    assert torch.equal(
        allocation.block_mask,
        torch.tensor([[[True, True, True, False], [True, True, False, False]]]),
    )


def test_allocator_assigns_one_sentinel_per_slot_at_minimum_budget() -> None:
    allocation = allocate_prefix_blocks(
        torch.tensor([[0.7, 0.2, 0.1]]),
        torch.full((1, 3, 4), 0.25),
        total_blocks=3,
    )

    assert torch.equal(allocation.widths, torch.ones((1, 3), dtype=torch.long))


def test_allocator_supports_floor_free_dynamic_teacher() -> None:
    allocation = allocate_prefix_blocks(
        torch.tensor([[0.9, 0.1]]),
        torch.tensor([[[0.6, 0.4], [0.6, 0.4]]]),
        total_blocks=1,
        min_blocks_per_slot=0,
    )

    assert allocation.widths.tolist() == [[1, 0]]
    assert int(allocation.block_mask.sum()) == 1


def test_allocator_rejects_infeasible_budgets() -> None:
    parent = torch.ones((1, 2))
    marginal = torch.full((1, 2, 4), 0.25)

    with pytest.raises(ValueError, match="at least one block"):
        allocate_prefix_blocks(parent, marginal, total_blocks=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        allocate_prefix_blocks(parent, marginal, total_blocks=9)


def test_allocator_supports_distinct_per_token_budgets() -> None:
    allocation = allocate_prefix_blocks(
        torch.tensor([[0.8, 0.2], [0.6, 0.4]]),
        torch.full((2, 2, 4), 0.25),
        total_blocks=torch.tensor([2, 7]),
    )

    assert torch.equal(allocation.block_mask.sum(dim=(1, 2)), torch.tensor([2, 7]))
    assert torch.equal(allocation.widths[0], torch.tensor([1, 1]))


def test_prefix_mask_supports_dense_boundary() -> None:
    mask = prefix_mask_from_widths(torch.tensor([[4, 4]]), num_blocks=4)
    assert bool(mask.all())


def test_hierarchical_completion_reduces_to_global_when_lambda_is_zero() -> None:
    partial = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    gate = torch.tensor([[0.75, 0.25]])
    coverage = torch.tensor([[0.5, 0.25]])

    hierarchical, aux = apply_hierarchical_completion(
        partial, gate, coverage, local_weight=torch.zeros_like(coverage)
    )

    observed = (gate.unsqueeze(-1) * partial).sum(dim=1)
    observed_mass = (gate * coverage).sum(dim=1, keepdim=True)
    assert torch.allclose(hierarchical, observed / observed_mass)
    assert torch.allclose(aux["observed_mass"], observed_mass)


def test_full_coverage_reduces_to_observed_dense_mixture() -> None:
    partial = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])
    gate = torch.tensor([[0.6, 0.4]])
    coverage = torch.ones_like(gate)

    output, aux = apply_hierarchical_completion(partial, gate, coverage)

    expected = (gate.unsqueeze(-1) * partial).sum(dim=1)
    assert torch.allclose(output, expected)
    assert torch.count_nonzero(aux["missing_mass"]) == 0


def test_hierarchical_completion_is_finite_at_low_coverage() -> None:
    partial = torch.tensor([[[1.0, -1.0], [0.5, 0.25]]])
    gate = torch.tensor([[0.9, 0.1]])
    coverage = torch.tensor([[0.0, 1.0e-12]])

    output, aux = apply_hierarchical_completion(
        partial,
        gate,
        coverage,
        local_weight=coverage,
        eps=1.0e-6,
        max_correction_ratio=2.0,
    )

    assert bool(torch.isfinite(output).all())
    assert bool(torch.isfinite(aux["correction"]).all())


def test_observed_override_preserves_full_path_dtype() -> None:
    partial = torch.zeros((1, 2, 2), dtype=torch.bfloat16)
    observed = torch.tensor([[1.125, -2.25]], dtype=torch.float32)
    gate = torch.tensor([[0.6, 0.4]])
    coverage = torch.ones_like(gate)

    output, _ = apply_hierarchical_completion(
        partial,
        gate,
        coverage,
        observed_override=observed,
    )

    assert output.dtype == torch.float32
    assert torch.equal(output, observed)
