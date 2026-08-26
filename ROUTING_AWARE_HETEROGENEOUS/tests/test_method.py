"""Focused tests for the routing-aware heterogeneous method."""

from __future__ import annotations

import torch

from ROUTING_AWARE_HETEROGENEOUS.allocation import allocate_widths
from ROUTING_AWARE_HETEROGENEOUS.config import MethodConfig
from ROUTING_AWARE_HETEROGENEOUS.core import RoutingAwarePruner
from ROUTING_AWARE_HETEROGENEOUS.ops import ridge_fold_down
from ROUTING_AWARE_HETEROGENEOUS.toy import ToyAdapter


def test_exact_discrete_budget() -> None:
    costs = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.5, 1.0]])
    options = torch.tensor([8, 4, 2])
    widths = allocate_widths(costs, options, budget=6)
    assert int(widths.sum()) == 6
    assert set(widths.tolist()) <= set(options.tolist())


def test_ridge_fold_preserves_width_and_runs() -> None:
    activation = torch.randn(16, 8)
    down = torch.randn(5, 8)
    folded, diagnostics = ridge_fold_down(activation, down, torch.tensor([0, 2, 4, 6]), ridge=1.0e-4, epsilon=1.0e-8)
    assert folded.shape == (5, 4)
    assert diagnostics["error_after"] >= 0.0


def test_guided_completion_does_not_change_natural_mass() -> None:
    adapter = ToyAdapter(hidden=8, channels=8, experts=3)
    config = MethodConfig(
        natural_sequences=4,
        guided_sequences=2,
        sequence_length=16,
        max_samples_per_expert=8,
        min_samples_per_expert=2,
        safe_samples_per_expert=1,
        retention=0.5,
        device="cpu",
    )
    inputs = torch.arange(64).reshape(4, 16)
    pruner = RoutingAwarePruner(adapter, config)
    pools = pruner.collect_calibration(inputs)
    natural_mass = pools.natural_mass.clone()
    result = pruner.run(inputs)
    assert torch.equal(natural_mass, pools.natural_mass)
    assert torch.equal(result.natural_mass, result.natural_mass.clone())
    assert torch.all(result.widths.sum(dim=1) == 12)