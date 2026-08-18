import torch

from NAPS_v2.naps_v2_core import (
    NapsV2Config,
    build_probe_sets,
    compensate_expert,
    dynamic_swap_fraction,
    effective_zero_mask,
    native_route,
    select_compensation_targets,
    select_v2_mask,
    stable_concat_score,
)
from NAPS_v2.export_naps_v2_checkpoint import apply_compensation_plan


def test_native_route_and_probe_anchor_are_distinct() -> None:
    probes = torch.eye(3)
    router = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _, selected, weights = native_route(probes, router, 1)
    probe_sets = build_probe_sets(probes, selected, weights, 2)
    assert probe_sets["native_rows"].tolist() == [2]
    assert not probe_sets["anchor_added"]
    assert probe_sets["coverage_rows"].tolist() == [2]


def test_zero_native_probe_adds_self_anchor() -> None:
    probes = torch.eye(3)
    selected = torch.tensor([[0], [0], [0]])
    weights = torch.ones(3, 1)
    probe_sets = build_probe_sets(probes, selected, weights, 2)
    assert probe_sets["native_rows"].numel() == 0
    assert probe_sets["anchor_added"]
    assert probe_sets["coverage_rows"].tolist() == [2]


def test_dynamic_swap_buckets_and_fixed_width() -> None:
    config = NapsV2Config()
    assert [dynamic_swap_fraction(x, config) for x in (0, 3, 5, 9, 17, 33)] == [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    gate = torch.arange(80, dtype=torch.float32).reshape(20, 4) + 1
    up = gate + 1
    down = torch.arange(80, dtype=torch.float32).reshape(4, 20) + 1
    zeros = effective_zero_mask(gate, up, down, config.effective_zero_threshold)
    scores = stable_concat_score(gate, up, down, config)
    order, diagnostics = select_v2_mask(
        torch.argsort(scores, descending=True, stable=True), scores, torch.ones(4, 20), zeros, 10, 5, config
    )
    assert order[:10].unique().numel() == 10
    assert diagnostics["actual_swaps"] == 1


def test_compensation_targets_and_trust_region() -> None:
    config = NapsV2Config(compensation_channel_cap_b6=2, representatives_per_target=1)
    output_mass = torch.tensor([10.0, 9.0, 8.0, 1.0])
    targets, fraction = select_compensation_targets(
        output_mass, torch.tensor([0, 1]), torch.zeros(4, dtype=torch.bool), 0.8, 2
    )
    assert targets.tolist() == [2]
    assert fraction > 0.8
    responses = torch.tensor([[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, 0.2, 0.5]])
    down = torch.eye(2, 4)
    updated, diagnostics = compensate_expert(
        responses, down, torch.tensor([0, 1]), torch.zeros(4, dtype=torch.bool), output_mass, 0, config
    )
    assert updated.shape == (2, 2)
    assert diagnostics["update_ratio_final"] <= 0.02 + 1.0e-6
    reconstructed = apply_compensation_plan(down, torch.tensor([0, 1]), diagnostics)
    assert torch.allclose(reconstructed, updated)
