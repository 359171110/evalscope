from __future__ import annotations

import torch

from NAPS.naps_core import (
    NapsConfig,
    build_one_to_one_merge_plan,
    effective_zero_mask,
    native_route,
    stable_concat_score,
    validate_merge_plan,
)
from NAPS.export_naps_checkpoint import merge_columns


def test_stable_concat_preserves_active_score_and_demotes_zero() -> None:
    config = NapsConfig()
    gate = torch.tensor([[1.0, -2.0], [0.0, 0.0]])
    up = torch.tensor([[3.0, -4.0], [0.0, 0.0]])
    down = torch.tensor([[5.0, 0.0], [-6.0, 0.0]])

    values = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    expected = values.square().mean().sqrt() / (values.abs().mean() + config.aimer_epsilon)
    score = stable_concat_score(gate, up, down, config)

    assert effective_zero_mask(gate, up, down, config.effective_zero_threshold).tolist() == [False, True]
    assert torch.equal(score[:1], expected.reshape(1))
    assert score[1].item() == -torch.inf


def test_native_route_uses_index_order_for_ties_and_renormalizes() -> None:
    probes = torch.tensor([[1.0, 0.0]])
    router = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    _, selected, weights = native_route(probes, router, top_k=2)

    assert selected.tolist() == [[0, 1]]
    assert torch.equal(weights, torch.tensor([[0.5, 0.5]]))


def test_zero_ties_are_deterministic_capacity_fillers() -> None:
    config = NapsConfig()
    gate = torch.tensor([[1.0], [0.0], [0.0]])
    up = gate.clone()
    down = gate.transpose(0, 1).clone()

    order = torch.argsort(stable_concat_score(gate, up, down, config), descending=True, stable=True)

    assert order.tolist() == [0, 1, 2]


def test_merge_keeps_mask_width_and_uses_unique_representatives() -> None:
    config = NapsConfig(beta_max=2.0, column_growth_max=10.0, expert_delta_max=10.0)
    responses = torch.tensor([[1.0, 0.5, 0.25], [0.5, 1.0, 0.25]])
    down = torch.tensor([[1.0, 0.5, 0.25], [0.5, 1.0, 0.25]])
    retained = torch.tensor([0, 1])
    displaced = torch.tensor([2])
    weights = torch.tensor([0.6, 0.4])

    plan = build_one_to_one_merge_plan(responses, down, retained, displaced, weights, config)
    validated, merged = validate_merge_plan(responses, down, retained, weights, plan, config)

    representatives = [pair["representative"] for pair in validated["pairs"]]
    assert len(representatives) == len(set(representatives))
    assert merged.shape == (down.shape[0], retained.numel())
    assert retained.tolist() == [0, 1]


def test_export_merge_changes_only_retained_columns() -> None:
    down = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    retained = torch.tensor([0, 2])
    payload = {
        "layers": {
            0: {
                1: {
                    "pairs": [{"pruned": 1, "representative": 2, "beta": 0.5}],
                }
            }
        }
    }

    merged = merge_columns(down, 0, 1, retained, payload)

    assert merged.shape == (2, 2)
    assert torch.equal(merged[:, 0], down[:, 0])
    assert torch.equal(merged[:, 1], down[:, 2] + 0.5 * down[:, 1])