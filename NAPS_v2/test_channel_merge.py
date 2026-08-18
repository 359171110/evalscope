from __future__ import annotations

import torch

from NAPS_v2.channel_merge import (
    ChannelMergeConfig,
    apply_channel_merge_plan,
    fit_channel_merge_plan,
)


def _recoverable_problem(holdout_sign: float = 1.0) -> tuple[torch.Tensor, ...]:
    fit_representative = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5])
    holdout_representative = torch.tensor([0.25, -0.25, 1.25, -1.25])
    fit_responses = torch.stack((
        fit_representative,
        torch.linspace(-1.0, 1.0, 8),
        1.5 * fit_representative,
        torch.zeros(8),
    ), dim=1)
    holdout_responses = torch.stack((
        holdout_representative,
        torch.linspace(-0.5, 0.5, 4),
        holdout_sign * 1.5 * holdout_representative,
        torch.zeros(4),
    ), dim=1)
    down = torch.tensor([
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 1.0, -0.25, 0.0],
    ])
    retained = torch.tensor([0, 1])
    utility = torch.tensor([1.0, 1.0, 10.0, 0.0])
    zero_mask = torch.tensor([False, False, False, True])
    return fit_responses, holdout_responses, down, retained, utility, zero_mask


def test_sparse_response_merge_recovers_pruned_channel_at_fixed_width() -> None:
    fit, holdout, down, retained, utility, zero_mask = _recoverable_problem()
    plan = fit_channel_merge_plan(
        fit,
        torch.ones(fit.shape[0]),
        holdout,
        torch.ones(holdout.shape[0]),
        down,
        retained,
        utility,
        zero_mask,
        ChannelMergeConfig(max_update_ratio=1.0),
    )

    merged_down = apply_channel_merge_plan(down, retained, plan)

    assert plan["accepted"]
    assert plan["target_channels"] == [2]
    assert plan["representative_channels"] == [0]
    assert plan["coefficients"] == [1.5]
    assert plan["fit_candidate_loss"] < plan["fit_baseline_loss"]
    assert plan["holdout_candidate_loss"] < plan["holdout_baseline_loss"]
    assert merged_down.shape == (down.shape[0], retained.numel())
    assert torch.allclose(merged_down[:, 0], down[:, 0] + 1.5 * down[:, 2])
    assert torch.equal(merged_down[:, 1], down[:, 1])


def test_sparse_response_merge_rejects_holdout_regression() -> None:
    fit, holdout, down, retained, utility, zero_mask = _recoverable_problem(holdout_sign=-1.0)
    plan = fit_channel_merge_plan(
        fit,
        torch.ones(fit.shape[0]),
        holdout,
        torch.ones(holdout.shape[0]),
        down,
        retained,
        utility,
        zero_mask,
        ChannelMergeConfig(max_update_ratio=1.0),
    )

    merged_down = apply_channel_merge_plan(down, retained, plan)

    assert not plan["accepted"]
    assert plan["fallback_reason"] == "fit_or_holdout_gate_rejected"
    assert plan["fit_candidate_loss"] < plan["fit_baseline_loss"]
    assert plan["holdout_candidate_loss"] > plan["holdout_baseline_loss"]
    assert torch.equal(merged_down, down.index_select(1, retained))


def test_sparse_response_merge_enforces_trust_region() -> None:
    fit, holdout, down, retained, utility, zero_mask = _recoverable_problem()
    plan = fit_channel_merge_plan(
        fit,
        torch.ones(fit.shape[0]),
        holdout,
        torch.ones(holdout.shape[0]),
        down,
        retained,
        utility,
        zero_mask,
        ChannelMergeConfig(max_update_ratio=0.01),
    )

    assert plan["update_ratio_raw"] > 0.01
    assert plan["update_ratio_final"] <= 0.0100001
    assert 0.0 < plan["trust_region_scale"] < 1.0


def test_sparse_response_merge_requires_independent_holdout_rows() -> None:
    fit, _, down, retained, utility, zero_mask = _recoverable_problem()
    plan = fit_channel_merge_plan(
        fit,
        torch.ones(fit.shape[0]),
        torch.empty(0, fit.shape[1]),
        torch.empty(0),
        down,
        retained,
        utility,
        zero_mask,
    )

    assert not plan["accepted"]
    assert plan["fallback_reason"] == "insufficient_holdout_rows"


def test_sparse_response_merge_falls_back_without_active_representatives() -> None:
    fit, holdout, down, retained, utility, zero_mask = _recoverable_problem()
    zero_mask[retained] = True
    plan = fit_channel_merge_plan(
        fit,
        torch.ones(fit.shape[0]),
        holdout,
        torch.ones(holdout.shape[0]),
        down,
        retained,
        utility,
        zero_mask,
    )

    assert not plan["accepted"]
    assert plan["fallback_reason"] == "no_active_retained_representatives"