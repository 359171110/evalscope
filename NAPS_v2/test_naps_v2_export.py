import torch

from NAPS_v2.export_naps_v2_checkpoint import apply_compensation_plan


def test_compensation_plan_changes_only_retained_down_columns() -> None:
    down = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    retained = torch.tensor([0, 2])
    plan = {
        "target_channels": [1],
        "representative_channels": [[2]],
        "coefficients": [[0.5]],
        "trust_region_scale": 1.0,
        "fallback_reason": None,
    }
    result = apply_compensation_plan(down, retained, plan)
    assert result.shape == (2, 2)
    assert torch.equal(result[:, 0], down[:, 0])
    assert torch.equal(result[:, 1], down[:, 2] + 0.5 * down[:, 1])


def test_failed_compensation_returns_mask_columns() -> None:
    down = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    retained = torch.tensor([1])
    plan = {"fallback_reason": "ridge_or_update_failure"}
    result = apply_compensation_plan(down, retained, plan)
    assert torch.equal(result, down[:, retained])