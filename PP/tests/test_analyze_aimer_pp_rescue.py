from __future__ import annotations

import torch

from PP.analyze_aimer_pp_rescue import extract_budget_records, inverse_ranks, pp_no_down_norm_score


def test_pp_no_down_norm_score_uses_top_q_absolute_swiglu_response() -> None:
    probes = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    gate = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    up = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    scores = pp_no_down_norm_score(probes, gate, up, top_q=1)

    expected = torch.tensor(
        [
            torch.nn.functional.silu(torch.tensor(2.0)) * 2.0,
            abs(torch.nn.functional.silu(torch.tensor(-2.0)) * 2.0),
        ]
    )
    assert torch.allclose(scores, expected)


def test_extract_budget_records_finds_rescue_and_displaced_channels() -> None:
    aimer = torch.tensor([[[0, 1, 2, 3]]])
    combined = torch.tensor([[[3, 0, 1, 2]]])
    pp = torch.tensor([[[3, 2, 1, 0]]])
    scores = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    records, summary = extract_budget_records(
        budget="B2",
        retained_channels=2,
        aimer_orders=aimer,
        combined_orders=combined,
        aimer_ranks=inverse_ranks(aimer),
        pp_ranks=inverse_ranks(pp),
        pp_scores=scores,
    )

    rescue = next(record for record in records if record["population"] == "rescue")
    displaced = next(record for record in records if record["population"] == "displaced")
    assert rescue["channel_id"] == 3
    assert rescue["aimer_rank"] == 4
    assert rescue["pp_rank"] == 1
    assert displaced["channel_id"] == 1
    assert summary["rescue_channels"] == 1
    assert summary["displaced_channels"] == 1
    assert summary["rescue_depth_below_cutoff"]["median"] == 2.0
    assert summary["displaced_aimer_rank"]["median"] == 2.0
    assert summary["rescue_fraction_below_aimer_cutoff"] == 1.0
    assert summary["rescue_fraction_aimer_low_pp_high"] == 1.0