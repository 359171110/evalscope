from __future__ import annotations

import torch

from AIMER_MIX_PLUS.plus_core import PseudoSource
from AIMER_UNIFY.unify_core import UnifyConfig, build_unify_ranking, keep_set_overlap


def test_overlap_is_one_when_keep_sets_match() -> None:
    left = torch.tensor([1.0, 0.8, 0.2, 0.0])
    right = torch.tensor([0.9, 0.7, 0.1, 0.0])
    assert keep_set_overlap(left, right, 2) == 1.0


def test_overlap_is_zero_when_keep_sets_disjoint() -> None:
    left = torch.tensor([1.0, 0.8, 0.2, 0.0])
    right = torch.tensor([0.0, 0.1, 0.8, 1.0])
    assert keep_set_overlap(left, right, 2) == 0.0


def test_high_overlap_follows_mix() -> None:
    mix = torch.tensor([[[0.9, 0.8, 0.1, 0.0]]])
    layerprop = PseudoSource(
        "layerprop",
        torch.tensor([[[0, 1, 2, 3]]]),
        coverage=torch.ones(1, 1),
        hit_counts=torch.full((1, 1), 1.0e6),
    )
    prp = PseudoSource(
        "prp",
        torch.tensor([[[0, 1, 2, 3]]]),
        coverage=torch.ones(1, 1),
        hit_counts=torch.ones(1, 1),
    )
    order, diag = build_unify_ranking(mix, 2, [layerprop, prp], UnifyConfig())
    assert diag["diagnostics"][0][0]["mix_alpha"] == 1.0
    assert torch.equal(order[0, 0, :2], torch.tensor([0, 1]))


def test_zero_overlap_follows_ffn_sources() -> None:
    mix = torch.tensor([[[0.9, 0.8, 0.1, 0.0]]])
    layerprop = PseudoSource(
        "layerprop",
        torch.tensor([[[3, 2, 1, 0]]]),
        coverage=torch.ones(1, 1),
        hit_counts=torch.full((1, 1), 1.0e6),
    )
    prp = PseudoSource(
        "prp",
        torch.tensor([[[3, 2, 1, 0]]]),
        coverage=torch.ones(1, 1),
        hit_counts=torch.ones(1, 1),
    )
    order, diag = build_unify_ranking(mix, 2, [layerprop, prp], UnifyConfig())
    assert diag["diagnostics"][0][0]["mix_alpha"] == 0.0
    assert torch.equal(order[0, 0, :2], torch.tensor([3, 2]))


def test_uncovered_layerprop_defers_to_prp() -> None:
    mix = torch.tensor([[[0.4, 0.3, 0.2, 0.1]]])
    layerprop = PseudoSource(
        "layerprop",
        torch.tensor([[[3, 2, 1, 0]]]),
        coverage=torch.zeros(1, 1),
        hit_counts=torch.zeros(1, 1),
    )
    prp = PseudoSource(
        "prp",
        torch.tensor([[[0, 1, 2, 3]]]),
        coverage=torch.ones(1, 1),
    )
    order, diag = build_unify_ranking(mix, 2, [layerprop, prp], UnifyConfig(layerprop_tau=8.0))
    assert diag["diagnostics"][0][0]["layerprop_lambda"] == 0.0
    assert torch.equal(order[0, 0, :2], torch.tensor([0, 1]))


def test_pp_is_rejected() -> None:
    mix = torch.tensor([[[0.4, 0.3, 0.2, 0.1]]])
    pp = PseudoSource("pp", torch.tensor([[[0, 1, 2, 3]]]))
    try:
        build_unify_ranking(mix, 2, [pp])
    except ValueError as error:
        assert "PP" in str(error)
    else:
        raise AssertionError("expected PP to be rejected")
