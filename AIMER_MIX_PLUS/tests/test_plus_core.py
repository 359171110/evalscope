from __future__ import annotations

import pytest
import torch

from AIMER_MIX_PLUS.plus_core import (
    AIMERMixPlusConfig,
    PseudoSource,
    build_plus_ranking,
    build_plus_ranking_from_order,
)


def _config() -> AIMERMixPlusConfig:
    return AIMERMixPlusConfig(
        boundary_fraction=0.5,
        minimum_boundary_channels=1,
        maximum_boundary_fraction=0.5,
        pseudo_floor=0.1,
    )


def test_no_sources_is_exact_aimer_mix_order() -> None:
    scores = torch.tensor([[[0.9, 0.7, 0.4, 0.1]]])
    order, diagnostics = build_plus_ranking(scores, retained_channels=2)
    assert torch.equal(order, torch.tensor([[[0, 1, 2, 3]]]))
    assert diagnostics["sources"] == []


def test_ignore_base_uses_only_pseudo_order() -> None:
    scores = torch.tensor([[[0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]]])
    source = PseudoSource(
        "layerprop",
        torch.tensor([[[7, 6, 5, 4, 3, 2, 1, 0]]]),
        coverage=torch.ones(1, 1),
        stability=torch.ones(1, 1),
    )
    config = AIMERMixPlusConfig(
        boundary_fraction=0.5,
        minimum_boundary_channels=1,
        maximum_boundary_fraction=0.5,
        ignore_base=True,
        base_boundary_weight=0.0,
        source_weights=(("layerprop", 1.0),),
    )
    order, diagnostics = build_plus_ranking(scores, 4, [source], config)
    assert torch.equal(order[0, 0, :4], torch.tensor([7, 6, 5, 4]))
    assert diagnostics["diagnostics"][0][0]["core_channels"] == 0
    assert diagnostics["config"]["ignore_base"] is True


def test_strong_layerprop_can_rescue_below_base_topk() -> None:
    scores = torch.tensor([[[0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]]])
    source = PseudoSource(
        "layerprop",
        torch.tensor([[[7, 6, 5, 4, 3, 2, 1, 0]]]),
        coverage=torch.ones(1, 1),
        stability=torch.ones(1, 1),
    )
    order, diagnostics = build_plus_ranking(scores, 4, [source], _config())
    assert any(int(channel) >= 4 for channel in order[0, 0, :4])
    assert diagnostics["diagnostics"][0][0]["swap_count"] > 0


def test_missing_source_does_not_disable_present_prp() -> None:
    scores = torch.tensor([[[0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]]])
    prp = PseudoSource(
        "prp",
        torch.tensor([[[7, 5, 3, 1, 6, 4, 2, 0]]]),
        coverage=torch.ones(1, 1),
        stability=torch.ones(1, 1),
    )
    order, diagnostics = build_plus_ranking(scores, 4, [prp], _config())
    assert diagnostics["diagnostics"][0][0]["active_sources"] == ["prp"]
    assert any(int(channel) >= 4 for channel in order[0, 0, :4])


def test_conflicting_sources_compete_without_fallback() -> None:
    scores = torch.tensor([[[0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]]])
    sources = [
        PseudoSource(
            "pp",
            torch.tensor([[[0, 2, 4, 6, 1, 3, 5, 7]]]),
            coverage=torch.tensor([[0.8]]),
            stability=torch.tensor([[0.9]]),
        ),
        PseudoSource(
            "prp",
            torch.tensor([[[7, 5, 3, 1, 6, 4, 2, 0]]]),
            coverage=torch.tensor([[0.7]]),
            stability=torch.tensor([[0.6]]),
        ),
    ]
    order, diagnostics = build_plus_ranking(scores, 4, sources, _config())
    assert torch.equal(torch.sort(order, dim=2).values, torch.arange(8).reshape(1, 1, 8))
    assert set(diagnostics["diagnostics"][0][0]["active_sources"]) == {"pp", "prp"}


def test_duplicate_source_names_are_rejected() -> None:
    scores = torch.ones(1, 1, 4)
    source = PseudoSource("pp", torch.tensor([[[0, 1, 2, 3]]]))
    with pytest.raises(ValueError, match="unique"):
        build_plus_ranking(scores, 2, [source, source], _config())


def test_ranking_only_base_is_preserved_without_sources() -> None:
    base = torch.tensor([[[2, 0, 3, 1]]])
    order, _diagnostics = build_plus_ranking_from_order(base, retained_channels=2)
    assert torch.equal(order, base)


def test_agreement_is_normalized_by_source_pairs() -> None:
    scores = torch.tensor([[[0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]]])
    sources = [
        PseudoSource(name, torch.tensor([[[7, 6, 5, 4, 3, 2, 1, 0]]]))
        for name in ("pp", "prp", "layerprop")
    ]
    _order, diagnostics = build_plus_ranking(scores, 4, sources, _config())
    record = diagnostics["diagnostics"][0][0]
    assert 0.0 <= record["agreement_mass"] <= float(record["rescue_channels"])


def test_low_confidence_source_has_lower_absolute_influence() -> None:
    scores = torch.tensor([[[0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]]])
    order = torch.tensor([[[7, 6, 5, 4, 3, 2, 1, 0]]])
    strong = PseudoSource("layerprop", order, coverage=torch.ones(1, 1), stability=torch.ones(1, 1))
    weak = PseudoSource(
        "layerprop",
        order,
        coverage=torch.full((1, 1), 0.1),
        stability=torch.ones(1, 1),
    )
    _strong_order, strong_diagnostics = build_plus_ranking(scores, 4, [strong], _config())
    _weak_order, weak_diagnostics = build_plus_ranking(scores, 4, [weak], _config())
    assert strong_diagnostics["diagnostics"][0][0]["pseudo_mass"] > (
        weak_diagnostics["diagnostics"][0][0]["pseudo_mass"]
    )


def test_layerprop_mix_lambda_is_coverage_gated() -> None:
    from AIMER_MIX_PLUS.plus_core import layerprop_mix_lambda

    assert layerprop_mix_lambda(0.0, 8.0) == 0.0
    assert layerprop_mix_lambda(8.0, 8.0) == 0.5
    assert abs(layerprop_mix_lambda(24.0, 8.0) - 0.75) < 1.0e-6
    assert layerprop_mix_lambda(float("inf"), 8.0) == 1.0


def test_adaptive_lp_prp_uses_hit_counts_not_fixed_weights() -> None:
    scores = torch.tensor([[[0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]]])
    layerprop = PseudoSource(
        "layerprop",
        torch.tensor([[[7, 6, 5, 4, 3, 2, 1, 0]]]),
        coverage=torch.ones(1, 1),
        hit_counts=torch.zeros(1, 1),
    )
    prp = PseudoSource(
        "prp",
        torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]]),
        coverage=torch.ones(1, 1),
        hit_counts=torch.ones(1, 1),
    )
    config = AIMERMixPlusConfig(
        boundary_fraction=1.0,
        minimum_boundary_channels=1,
        maximum_boundary_fraction=1.0,
        ignore_base=True,
        adaptive_lp_prp=True,
        layerprop_tau=8.0,
        source_weights=(("layerprop", 1.6), ("prp", 0.5)),
    )
    uncovered, uncovered_diag = build_plus_ranking(scores, 4, [layerprop, prp], config)
    assert uncovered_diag["diagnostics"][0][0]["layerprop_lambda"] == 0.0
    assert torch.equal(uncovered[0, 0, :4], torch.tensor([0, 1, 2, 3]))

    covered_lp = PseudoSource(
        "layerprop",
        layerprop.order,
        coverage=torch.ones(1, 1),
        hit_counts=torch.full((1, 1), 1.0e6),
    )
    covered, covered_diag = build_plus_ranking(scores, 4, [covered_lp, prp], config)
    assert covered_diag["diagnostics"][0][0]["layerprop_lambda"] > 0.99
    assert torch.equal(covered[0, 0, :4], torch.tensor([7, 6, 5, 4]))
