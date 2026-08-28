from __future__ import annotations

import pytest
import torch

from CSP.csp_core import (
    canonical_structural_score,
    canonical_structural_score_packed,
    allocate_participation_widths,
    participation_block_spectrum,
    rank_channels_by_csp,
    ranking_table,
    retained_prefix,
    validate_rankings,
)


def test_csp_canonicalization_is_invariant_to_function_preserving_up_down_rescaling() -> None:
    gate = torch.randn(8, 5)
    up = torch.randn(8, 5)
    down = torch.randn(5, 8)
    original = canonical_structural_score(gate, up, down, canonicalize=True)
    rescaled = canonical_structural_score(gate, up * 7.0, down / 7.0, canonicalize=True)
    assert torch.allclose(original, rescaled, rtol=1.0e-5, atol=1.0e-7)


def test_csp_canonicalization_is_disabled_by_default() -> None:
    gate = torch.randn(8, 5)
    up = torch.randn(8, 5)
    down = torch.randn(5, 8)
    raw = canonical_structural_score(gate, up, down)
    canonical = canonical_structural_score(gate, up, down, canonicalize=True)
    assert not torch.allclose(raw, canonical)


def test_csp_applies_input_scale_to_both_input_directions() -> None:
    gate = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    up = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    down = torch.ones(2, 2)
    scale = torch.tensor([3.0, 1.0])
    assert not torch.allclose(
        canonical_structural_score(gate, up, down),
        canonical_structural_score(gate, up, down, input_scale=scale),
    )


def test_csp_packed_matches_split_and_dead_path_is_excluded() -> None:
    gate = torch.arange(24, dtype=torch.float32).reshape(8, 3) + 1
    up = torch.arange(24, dtype=torch.float32).reshape(8, 3) + 10
    down = torch.arange(24, dtype=torch.float32).reshape(3, 8) + 20
    packed = torch.cat((gate, up), dim=0)
    assert torch.allclose(
        canonical_structural_score_packed(packed, down),
        canonical_structural_score(gate, up, down),
    )
    dead = canonical_structural_score(gate, torch.zeros_like(up), down)
    assert bool(torch.isneginf(dead).all())


def test_csp_ranking_ties_and_complete_table_are_deterministic() -> None:
    scores = torch.tensor([[1.0, 3.0, 3.0, 2.0], [0.0, 0.0, 0.0, 0.0]])
    ranked = rank_channels_by_csp(scores)
    assert torch.equal(ranked[0], torch.tensor([1, 2, 3, 0]))
    assert torch.equal(retained_prefix(ranked[0], 2), torch.tensor([1, 2]))
    table = {0: ranking_table(scores, 2)}
    validate_rankings(table, 1, 2, 4, layer_ids=(0,))


def test_csp_handles_large_weights_without_product_overflow() -> None:
    gate = torch.full((2, 3), 1.0e20)
    up = torch.full((2, 3), 2.0e20)
    down = torch.full((3, 2), 3.0e20)
    scores = canonical_structural_score(gate, up, down)
    assert bool(torch.isfinite(scores).all())
    assert torch.allclose(scores[0], scores[1])


def test_csp_score_matches_explicit_canonical_signature() -> None:
    gate = torch.tensor([[1.0, 2.0], [0.5, 0.25]])
    up = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    down = torch.tensor([[5.0, 0.5], [6.0, 1.5]])
    scores = canonical_structural_score(
        gate, up, down, functional_viability_threshold=0.0, canonicalize=True
    )
    explicit = []
    for index in range(2):
        g = gate[index]
        u = up[index]
        d = down[:, index]
        alpha = torch.sqrt(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(u))
        signature = torch.cat((g, alpha * u, d / alpha)).abs()
        explicit.append(torch.log(torch.tensor(6.0) * signature.square().sum() / signature.abs().sum().square()))
    assert torch.allclose(scores, torch.stack(explicit).to(dtype=scores.dtype), rtol=1.0e-6, atol=1.0e-7)


def test_participation_allocator_keeps_exact_budget_and_prefers_concentrated_experts() -> None:
    scores = torch.tensor([
        [10.0, 9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        [10.0, 9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
    ])
    spectrum = participation_block_spectrum(scores, block_size=2)
    widths = allocate_participation_widths(
        spectrum,
        candidate_widths=(2, 4, 6),
        total_blocks=8,
        block_size=2,
    )

    assert widths.tolist() == [1, 3, 1, 3]
    assert int(widths.sum()) == 8


def test_participation_allocator_rejects_non_aligned_or_impossible_options() -> None:
    spectrum = torch.ones(2, 4)
    with pytest.raises(ValueError, match="aligned"):
        allocate_participation_widths(spectrum, candidate_widths=(1, 2), total_blocks=3, block_size=2)
    with pytest.raises(ValueError, match="exactly representable"):
        allocate_participation_widths(spectrum, candidate_widths=(1, 3), total_blocks=3)
