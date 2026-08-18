import torch

from NAPS_v2.puzzlecomp import (
    PuzzleCompConfig,
    build_storage_plan,
    pairwise_puzzle_compensate,
    puzzle_merge_weight_pair,
    split_ranked_channels,
)


def test_storage_plan_preserves_pair_budget() -> None:
    plan = build_storage_plan(768, 384, 0.05)

    assert plan.reserve_channels == 38
    assert plan.core_channels_per_expert == 346
    assert plan.shared_residual_channels_per_pair == 76
    assert plan.effective_channels_per_expert == 422
    assert plan.stored_channels_per_pair == 768
    assert plan.materialized_channels_per_pair == 844
    assert plan.stored_channels_per_pair == 2 * plan.retained_width


def test_split_ranked_channels_is_fixed_width() -> None:
    plan = build_storage_plan(32, 16, 0.25)
    core, sacrificed, pruned = split_ranked_channels(torch.arange(32), plan)

    assert core.tolist() == list(range(8))
    assert sacrificed.tolist() == list(range(8, 16))
    assert pruned.tolist() == list(range(16, 32))


def test_dual_mask_keeps_shared_magnitude_and_expert_specific_saliency() -> None:
    config = PuzzleCompConfig(similarity_threshold=0.2)
    left = torch.tensor([[2.0, -2.0], [1.0, 4.0]])
    right = torch.tensor([[2.1, 2.0], [8.0, 1.0]])
    left_saliency = torch.tensor([[3.0, 1.0], [1.0, 5.0]])
    right_saliency = torch.tensor([[1.0, 3.0], [6.0, 1.0]])

    result = puzzle_merge_weight_pair(left, right, left_saliency, right_saliency, config)

    assert result["similarity_mask"][0, 0]
    assert torch.allclose(result["left_reconstructed"][0, 0], torch.tensor(2.05))
    assert torch.allclose(result["right_reconstructed"][0, 0], torch.tensor(2.05))
    assert result["left_mask"][0, 1]
    assert result["right_mask"][0, 1]
    assert not result["left_mask"][1, 0]
    assert result["right_mask"][1, 0]
    assert result["left_mask"][1, 1]
    assert not result["right_mask"][1, 1]


def test_pairwise_puzzle_compensation_returns_fixed_width() -> None:
    torch.manual_seed(11)
    source_width, retained_width, hidden_size = 32, 16, 8
    config = PuzzleCompConfig(similarity_threshold=0.5, reserve_fraction=0.25)
    ranking = torch.arange(source_width)
    left_gate = torch.randn(source_width, hidden_size)
    left_up = torch.randn(source_width, hidden_size)
    left_down = torch.randn(hidden_size, source_width)
    right_gate = left_gate + 0.01 * torch.randn_like(left_gate)
    right_up = left_up + 0.01 * torch.randn_like(left_up)
    right_down = left_down + 0.01 * torch.randn_like(left_down)
    probes = torch.randn(12, hidden_size)

    result = pairwise_puzzle_compensate(
        left_gate,
        left_up,
        left_down,
        right_gate,
        right_up,
        right_down,
        ranking,
        ranking,
        retained_width,
        probes,
        probes,
        probes,
        probes,
        torch.ones(probes.shape[0]),
        torch.ones(probes.shape[0]),
        config=config,
    )

    effective_width = retained_width + 8
    assert result["left"]["gate"].shape == (effective_width, hidden_size)
    assert result["left"]["up"].shape == (effective_width, hidden_size)
    assert result["left"]["down"].shape == (hidden_size, effective_width)
    assert result["right"]["gate"].shape == (effective_width, hidden_size)
    assert result["candidate_left"]["gate"].shape == (effective_width, hidden_size)
    assert result["candidate_right"]["down"].shape == (hidden_size, effective_width)
    assert result["diagnostics"]["storage_plan"]["stored_channels_per_pair"] == 2 * retained_width
    assert torch.isfinite(result["left"]["gate"]).all()
    assert torch.isfinite(result["right"]["down"]).all()


def test_pairwise_puzzle_compensation_rejects_insufficient_active_residuals() -> None:
    torch.manual_seed(12)
    source_width, retained_width, hidden_size = 32, 16, 8
    ranking = torch.arange(source_width)
    gate = torch.randn(source_width, hidden_size)
    up = torch.randn(source_width, hidden_size)
    down = torch.randn(hidden_size, source_width)
    probes = torch.randn(8, hidden_size)
    zero_mask = torch.zeros(source_width, dtype=torch.bool)
    zero_mask[retained_width:] = True

    result = pairwise_puzzle_compensate(
        gate,
        up,
        down,
        gate.clone(),
        up.clone(),
        down.clone(),
        ranking,
        ranking,
        retained_width,
        probes,
        probes,
        left_zero_mask=zero_mask,
        right_zero_mask=zero_mask,
    )

    assert not result["accepted"]
    assert result["diagnostics"]["fallback_reason"] == "insufficient_shared_channels"
    assert result["left"]["gate"].shape == (retained_width, hidden_size)