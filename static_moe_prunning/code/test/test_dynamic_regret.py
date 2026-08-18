from __future__ import annotations

import torch

from src.dynamic_regret import (
    build_apa_teacher_parent_score,
    compute_dynamic_regret_batch,
    scatter_physical_expert_blocks,
)


def test_apa_teacher_parent_score_protects_top1_and_is_token_conditioned() -> None:
    gate = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]])
    amp = torch.ones_like(gate)
    aimer = torch.ones_like(gate)

    score = build_apa_teacher_parent_score(gate, amp, aimer)

    assert score.shape == gate.shape
    assert score[0, 0] == 1.0
    assert score[1, 1] == 1.0
    assert not torch.equal(score[0], score[1])


def test_parent_score_components_are_explicit_and_distinct() -> None:
    gate = torch.tensor([[0.7, 0.2, 0.1]])
    amp = torch.tensor([[1.0, 4.0, 1.0]])
    aimer = torch.tensor([[1.0, 1.0, 4.0]])

    top_p = build_apa_teacher_parent_score(gate, amp, aimer, mode="top_p")
    dual = build_apa_teacher_parent_score(gate, amp, aimer, mode="dual")
    raw_gate = build_apa_teacher_parent_score(gate, amp, aimer, mode="gate")

    assert top_p[0, 0] == 1.0
    assert dual[0, 0] == 1.0
    assert torch.allclose(raw_gate, gate)
    assert not torch.allclose(top_p, dual)


def test_dynamic_regret_batch_uses_floor_free_exact_teacher_budget() -> None:
    result = compute_dynamic_regret_batch(
        gate=torch.tensor([[0.8, 0.2]]),
        selected_experts=torch.tensor([[2, 0]]),
        amp_layer=torch.ones(3),
        aimer_layer=torch.ones(3),
        block_coverage_layer=torch.tensor(
            [
                [0.6, 0.4],
                [0.6, 0.4],
                [0.6, 0.4],
            ]
        ),
        total_blocks=2,
        num_experts=3,
    )

    assert int(result.widths.sum()) == 2
    assert result.widths.tolist() == [[2, 0]]
    assert result.block_values.shape == (3, 2)
    assert torch.all(result.block_values[:, :-1] >= result.block_values[:, 1:])
    assert result.unconditional_block_values[0].sum() > 0
    assert result.block_values[0].sum() == 0


def test_scatter_dynamic_regret_uses_physical_expert_ids() -> None:
    selected = torch.tensor([[2, 0], [2, 1]])
    values = torch.tensor(
        [
            [[1.0, 0.5], [2.0, 0.0]],
            [[3.0, 1.5], [4.0, 2.0]],
        ]
    )

    scattered = scatter_physical_expert_blocks(selected, values, num_experts=3)

    assert scattered.tolist() == [[2.0, 0.0], [4.0, 2.0], [4.0, 2.0]]
