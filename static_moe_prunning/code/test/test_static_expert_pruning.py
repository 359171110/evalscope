from __future__ import annotations

import pytest
import torch

from src.channel_runtime import LayerChannelTable, channel_layer_to_device
from src.runtime_pruner import compute_moe_weighted_hidden_states, route_qwen3_topk
from src.static_expert_pruning import (
    aggregate_route_count_folds,
    allocate_compute_calibrated_prefix_widths,
    allocate_fold_constrained_prefix_widths,
    allocate_route_envelope_constrained_prefix_widths,
    allocate_static_prefix_widths_per_layer,
    build_reference_centered_route_envelope_costs,
    build_layer_routing_entropy_prior,
    StaticExpertRuntimeStats,
    _static_expert_core,
    allocate_static_prefix_widths,
    build_protected_min_widths,
    build_static_profile,
    build_static_block_values,
    gather_static_widths,
    profile_widths_by_layer,
    patch_qwen3_moe_blocks_static_expert,
    validate_static_profile_payload,
)


class _FixedRouter(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.new_tensor([[9.0, 8.0, 7.0, 6.0]]).expand(
            hidden_states.shape[0], -1
        )


def test_reap_router_mask_excludes_deleted_experts_before_topk() -> None:
    router_logits, routing_weights, selected_experts = route_qwen3_topk(
        _FixedRouter(),
        torch.ones(2, 3),
        top_k=2,
        norm_topk_prob=True,
        retained_expert_mask=torch.tensor([False, True, False, True]),
    )

    assert router_logits.shape == (2, 4)
    assert selected_experts.tolist() == [[1, 3], [1, 3]]
    assert torch.allclose(routing_weights.sum(dim=-1), torch.ones(2))


def test_reap_router_mask_matches_physically_pruned_expert_mixture() -> None:
    hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    experts = torch.nn.ModuleList([_Expert(), _Expert(), _Expert(), _Expert()])
    router = _Linear(
        torch.tensor(
            [
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.5],
            ]
        )
    )
    retained = torch.tensor([False, True, False, True])
    _, runtime_weights, runtime_selected = route_qwen3_topk(
        router,
        hidden_states,
        top_k=2,
        norm_topk_prob=True,
        retained_expert_mask=retained,
    )
    runtime_output, _, _ = compute_moe_weighted_hidden_states(
        hidden_states,
        experts,
        runtime_selected,
        runtime_weights,
    )

    pruned_router = _Linear(router.weight.detach()[retained].clone())
    pruned_experts = torch.nn.ModuleList([experts[1], experts[3]])
    _, pruned_weights, pruned_selected = route_qwen3_topk(
        pruned_router,
        hidden_states,
        top_k=2,
        norm_topk_prob=True,
    )
    pruned_output, _, _ = compute_moe_weighted_hidden_states(
        hidden_states,
        pruned_experts,
        pruned_selected,
        pruned_weights,
    )

    assert torch.allclose(runtime_output, pruned_output)


def test_reference_centered_route_envelope_uses_lower_cost_below_reference() -> None:
    route_folds = torch.tensor(
        [
            [[4.0, 5.0, 1.0]],
            [[2.0, 3.0, 5.0]],
        ]
    )
    reference = torch.tensor([[1, 1, 0]])

    costs, audit = build_reference_centered_route_envelope_costs(
        route_folds,
        reference,
        num_blocks=2,
        expansion=0.0,
    )

    assert costs.shape == (1, 1, 3, 2)
    assert costs[0, 0, 0].tolist() == pytest.approx([0.2, 0.4])
    assert costs[0, 0, 1].tolist() == pytest.approx([0.3, 0.5])
    assert costs[0, 0, 2].tolist() == pytest.approx([0.5, 0.5])
    assert audit["observed_fold_count"] == 2
    assert audit["expansion"] == pytest.approx(0.0)


def test_route_envelope_rejects_cross_fold_coordinate_mix() -> None:
    values = torch.tensor(
        [
            [
                [10.0, 9.0],
                [8.0, 7.0],
                [1.0, 0.5],
            ]
        ]
    )
    route_folds = torch.tensor(
        [
            [[4.0, 5.0, 1.0]],
            [[2.0, 3.0, 5.0]],
        ]
    )
    reference = torch.tensor([[1, 1, 0]])

    nominal_widths, nominal_audit = allocate_fold_constrained_prefix_widths(
        values,
        route_folds,
        reference,
        total_blocks=2,
        dual_iterations=256,
    )
    robust_widths, robust_audit = allocate_route_envelope_constrained_prefix_widths(
        values,
        route_folds,
        reference,
        total_blocks=2,
        envelope_expansion=0.0,
        dual_iterations=256,
    )

    assert nominal_widths.tolist() == [[2, 0, 0]]
    assert nominal_audit["all_fold_constraints_satisfied"] is True
    assert robust_widths.tolist() == [[1, 1, 0]]
    assert robust_audit["all_fold_constraints_satisfied"] is True
    assert robust_audit["route_envelope"]["constraint_relative_delta"] <= 1.0e-12


def test_fold_constrained_allocator_repairs_cross_fold_compute_violation() -> None:
    values = torch.tensor(
        [
            [
                [10.0, 9.0],
                [8.0, 7.0],
            ]
        ]
    )
    route_folds = torch.tensor(
        [
            [[10.0, 1.0]],
            [[1.0, 10.0]],
        ]
    )
    reference = torch.tensor([[1, 1]])

    widths, audit = allocate_fold_constrained_prefix_widths(
        values,
        route_folds,
        reference,
        total_blocks=2,
        dual_iterations=256,
    )

    assert widths.tolist() == [[1, 1]]
    assert int(widths.sum()) == 2
    assert audit["all_fold_constraints_satisfied"] is True
    assert max(audit["candidate_minus_reference_retained_cost_by_fold"]) <= 1.0e-12


def test_fold_constrained_allocator_reports_infeasible_hard_floor() -> None:
    values = torch.tensor(
        [
            [
                [10.0, 9.0],
                [8.0, 7.0],
            ]
        ]
    )
    route_folds = torch.tensor(
        [
            [[10.0, 1.0]],
            [[1.0, 10.0]],
        ]
    )
    reference = torch.tensor([[1, 1]])

    widths, audit = allocate_fold_constrained_prefix_widths(
        values,
        route_folds,
        reference,
        total_blocks=2,
        min_widths=torch.tensor([[2, 0]]),
        dual_iterations=128,
    )

    assert widths.tolist() == [[2, 0]]
    assert audit["all_fold_constraints_satisfied"] is False
    assert audit["maximum_relative_fold_violation"] > 0.0


def test_route_fold_aggregation_normalizes_mean_and_worst_case() -> None:
    folds = torch.tensor(
        [
            [[9.0, 1.0]],
            [[1.0, 9.0]],
        ]
    )
    mean, mean_audit = aggregate_route_count_folds(folds, aggregation="mean")
    worst, worst_audit = aggregate_route_count_folds(folds, aggregation="worst_case")
    assert torch.allclose(mean, torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    assert torch.allclose(worst, torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    assert mean_audit["fold_count"] == 2
    assert worst_audit["aggregation"] == "worst_case"


def test_route_fold_cvar_selects_upper_tail_and_renormalizes() -> None:
    folds = torch.tensor(
        [
            [[9.0, 1.0]],
            [[8.0, 2.0]],
            [[7.0, 3.0]],
            [[6.0, 4.0]],
        ]
    )
    robust, audit = aggregate_route_count_folds(
        folds, aggregation="cvar", cvar_alpha=0.5
    )
    assert robust.shape == (1, 2)
    assert float(robust[0, 0]) > float(robust[0, 1])
    assert float(robust.sum()) == pytest.approx(1.0)
    assert audit["cvar_alpha"] == pytest.approx(0.5)


def test_route_fold_aggregation_rejects_invalid_cvar_alpha() -> None:
    with pytest.raises(ValueError, match="cvar_alpha"):
        aggregate_route_count_folds(torch.ones((2, 1, 2)), aggregation="cvar", cvar_alpha=1.0)


def test_layer_entropy_prior_is_mean_one_and_protects_high_entropy_layers() -> None:
    distribution = torch.tensor([[0.45, 0.05], [0.25, 0.25]], dtype=torch.float64)
    prior, audit = build_layer_routing_entropy_prior(distribution, gamma=1.0)
    assert float(prior.mean()) == pytest.approx(1.0)
    assert float(prior[1]) > float(prior[0])
    assert audit["gamma"] == pytest.approx(1.0)


class _Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states, self.weight)


class _Expert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        eye = torch.eye(4)
        self.gate_proj = _Linear(eye)
        self.up_proj = _Linear(eye)
        self.down_proj = _Linear(eye)
        self.act_fn = torch.nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.linear(hidden_states, self.gate_proj.weight)
        up = torch.nn.functional.linear(hidden_states, self.up_proj.weight)
        return torch.nn.functional.linear(gate * up, self.down_proj.weight)


class _Qwen2StyleSparseMoe(torch.nn.Module):
    """Minimal Qwen2-MoE contract: shared expert and (hidden, router_logits)."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = _Linear(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
        self.experts = torch.nn.ModuleList([_Expert(), _Expert()])
        self.shared_expert = _Expert()
        self.shared_expert_gate = _Linear(torch.zeros(1, 4))
        self.top_k = 2
        self.norm_topk_prob = False

    def forward(self, hidden_states: torch.Tensor):
        batch, sequence, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(flat)
        routing_weights = torch.softmax(router_logits.float(), dim=-1).to(flat.dtype)
        selected = torch.arange(2).view(1, 2).expand(flat.shape[0], -1)
        output = sum(
            routing_weights[:, expert_idx : expert_idx + 1]
            * self.experts[expert_idx](flat)
            for expert_idx in range(2)
        )
        shared = torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return (output + shared).reshape(batch, sequence, hidden_dim), router_logits


class _Qwen2StyleLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Qwen2StyleSparseMoe()


class _Qwen2StyleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Qwen2StyleLayer()])


def _channel_table() -> LayerChannelTable:
    return LayerChannelTable(
        ranked_indices=torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
        block_relative_scores=torch.ones(2, 2),
        block_coverage_scores=torch.full((2, 2), 0.5),
        block_sizes=torch.tensor([2, 2]),
        intermediate_size=4,
    )


def test_static_allocator_enforces_exact_budget_and_prefix_order() -> None:
    values = torch.tensor(
        [
            [
                [1.0, 0.8, 0.7],
                [0.9, 0.1, 0.05],
            ]
        ]
    )

    widths = allocate_static_prefix_widths(
        values,
        total_blocks=4,
        min_blocks_per_expert=1,
    )

    assert widths.tolist() == [[3, 1]]
    assert int(widths.sum().item()) == 4


def test_static_allocator_respects_per_expert_caps() -> None:
    values = torch.tensor(
        [
            [
                [1.0, 0.9, 0.8],
                [0.7, 0.6, 0.5],
            ]
        ]
    )
    caps = torch.tensor([[2, 3]])

    widths = allocate_static_prefix_widths(
        values,
        total_blocks=5,
        min_blocks_per_expert=1,
        max_widths=caps,
    )

    assert widths.tolist() == [[2, 3]]


def test_static_allocator_respects_per_expert_minimum_widths() -> None:
    values = torch.tensor(
        [
            [
                [1.0, 0.9, 0.8],
                [0.7, 0.6, 0.5],
                [0.4, 0.3, 0.2],
            ]
        ]
    )
    minimums = torch.tensor([[0, 2, 1]])

    widths = allocate_static_prefix_widths(
        values,
        total_blocks=5,
        min_widths=minimums,
    )

    assert widths.tolist() == [[2, 2, 1]]
    assert int(widths.sum().item()) == 5


def test_static_allocator_rejects_minimum_width_above_cap() -> None:
    values = torch.ones(1, 2, 3)

    with pytest.raises(ValueError, match="min_widths"):
        allocate_static_prefix_widths(
            values,
            total_blocks=4,
            min_widths=torch.tensor([[3, 1]]),
            max_widths=torch.tensor([[2, 3]]),
        )


def test_protected_minimums_use_physical_layer_and_expert_ids() -> None:
    floors = build_protected_min_widths(
        num_layers=4,
        num_experts=5,
        num_blocks=12,
        protected_experts=[(1, 3, 12), (2, 4, 6)],
    )

    assert floors.shape == (4, 5)
    assert int(floors.sum()) == 18
    assert int(floors[1, 3]) == 12
    assert int(floors[2, 4]) == 6


def test_protected_minimums_reject_duplicate_physical_experts() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_protected_min_widths(
            num_layers=2,
            num_experts=3,
            num_blocks=4,
            protected_experts=[(1, 2, 3), (1, 2, 4)],
        )


def test_static_allocator_rejects_non_monotone_prefix_values() -> None:
    values = torch.tensor([[[1.0, 0.5, 0.7]]])

    with pytest.raises(ValueError, match="non-increasing"):
        allocate_static_prefix_widths(values, total_blocks=2)


def test_per_layer_allocator_enforces_each_layer_budget_exactly() -> None:
    values = torch.tensor(
        [
            [[10.0, 9.0], [8.0, 1.0]],
            [[7.0, 6.0], [5.0, 4.0]],
        ]
    )

    widths = allocate_static_prefix_widths_per_layer(
        values,
        total_blocks_by_layer=torch.tensor([2, 3]),
    )

    assert widths.tolist() == [[2, 0], [2, 1]]
    assert widths.sum(dim=1).tolist() == [2, 3]


def test_per_layer_allocator_rejects_infeasible_floor_budget() -> None:
    values = torch.ones(2, 2, 3)

    with pytest.raises(ValueError, match="layer 1"):
        allocate_static_prefix_widths_per_layer(
            values,
            total_blocks_by_layer=torch.tensor([3, 2]),
            min_widths=torch.tensor([[1, 1], [2, 1]]),
        )


def test_compute_calibrated_allocator_matches_low_compute_anchor() -> None:
    values = torch.tensor([[[10.0, 9.0], [2.0, 1.0]]])
    routes = torch.tensor([[10.0, 1.0]])

    widths, audit = allocate_compute_calibrated_prefix_widths(
        values,
        routes,
        total_blocks=2,
        target_routed_pruning_ratio=1.0 - 2.0 / 22.0,
    )

    assert widths.tolist() == [[0, 2]]
    assert int(widths.sum()) == 2
    assert audit["achieved_train_routed_pruning_ratio"] == pytest.approx(
        1.0 - 2.0 / 22.0
    )
    assert audit["target_bracketed"] is True


def test_compute_calibrated_allocator_preserves_floors_and_exact_structure() -> None:
    values = torch.tensor([[[5.0, 4.0], [3.0, 2.0], [1.0, 0.5]]])
    routes = torch.tensor([[8.0, 2.0, 1.0]])

    widths, audit = allocate_compute_calibrated_prefix_widths(
        values,
        routes,
        total_blocks=3,
        target_routed_pruning_ratio=0.5,
        min_widths=torch.tensor([[1, 0, 0]]),
    )

    assert int(widths.sum()) == 3
    assert int(widths[0, 0]) >= 1
    assert 0.0 <= audit["achieved_train_routed_pruning_ratio"] <= 1.0


def test_compute_calibrated_allocator_rejects_invalid_route_counts() -> None:
    values = torch.ones((1, 2, 2))

    with pytest.raises(ValueError, match="route_counts"):
        allocate_compute_calibrated_prefix_widths(
            values,
            torch.ones((2, 1)),
            total_blocks=2,
            target_routed_pruning_ratio=0.5,
        )


def test_gather_static_widths_depends_on_physical_expert_not_router_rank() -> None:
    layer_widths = torch.tensor([1, 4, 2, 3])
    selected = torch.tensor([[2, 0], [0, 2]])

    gathered = gather_static_widths(layer_widths, selected)

    assert gathered.tolist() == [[2, 1], [1, 2]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device-boundary regression")
def test_gather_static_widths_moves_cpu_profile_to_router_device() -> None:
    layer_widths = torch.tensor([1, 4, 2, 3], device="cpu")
    selected = torch.tensor([[2, 0]], device="cuda")

    gathered = gather_static_widths(layer_widths, selected)

    assert gathered.device == selected.device
    assert gathered.cpu().tolist() == [[2, 1]]


def test_route_rms_values_include_usage_but_preserve_prefix_monotonicity() -> None:
    coverage = torch.tensor(
        [
            [
                [0.6, 0.3, 0.1],
                [0.5, 0.3, 0.2],
            ]
        ]
    )
    route_counts = torch.tensor([[100.0, 25.0]])

    values = build_static_block_values(
        coverage,
        route_counts=route_counts,
        mode="route_rms",
    )

    assert values.shape == coverage.shape
    assert torch.all(values[..., :-1] >= values[..., 1:])
    assert torch.allclose(values[0, 0] / values[0, 1], torch.tensor([4.8, 4.0, 2.0]))


def test_dual_route_rms_combines_static_priors_geometrically() -> None:
    coverage = torch.tensor([[[0.7, 0.3], [0.7, 0.3]]])
    route_counts = torch.tensor([[10.0, 10.0]])
    amp = torch.tensor([[4.0, 1.0]])
    aimer = torch.tensor([[1.0, 1.0]])

    values = build_static_block_values(
        coverage,
        route_counts=route_counts,
        amp=amp,
        aimer=aimer,
        mode="dual_route_rms",
    )

    assert torch.allclose(values[0, 0], 2.0 * values[0, 1])


def test_static_core_full_width_matches_dense_expert_mixture() -> None:
    output, aux = _static_expert_core(
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.nn.ModuleList([_Expert(), _Expert()]),
        torch.tensor([[1, 0]]),
        torch.tensor([[0.3, 0.7]]),
        torch.tensor([2, 2]),
        _channel_table(),
        correction_mode="none",
    )

    assert torch.allclose(output, torch.tensor([[1.0, 4.0, 9.0, 16.0]]))
    assert aux["widths"].tolist() == [[2, 2]]
    assert bool(aux["full_mask"].all())


def test_static_core_uses_physical_width_and_supports_zero_width() -> None:
    _, aux = _static_expert_core(
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.nn.ModuleList([_Expert(), _Expert()]),
        torch.tensor([[1, 0]]),
        torch.tensor([[0.3, 0.7]]),
        torch.tensor([0, 1]),
        _channel_table(),
        correction_mode="none",
    )

    assert aux["widths"].tolist() == [[1, 0]]
    assert aux["block_keep_mask"].tolist() == [[[True, False], [False, False]]]


def test_channel_layer_to_device_preserves_channel_metadata() -> None:
    channel_layer = _channel_table()

    moved = channel_layer_to_device(channel_layer, "cpu")

    assert moved.intermediate_size == channel_layer.intermediate_size
    assert moved.ranked_indices.device.type == "cpu"
    assert torch.equal(moved.ranked_indices, channel_layer.ranked_indices)
    assert torch.equal(moved.block_sizes, channel_layer.block_sizes)


def test_qwen2_moe_patch_preserves_router_logits_output_contract() -> None:
    model = _Qwen2StyleModel()
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    with patch_qwen3_moe_blocks_static_expert(
        model,
        profile_widths={0: torch.tensor([2, 2])},
        channel_table={0: _channel_table()},
        correction_mode="none",
    ):
        output = model.model.layers[0].mlp(hidden)

    assert isinstance(output, tuple)
    assert len(output) == 2
    pruned_hidden, router_logits = output
    assert pruned_hidden.shape == hidden.shape
    assert router_logits.shape == (1, 2)


def test_static_patch_excludes_zero_width_experts_before_topk() -> None:
    model = _Qwen2StyleModel()
    model.model.layers[0].mlp.top_k = 1
    stats = StaticExpertRuntimeStats(
        profile_widths=torch.tensor([[2, 0]]),
        num_blocks=2,
    )

    with patch_qwen3_moe_blocks_static_expert(
        model,
        profile_widths={0: torch.tensor([2, 0])},
        channel_table={0: _channel_table()},
        correction_mode="none",
        runtime_stats=stats,
    ):
        model.model.layers[0].mlp(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))

    assert stats.aggregate_width_histogram() == {2: 1}


def test_full_width_patch_uses_exact_native_forward_and_records_zero_pruning() -> None:
    model = _Qwen2StyleModel()
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    expected = model.model.layers[0].mlp(hidden)
    stats = StaticExpertRuntimeStats(
        profile_widths=torch.tensor([[2, 2]]),
        num_blocks=2,
    )

    with patch_qwen3_moe_blocks_static_expert(
        model,
        profile_widths={0: torch.tensor([2, 2])},
        channel_table={0: _channel_table()},
        correction_mode="none",
        runtime_stats=stats,
    ):
        actual = model.model.layers[0].mlp(hidden)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert stats.routed_pruning_ratio() == 0.0


def test_static_runtime_stats_distinguish_structural_and_routed_compute() -> None:
    stats = StaticExpertRuntimeStats(
        profile_widths=torch.tensor([[0, 2], [1, 1]]),
        num_blocks=2,
    )
    stats.update(0, torch.tensor([[2, 2], [0, 2]]))

    assert stats.structural_pruning_ratio() == pytest.approx(0.5)
    assert stats.routed_pruning_ratio() == pytest.approx(0.25)
    assert stats.aggregate_width_histogram() == {0: 1, 2: 3}


def test_uniform_profile_is_balanced_and_budget_exact() -> None:
    coverage = torch.full((2, 3, 4), 0.25)

    profile = build_static_profile(
        coverage,
        mode="uniform",
        total_blocks=12,
    )

    assert profile.tolist() == [[2, 2, 2], [2, 2, 2]]
    assert int(profile.sum()) == 12


def test_profile_width_mapping_preserves_layer_ids() -> None:
    widths = torch.tensor([[1, 2], [3, 4]])

    mapping = profile_widths_by_layer(widths, layer_ids=[7, 11])

    assert sorted(mapping) == [7, 11]
    assert mapping[7].tolist() == [1, 2]
    assert mapping[11].tolist() == [3, 4]


def test_profile_validator_rejects_test_calibration_and_budget_tampering() -> None:
    base = {
        "schema_version": 1,
        "profile_construction": "calibrated",
        "calibration_split": "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": [0],
        "num_layers": 1,
        "num_experts": 2,
        "num_blocks": 3,
        "profile_widths": torch.tensor([[1, 2]]),
        "total_blocks": 3,
        "maximum_blocks": 6,
    }

    validate_static_profile_payload(base)
    with pytest.raises(ValueError, match="train split"):
        validate_static_profile_payload({**base, "calibration_split": "test"})
    with pytest.raises(ValueError, match="total_blocks"):
        validate_static_profile_payload({**base, "total_blocks": 4})


def test_validate_static_profile_payload_accepts_calibration_free_profile() -> None:
    widths = torch.tensor([[2, 0]], dtype=torch.long)
    payload = {
        "schema_version": 1,
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "profile_widths": widths,
        "num_layers": 1,
        "num_experts": 2,
        "layer_ids": [0],
        "num_blocks": 2,
        "total_blocks": 2,
        "maximum_blocks": 4,
    }

    assert torch.equal(validate_static_profile_payload(payload), widths)
