from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from NAPS_v2.build_naps_v2_artifacts import build_router_probe_route, rms_norm_rows
from NAPS_v2.build_naps_v2_heterogeneous import (
    assign_expert_widths,
    assign_expert_widths_adaptive,
    expert_aimer_score,
)
from NAPS_v2.reprofile_naps_v2_heterogeneous_aimer_pp import assign_expert_widths_aimer_pp
from NAPS_v2.export_naps_v2_heterogeneous_checkpoint import (
    pad_columns,
    pad_rows,
    padded_swiglu_expert_output,
    swiglu_expert_output,
)
from NAPS_v2.naps_v2_core import swiglu_response
from NAPS_v2.reprofile_naps_v2_heterogeneous import adaptive_widths_from_rankings
from NAPS_v2.reprofile_naps_v2_heterogeneous_gaussian import load_gaussian_assignments


class _Gemma4ProbeAdapter:
    model_family = "gemma4"
    router_top_k = 1
    text_config = {"rms_norm_eps": 1.0e-6}

    def router_name(self, layer_id: int) -> str:
        return "router"

    def router_scale_name(self, layer_id: int) -> str:
        return "router_scale"

    def router_per_expert_scale_name(self, layer_id: int) -> str:
        return "per_expert_scale"

    def expert_input_norm_name(self, layer_id: int) -> str:
        return "expert_norm"


def test_expert_aimer_score_matches_original_definition() -> None:
    gate = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    up = torch.tensor([[2.0, -1.0], [4.0, -3.0]])
    down = torch.tensor([[1.0, 2.0], [-3.0, -4.0]])
    values = torch.cat((gate.flatten(), up.flatten(), down.flatten()))
    expected = values.abs().mean() / values.square().mean().sqrt()
    assert torch.allclose(expert_aimer_score(gate, up, down), expected)


def test_width_assignment_is_stable_and_budget_preserving() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assigned = assign_expert_widths(scores, (2, 4, 6))
    assert assigned.tolist() == [6, 4, 4, 2]
    assert int(assigned.sum().item()) == 16


def test_width_assignment_adapts_to_aimer_score_clusters() -> None:
    scores = torch.tensor([10.0, 5.3, 5.2, 5.1, 5.0, 4.9, 4.8, 0.0])
    fixed = assign_expert_widths(scores, (2, 4, 6))
    assigned = assign_expert_widths_adaptive(scores, (2, 4, 6))
    assert fixed.tolist() == [2, 2, 4, 4, 4, 4, 6, 6]
    assert assigned.tolist() == [2, 4, 4, 4, 4, 4, 4, 6]
    assert int(assigned.sum().item()) == 32


def test_width_assignment_rejects_asymmetric_width_steps() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
    try:
        assign_expert_widths(scores, (2, 4, 8))
    except ValueError as error:
        assert str(error) == "widths must be symmetric around the medium width"
    else:
        raise AssertionError("asymmetric widths must be rejected")


def test_aimer_pp_rank_fusion_preserves_quartile_budget() -> None:
    aimer_scores = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.0])
    pp_rescue_counts = torch.tensor([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assigned, fused_rank = assign_expert_widths_aimer_pp(
        aimer_scores, pp_rescue_counts, (2, 4, 6)
    )
    assert assigned.tolist() == [4, 2, 2, 4, 4, 4, 6, 6]
    assert fused_rank.tolist() == [7.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert int(assigned.sum().item()) == 32


def test_aimer_pp_rank_fusion_falls_back_to_aimer_when_pp_is_tied() -> None:
    aimer_scores = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.0])
    pp_rescue_counts = torch.full((8,), 23.0)
    assigned, _ = assign_expert_widths_aimer_pp(
        aimer_scores, pp_rescue_counts, (2, 4, 6)
    )
    assert torch.equal(assigned, assign_expert_widths(aimer_scores, (2, 4, 6)))


def test_adaptive_widths_from_cached_rankings() -> None:
    rankings = {
        "table": {
            0: {"expert_aimer_scores": torch.tensor([10.0, 5.1, 5.0, 0.0])},
            1: {"expert_aimer_scores": torch.tensor([9.0, 8.9, 4.0, 0.0])},
        }
    }
    assigned = adaptive_widths_from_rankings(rankings, (2, 4, 6))
    assert assigned.tolist() == [[2, 4, 4, 6], [2, 4, 4, 6]]
    assert assigned.sum(dim=1).tolist() == [16, 16]


def test_gaussian_assignments_preserve_fixed_quartiles_and_budget(tmp_path: Path) -> None:
    assignment_csv = tmp_path / "quarter_assignment.csv"
    assignment_csv.write_text(
        "model_family,layer_id,expert_id,allocation,assigned_width\n"
        "qwen3,0,0,small,2\n"
        "qwen3,0,1,medium,4\n"
        "qwen3,0,2,medium,4\n"
        "qwen3,0,3,large,6\n"
        "qwen3,1,0,large,6\n"
        "qwen3,1,1,medium,4\n"
        "qwen3,1,2,medium,4\n"
        "qwen3,1,3,small,2\n",
        encoding="utf-8",
    )

    assigned = load_gaussian_assignments(assignment_csv, 2, 4, (2, 4, 6))

    assert assigned.tolist() == [[2, 4, 4, 6], [6, 4, 4, 2]]
    assert assigned.sum(dim=1).tolist() == [16, 16]


def test_padding_preserves_selected_values_and_zero_fills() -> None:
    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    columns = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    padded_rows = pad_rows(rows, 4)
    padded_columns = pad_columns(columns, 4)
    assert padded_rows.shape == (4, 2)
    assert padded_columns.shape == (2, 4)
    assert torch.equal(padded_rows[:2], rows)
    assert torch.equal(padded_columns[:, :2], columns)
    assert torch.count_nonzero(padded_rows[2:]) == 0
    assert torch.count_nonzero(padded_columns[:, 2:]) == 0


def test_padded_separate_expert_matches_narrow_expert() -> None:
    hidden_states = torch.randn(5, 6)
    gate = torch.randn(8, 6)
    up = torch.randn(8, 6)
    down = torch.randn(6, 8)
    retained = torch.tensor([6, 1, 4])

    narrow_output = swiglu_expert_output(
        hidden_states,
        gate.index_select(0, retained),
        up.index_select(0, retained),
        down.index_select(1, retained),
    )
    padded_output = padded_swiglu_expert_output(
        hidden_states, gate, up, down, retained, padded_width=5
    )

    assert torch.allclose(narrow_output, padded_output, rtol=1.0e-5, atol=1.0e-5)


def test_padded_fused_expert_matches_narrow_expert() -> None:
    hidden_states = torch.randn(5, 6)
    gate = torch.randn(8, 6)
    up = torch.randn(8, 6)
    down = torch.randn(6, 8)
    retained = torch.tensor([6, 1, 4])
    fused = torch.cat((gate, up), dim=0)
    padded_fused = torch.cat(
        (
            pad_rows(gate.index_select(0, retained), 5),
            pad_rows(up.index_select(0, retained), 5),
        ),
        dim=0,
    )

    narrow_output = swiglu_expert_output(
        hidden_states,
        fused[:8].index_select(0, retained),
        fused[8:].index_select(0, retained),
        down.index_select(1, retained),
    )
    padded_output = swiglu_expert_output(
        hidden_states, padded_fused[:5], padded_fused[5:], pad_columns(down.index_select(1, retained), 5)
    )

    assert torch.allclose(narrow_output, padded_output, rtol=1.0e-5, atol=1.0e-5)


def test_gemma4_gelu_tanh_response_matches_reference() -> None:
    probes = torch.randn(4, 6)
    gate = torch.randn(8, 6)
    up = torch.randn(8, 6)
    expected = F.gelu(probes @ gate.T, approximate="tanh") * (probes @ up.T)
    actual = swiglu_response(probes, gate, up, activation="gelu_pytorch_tanh")
    assert torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_gemma4_route_and_expert_norms_share_raw_probe_input() -> None:
    raw_probes = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    tensors = {
        "router": raw_probes,
        "router_scale": torch.tensor([1.0, 2.0]),
        "per_expert_scale": torch.ones(2),
        "expert_norm": torch.tensor([0.5, 1.5]),
    }

    with patch(
        "NAPS_v2.build_naps_v2_artifacts.load_tensor",
        side_effect=lambda model_path, weight_map, name: tensors[name],
    ):
        _, route_probes, expert_probes, _, _, _ = build_router_probe_route(
            Path("."), {}, _Gemma4ProbeAdapter(), 0, torch.device("cpu")
        )

    expected_route = rms_norm_rows(raw_probes, torch.ones(2), 1.0e-6)
    expected_expert = rms_norm_rows(raw_probes, tensors["expert_norm"], 1.0e-6)
    double_normalized = rms_norm_rows(expected_route, tensors["expert_norm"], 1.0e-6)
    assert torch.allclose(route_probes, expected_route)
    assert torch.allclose(expert_probes, expected_expert)
    assert not torch.equal(expert_probes, double_normalized)


def test_gemma4_padded_expert_matches_narrow_expert() -> None:
    hidden_states = torch.randn(5, 6)
    gate = torch.randn(8, 6)
    up = torch.randn(8, 6)
    down = torch.randn(6, 8)
    retained = torch.tensor([6, 1, 4])

    narrow_output = swiglu_expert_output(
        hidden_states,
        gate.index_select(0, retained),
        up.index_select(0, retained),
        down.index_select(1, retained),
        activation="gelu_pytorch_tanh",
    )
    padded_output = padded_swiglu_expert_output(
        hidden_states,
        gate,
        up,
        down,
        retained,
        padded_width=5,
        activation="gelu_pytorch_tanh",
    )
    assert torch.allclose(narrow_output, padded_output, rtol=1.0e-5, atol=1.0e-5)
