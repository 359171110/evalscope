from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from scripts.build_tail_risk_profile import main


def _save(path: Path, payload: dict) -> Path:
    torch.save(payload, path)
    return path


def test_profile_builder_can_decouple_tail_coverage_from_risk_floor_cache(
    tmp_path: Path, monkeypatch
) -> None:
    teacher = _save(
        tmp_path / "teacher.pt",
        {
            "model_path": "model",
            "dataset": "wikitext-2-raw-v1",
            "dataset_config": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "calibration_token_offset": 8,
            "calibration_token_end": 16,
            "calibration_source": {"source_type": "huggingface_dataset"},
            "test_metrics_used": False,
            "parent_mode": "dual",
            "unconditional_block_values": {0: torch.tensor([[4.0, 2.0], [3.0, 1.0]])},
        },
    )
    reference = _save(
        tmp_path / "reference.pt",
        {
            "split": "train",
            "sequence_length": 2048,
            "table": {0: {"block_coverage_scores": torch.ones(2, 2)}},
        },
    )
    tail = _save(
        tmp_path / "tail.pt",
        {
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "block_size": 64,
            "tail_lambda": 0.5,
            "score_mode": "tail",
            "table": {
                0: {"block_coverage_scores": torch.tensor([[2.0, 1.0], [1.5, 1.0]])}
            },
            "expert_tail_risk_proxy": {0: torch.tensor([10.0, 1.0])},
        },
    )
    risk = _save(
        tmp_path / "risk.pt",
        {
            "dataset": "allenai/c4",
            "split": "train",
            "sequence_length": 2048,
            "expert_tail_risk_proxy": {0: torch.tensor([1.0, 10.0])},
        },
    )
    output = tmp_path / "profile.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tail_risk_profile.py",
            "--teacher-cache",
            str(teacher),
            "--reference-channel-cache",
            str(reference),
            "--tail-channel-cache",
            str(tail),
            "--risk-floor-cache",
            str(risk),
            "--output-profile",
            str(output),
            "--target-pruning-ratio",
            "0.5",
            "--risk-floor-min-width",
            "1",
            "--risk-floor-early-layers",
            "1",
            "--risk-floor-quantile",
            "0.5",
            "--risk-floor-relative-max",
            "0.0",
        ],
    )

    assert main() == 0
    profile = torch.load(output, map_location="cpu", weights_only=True)

    assert profile["risk_floor"]["selected_experts"] == [
        {"layer": 0, "expert": 1, "risk": 10.0, "min_width": 1}
    ]
    assert profile["cache_provenance"]["channel"]["dataset"] == "wikitext-2-raw-v1"
    assert profile["cache_provenance"]["risk_floor"]["dataset"] == "allenai/c4"
    assert profile["cache_provenance"]["risk_floor"]["path"] == str(risk.resolve())
    assert len(profile["cache_provenance"]["risk_floor"]["sha256"]) == 64
    teacher_provenance = profile["cache_provenance"]["conditional_dual_teacher"]
    assert teacher_provenance["dataset"] == "wikitext-2-raw-v1"
    assert teacher_provenance["dataset_config"] == "wikitext-2-raw-v1"
    assert teacher_provenance["calibration_token_offset"] == 8
    assert teacher_provenance["calibration_token_end"] == 16
    assert teacher_provenance["calibration_source"] == {
        "source_type": "huggingface_dataset"
    }


def test_tail_risk_profile_builder_rejects_non_dual_teacher(
    tmp_path: Path, monkeypatch
) -> None:
    teacher = _save(
        tmp_path / "teacher.pt",
        {
            "model_path": "model",
            "dataset": "train",
            "split": "train",
            "sequence_length": 2048,
            "test_metrics_used": False,
            "parent_mode": "combined",
            "unconditional_block_values": {0: torch.ones(1, 1)},
        },
    )
    channel = _save(
        tmp_path / "channel.pt",
        {
            "dataset": "train",
            "split": "train",
            "sequence_length": 2048,
            "block_size": 64,
            "tail_lambda": 0.5,
            "table": {0: {"block_coverage_scores": torch.ones(1, 1)}},
            "expert_tail_risk_proxy": {0: torch.ones(1)},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tail_risk_profile.py",
            "--teacher-cache",
            str(teacher),
            "--reference-channel-cache",
            str(channel),
            "--tail-channel-cache",
            str(channel),
            "--output-profile",
            str(tmp_path / "profile.pt"),
            "--target-pruning-ratio",
            "0.5",
        ],
    )

    with pytest.raises(ValueError, match="parent_mode=dual"):
        main()


def test_tail_risk_profile_builder_enforces_exact_per_layer_budget(
    tmp_path: Path, monkeypatch
) -> None:
    teacher = _save(
        tmp_path / "teacher.pt",
        {
            "model_path": "model",
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "test_metrics_used": False,
            "parent_mode": "dual",
            "unconditional_block_values": {
                0: torch.tensor([[10.0, 9.0], [8.0, 7.0]]),
                1: torch.tensor([[4.0, 3.0], [2.0, 1.0]]),
            },
        },
    )
    channel = _save(
        tmp_path / "channel.pt",
        {
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "block_size": 64,
            "tail_lambda": 0.5,
            "score_mode": "tail",
            "table": {
                0: {"block_coverage_scores": torch.ones(2, 2)},
                1: {"block_coverage_scores": torch.ones(2, 2)},
            },
        },
    )
    output = tmp_path / "profile.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tail_risk_profile.py",
            "--teacher-cache",
            str(teacher),
            "--reference-channel-cache",
            str(channel),
            "--tail-channel-cache",
            str(channel),
            "--output-profile",
            str(output),
            "--target-pruning-ratio",
            "0.5",
            "--allocation-scope",
            "per_layer",
            "--retained-blocks-per-layer",
            "2",
        ],
    )

    assert main() == 0
    profile = torch.load(output, map_location="cpu", weights_only=True)

    assert profile["allocation_scope"] == "per_layer"
    assert profile["target_blocks_by_layer"] == [2, 2]
    assert profile["actual_blocks_by_layer"] == [2, 2]
    assert profile["profile_widths"].sum(dim=1).tolist() == [2, 2]


def test_profile_builder_aggregates_repeated_output_saliency_caches(
    tmp_path: Path, monkeypatch
) -> None:
    teacher = _save(
        tmp_path / "teacher.pt",
        {
            "model_path": "model",
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "test_metrics_used": False,
            "parent_mode": "dual",
            "unconditional_block_values": {0: torch.ones(2, 2)},
        },
    )
    channel = _save(
        tmp_path / "channel.pt",
        {
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "block_size": 64,
            "tail_lambda": 0.5,
            "score_mode": "tail",
            "table": {0: {"block_coverage_scores": torch.ones(2, 2)}},
            "expert_tail_risk_proxy": {0: torch.ones(2)},
        },
    )
    output_caches = []
    for index, values in enumerate(
        (torch.tensor([1.0, 3.0]), torch.tensor([30.0, 10.0]))
    ):
        output_caches.append(
            _save(
                tmp_path / f"output_{index}.pt",
                {
                    "split": "train",
                    "sequence_length": 2048,
                    "test_metrics_used": False,
                    "calibration_token_offset": index * 4096,
                    "calibration_token_end": (index + 1) * 4096,
                    "expert_output_saliency_mean": {0: values},
                    "output_saliency_formula": "mean_active_token(gate * l2_norm(expert_output))",
                },
            )
        )
    output = tmp_path / "profile.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tail_risk_profile.py",
            "--teacher-cache",
            str(teacher),
            "--reference-channel-cache",
            str(channel),
            "--tail-channel-cache",
            str(channel),
            "--output-profile",
            str(output),
            "--target-pruning-ratio",
            "0.5",
            "--output-saliency-cache",
            str(output_caches[0]),
            "--output-saliency-cache",
            str(output_caches[1]),
            "--output-saliency-beta",
            "1.0",
        ],
    )

    assert main() == 0
    profile = torch.load(output, map_location="cpu", weights_only=True)
    provenance = profile["output_saliency_provenance"]
    assert provenance["fold_count"] == 2
    assert provenance["aggregation"] == (
        "per_fold_layer_mean_normalize_then_arithmetic_mean"
    )
    assert torch.allclose(profile["output_saliency_factor"], torch.ones(1, 2))
    assert profile["mode"] == "conditional_dual_tail_0p50_output_saliency_b1p00"


def test_profile_builder_can_add_sparse_output_saliency_safety_floors(
    tmp_path: Path, monkeypatch
) -> None:
    teacher = _save(
        tmp_path / "teacher.pt",
        {
            "model_path": "model",
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "test_metrics_used": False,
            "parent_mode": "dual",
            "unconditional_block_values": {
                0: torch.tensor([[10.0, 9.0], [0.1, 0.1]])
            },
        },
    )
    channel = _save(
        tmp_path / "channel.pt",
        {
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "block_size": 64,
            "tail_lambda": 0.5,
            "score_mode": "tail",
            "table": {0: {"block_coverage_scores": torch.ones(2, 2)}},
            "expert_tail_risk_proxy": {0: torch.tensor([1.0, 1.0])},
        },
    )
    saliency = _save(
        tmp_path / "saliency.pt",
        {
            "split": "train",
            "sequence_length": 2048,
            "test_metrics_used": False,
            "calibration_token_offset": 0,
            "calibration_token_end": 4096,
            "expert_output_saliency_mean": {0: torch.tensor([1.0, 10.0])},
            "output_saliency_formula": "mean_active_token(gate * l2_norm(expert_output))",
        },
    )
    output = tmp_path / "profile.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tail_risk_profile.py",
            "--teacher-cache",
            str(teacher),
            "--reference-channel-cache",
            str(channel),
            "--tail-channel-cache",
            str(channel),
            "--output-profile",
            str(output),
            "--target-pruning-ratio",
            "0.5",
            "--output-saliency-cache",
            str(saliency),
            "--output-saliency-floor-min-width",
            "1",
            "--output-saliency-floor-quantile",
            "0.5",
            "--output-saliency-floor-relative-max",
            "0.0",
        ],
    )

    assert main() == 0
    profile = torch.load(output, map_location="cpu", weights_only=True)
    floor = profile["output_saliency_risk_floor"]
    assert len(floor["selected_experts"]) == 1
    selected = floor["selected_experts"][0]
    assert selected["layer"] == 0
    assert selected["expert"] == 1
    assert selected["risk"] == pytest.approx(1.8181818723678589)
    assert selected["min_width"] == 1
    assert floor["newly_constrained_count"] == 1
    assert profile["profile_widths"].tolist() == [[1, 1]]
    assert profile["mode"] == "conditional_dual_tail_0p50_output_floor_w1"


def test_profile_builder_can_add_unique_contribution_safety_floors(
    tmp_path: Path, monkeypatch
) -> None:
    teacher = _save(
        tmp_path / "teacher.pt",
        {
            "model_path": "model",
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "test_metrics_used": False,
            "parent_mode": "dual",
            "unconditional_block_values": {0: torch.ones(5, 2)},
        },
    )
    channel = _save(
        tmp_path / "channel.pt",
        {
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "block_size": 64,
            "tail_lambda": 0.5,
            "score_mode": "tail",
            "table": {0: {"block_coverage_scores": torch.ones(5, 2)}},
            "expert_tail_risk_proxy": {0: torch.ones(5)},
        },
    )
    co_route = torch.tensor(
        [
            [0.0, 0.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
        ]
    )
    saliency = _save(
        tmp_path / "saliency.pt",
        {
            "split": "train",
            "sequence_length": 2048,
            "test_metrics_used": False,
            "calibration_token_offset": 0,
            "calibration_token_end": 4096,
            "expert_output_saliency_mean": {
                0: torch.tensor([1.0, 1.0, 10.0, 1.0, 1.0])
            },
            "expert_co_route_context": {0: co_route},
            "output_saliency_formula": "mean_active_token(gate * l2_norm(expert_output))",
            "co_route_formula": "sum_token(dense_topk_gate_outer_product); diagonal retained in cache",
        },
    )
    output = tmp_path / "profile.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tail_risk_profile.py",
            "--teacher-cache",
            str(teacher),
            "--reference-channel-cache",
            str(channel),
            "--tail-channel-cache",
            str(channel),
            "--output-profile",
            str(output),
            "--target-pruning-ratio",
            "0.5",
            "--output-saliency-cache",
            str(saliency),
            "--unique-contribution-floor-min-width",
            "1",
            "--unique-contribution-floor-quantile",
            "0.8",
            "--unique-contribution-floor-relative-max",
            "0.0",
        ],
    )

    assert main() == 0
    profile = torch.load(output, map_location="cpu", weights_only=True)
    floor = profile["unique_contribution_risk_floor"]
    assert floor["selected_experts"][0]["layer"] == 0
    assert floor["selected_experts"][0]["expert"] == 2
    assert floor["newly_constrained_count"] == 1
    assert profile["profile_widths"][0, 2] >= 1
    assert profile["mode"] == "conditional_dual_tail_0p50_unique_floor_w1"


def test_profile_builder_can_add_frontier_committee_regret_floors(
    tmp_path: Path, monkeypatch
) -> None:
    teacher = _save(
        tmp_path / "teacher.pt",
        {
            "model_path": "model",
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "test_metrics_used": False,
            "parent_mode": "dual",
            "unconditional_block_values": {0: torch.ones(3, 3)},
        },
    )
    channel = _save(
        tmp_path / "channel.pt",
        {
            "dataset": "wikitext-2-raw-v1",
            "split": "train",
            "sequence_length": 2048,
            "block_size": 64,
            "tail_lambda": 0.5,
            "score_mode": "tail",
            "table": {
                0: {
                    "block_coverage_scores": torch.tensor(
                        [[3.0, 2.0, 1.0]] * 3
                    )
                }
            },
            "expert_tail_risk_proxy": {0: torch.ones(3)},
        },
    )
    reference = _save(
        tmp_path / "reference_profile.pt",
        {
            "profile_widths": torch.tensor([[0, 1, 3]]),
            "total_blocks": 4,
            "maximum_blocks": 9,
            "test_metrics_used_for_profile": False,
        },
    )
    regret_caches = []
    for index, values in enumerate(
        (
            torch.tensor([[10.0, 1.0, 1.0], [1.0, 2.0, 9.0], [99.0] * 3]),
            torch.tensor([[8.0, 1.0, 1.0], [1.0, 4.0, 9.0], [99.0] * 3]),
        )
    ):
        regret_caches.append(
            _save(
                tmp_path / f"regret_{index}.pt",
                {
                    "split": "train",
                    "sequence_length": 2048,
                    "test_metrics_used": False,
                    "calibration_token_offset": index * 4096,
                    "expert_block_committee_residual_mean": {0: values},
                    "block_committee_regret_formula": "formula",
                    "block_committee_regret_approximation": "diagonal",
                },
            )
        )
    output = tmp_path / "profile.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tail_risk_profile.py",
            "--teacher-cache",
            str(teacher),
            "--reference-channel-cache",
            str(channel),
            "--tail-channel-cache",
            str(channel),
            "--output-profile",
            str(output),
            "--target-pruning-ratio",
            "0.5",
            "--frontier-reference-profile",
            str(reference),
            "--frontier-regret-cache",
            str(regret_caches[0]),
            "--frontier-regret-cache",
            str(regret_caches[1]),
            "--frontier-regret-floor-quantile",
            "0.5",
            "--frontier-regret-width-increment",
            "1",
            "--frontier-regret-fold-aggregation",
            "minimum",
        ],
    )

    assert main() == 0
    profile = torch.load(output, map_location="cpu", weights_only=True)
    floor = profile["frontier_committee_regret_floor"]
    assert floor["selected_count"] == 1
    assert floor["selected_experts"][0]["expert"] == 0
    assert floor["selected_experts"][0]["min_width"] == 1
    assert profile["profile_widths"][0, 0] >= 1
    assert profile["mode"] == "conditional_dual_tail_0p50_frontier_regret_q0p500"
