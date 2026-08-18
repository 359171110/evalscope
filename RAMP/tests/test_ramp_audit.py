from __future__ import annotations

import json

import pytest
import torch

from collect_ramp_audit import load_decision, validate_audit_inputs
from evaluate_ramp_audit import classify_preregistered_outcome, paired_bootstrap_median_ci


def test_audit_rejects_decision_hash_mismatch(tmp_path) -> None:
    covariance = tmp_path / "covariance.pt"
    torch.save({"value": 1}, covariance)
    representatives = [
        {"layer": layer, "expert": expert}
        for layer in range(4)
        for expert in range(6)
    ]
    decisions = [
        {
            "layer": item["layer"],
            "expert": item["expert"],
            "keep_count": 2,
            "ramp_keep_indices": [0, 1],
            "rms_keep_indices": [0, 1],
            "tail_keep_indices": [0, 1],
        }
        for item in representatives
    ]
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "experiment": "RAMP-E0",
                "frozen_before_audit": True,
                "covariance_cache_sha256": "0" * 64,
                "decisions": decisions,
            }
        ),
        encoding="utf-8",
    )

    decision = load_decision(decision_path)
    with pytest.raises(ValueError, match="covariance SHA"):
        validate_audit_inputs(
            {
                "experiment": "RAMP-E0",
                "smoke_only": False,
                "audit_collected": False,
                "representative_experts": representatives,
            },
            covariance,
            decision,
        )


def test_paired_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_median_ci([0.1, 0.2, 0.3, 0.4], seed=42, iterations=1000)
    second = paired_bootstrap_median_ci([0.1, 0.2, 0.3, 0.4], seed=42, iterations=1000)

    assert first == second
    assert first[0] <= 0.25 <= first[1]


def test_low_reconstructability_is_preregistered_no_go() -> None:
    assert classify_preregistered_outcome({"median_audit_r2_pruned": 0.05}) == "NO_GO_CORE_RECONSTRUCTABILITY"