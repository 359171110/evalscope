from __future__ import annotations

from src.static_research_validator import validate_static_research


def _row(mode: str, ppl: float) -> dict:
    return {
        "mode": mode,
        "ppl": ppl,
        "windows": 114,
        "tokens": 233368,
        "sequence_length": 2048,
        "standard_protocol": True,
        "dataset": "wikitext-2-raw-v1",
        "split": "test",
        "total_profile_blocks": 36864,
        "maximum_profile_blocks": 73728,
        "structural_pruning_ratio": 0.5,
        "correction_mode": "none",
        "profile_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "profile_sha256": f"sha-{mode}",
        "profile_file_sha256": f"file-sha-{mode}",
        "cache_provenance": {"channel": {"split": "train", "sha256": "cache"}},
    }


def _report() -> str:
    return " ".join(
        [
            "MoE-Slimming",
            "MoSE",
            "POP",
            "REAP",
            "MAESTRO",
            "FLAP",
            "MoE-Pruner",
            "Mixture Compressor",
            "DTop-p",
        ]
    )


def test_validator_passes_only_when_dual_utility_beats_all_matched_baselines() -> None:
    rows = [
        _row("uniform", 10.0),
        _row("rms", 9.8),
        _row("route_rms", 9.2),
        _row("dual_route_rms", 9.1),
        _row("dynamic_regret", 9.0),
        _row("dynamic_expected_utility", 8.9),
        _row("expected_utility_gate", 8.95),
        _row("expected_utility_top_p", 8.92),
        _row("expected_utility_dual", 8.8),
    ]

    result = validate_static_research(rows, novelty_report=_report())

    assert result["passed"] is True
    assert result["best_baseline"]["mode"] == "dynamic_expected_utility"
    assert result["candidate"]["ppl"] == 8.8


def test_validator_rejects_smoke_and_non_improving_candidate() -> None:
    rows = [
        _row("uniform", 10.0),
        _row("rms", 9.8),
        _row("route_rms", 9.2),
        _row("dual_route_rms", 9.1),
        _row("dynamic_regret", 9.2),
        _row("dynamic_expected_utility", 9.3),
        _row("expected_utility_gate", 9.25),
        _row("expected_utility_top_p", 9.22),
        _row("expected_utility_dual", 9.4),
    ]
    rows[0]["windows"] = 1
    rows[0]["standard_protocol"] = False

    result = validate_static_research(rows, novelty_report=_report())

    assert result["passed"] is False
    assert any("uniform" in issue for issue in result["issues"])
    assert any("strictly beat" in issue for issue in result["issues"])


def test_validator_requires_explicit_novelty_separation() -> None:
    rows = [
        _row("uniform", 10.0),
        _row("rms", 9.8),
        _row("route_rms", 9.2),
        _row("dual_route_rms", 9.1),
        _row("dynamic_regret", 9.0),
        _row("dynamic_expected_utility", 8.9),
        _row("expected_utility_gate", 8.95),
        _row("expected_utility_top_p", 8.92),
        _row("expected_utility_dual", 8.8),
    ]

    result = validate_static_research(rows, novelty_report="MoE-Slimming only")

    assert result["passed"] is False
    assert any("novelty report" in issue for issue in result["issues"])
