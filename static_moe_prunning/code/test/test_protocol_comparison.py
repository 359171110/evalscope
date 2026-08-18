from __future__ import annotations

import pytest

from src.protocol_comparison import validate_profile_pair


def _profile(method: str, blocks_by_layer: list[int]) -> dict:
    return {
        "method": method,
        "model_path": "/models/qwen3",
        "num_layers": 2,
        "num_experts": 4,
        "num_blocks": 2,
        "maximum_blocks": 16,
        "total_blocks": sum(blocks_by_layer),
        "actual_blocks_by_layer": blocks_by_layer,
        "cache_provenance": {
            "calibration": {"input_ids_sha256": "a" * 64},
        },
    }


def test_method_native_pair_requires_total_budget_and_shared_artifacts() -> None:
    audit = validate_profile_pair(
        _profile("official_reap", [4, 4]),
        _profile("route_tail", [5, 3]),
        group="method_native",
        evaluation_cache_sha256="b" * 64,
        expected_evaluation_cache_sha256="b" * 64,
    )

    assert audit["passed"] is True
    assert audit["total_budget_matched"] is True
    assert audit["per_layer_budget_matched"] is False


def test_per_layer_pair_rejects_cross_layer_reallocation() -> None:
    with pytest.raises(ValueError, match="per-layer"):
        validate_profile_pair(
            _profile("official_reap", [4, 4]),
            _profile("route_tail", [5, 3]),
            group="per_layer_controlled",
            evaluation_cache_sha256="b" * 64,
            expected_evaluation_cache_sha256="b" * 64,
        )


def test_pair_rejects_different_calibration_or_evaluation_cache() -> None:
    candidate = _profile("route_tail", [4, 4])
    candidate["cache_provenance"]["calibration"]["input_ids_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="calibration"):
        validate_profile_pair(
            _profile("official_reap", [4, 4]),
            candidate,
            group="per_layer_controlled",
            evaluation_cache_sha256="b" * 64,
            expected_evaluation_cache_sha256="b" * 64,
        )

    with pytest.raises(ValueError, match="evaluation cache"):
        validate_profile_pair(
            _profile("official_reap", [4, 4]),
            _profile("route_tail", [4, 4]),
            group="per_layer_controlled",
            evaluation_cache_sha256="d" * 64,
            expected_evaluation_cache_sha256="b" * 64,
        )