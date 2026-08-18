from __future__ import annotations

from collections.abc import Mapping


def _calibration_token_hash(profile: Mapping[str, object]) -> str | None:
    provenance = profile.get("cache_provenance")
    if not isinstance(provenance, Mapping):
        return None
    calibration = provenance.get("calibration")
    if not isinstance(calibration, Mapping):
        return None
    value = calibration.get("input_ids_sha256")
    return None if value is None else str(value)


def validate_profile_pair(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    group: str,
    evaluation_cache_sha256: str,
    expected_evaluation_cache_sha256: str,
) -> dict[str, object]:
    """Validate REAP/static profile fairness before paired evaluation."""

    if group not in {"method_native", "per_layer_controlled"}:
        raise ValueError("group must be method_native or per_layer_controlled.")
    if evaluation_cache_sha256 != expected_evaluation_cache_sha256:
        raise ValueError("evaluation cache SHA256 does not match the frozen protocol.")
    for key in ("num_layers", "num_experts", "num_blocks", "maximum_blocks"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"profile topology field {key} does not match.")
    reference_calibration = _calibration_token_hash(reference)
    candidate_calibration = _calibration_token_hash(candidate)
    if not reference_calibration or reference_calibration != candidate_calibration:
        raise ValueError("calibration token artifact does not match between profiles.")
    reference_total = int(reference.get("total_blocks", -1))
    candidate_total = int(candidate.get("total_blocks", -1))
    if reference_total != candidate_total:
        raise ValueError("total routed-expert block budget does not match.")
    reference_layers = [int(value) for value in reference.get("actual_blocks_by_layer", [])]
    candidate_layers = [int(value) for value in candidate.get("actual_blocks_by_layer", [])]
    per_layer_matched = reference_layers == candidate_layers
    if group == "per_layer_controlled" and not per_layer_matched:
        raise ValueError("per-layer routed-expert block budget does not match.")
    return {
        "passed": True,
        "group": group,
        "reference_method": reference.get("method"),
        "candidate_method": candidate.get("method"),
        "total_budget_matched": True,
        "per_layer_budget_matched": per_layer_matched,
        "calibration_input_ids_sha256": reference_calibration,
        "evaluation_cache_sha256": evaluation_cache_sha256,
    }