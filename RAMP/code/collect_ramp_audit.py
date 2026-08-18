from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from collect_ramp_covariances import collect_split, file_sha256, load_payload
from ramp_protocol import DEFAULT_SPLIT_QUOTAS, E1_SPLIT_QUOTAS, build_stratified_split_indices, index_tensor_sha256
from src.calibration_data import load_shared_calibration_tokens
from src.model_loading import load_supported_moe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the frozen RAMP audit split only.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-family", default="qwen3")
    parser.add_argument("--calibration-token-cache", type=Path, required=True)
    parser.add_argument("--fit-validation-cache", type=Path, required=True)
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--experiment", choices=("RAMP-E0", "RAMP-E1"), default="RAMP-E0")
    return parser.parse_args()


def load_decision(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("frozen_before_audit") is not True:
        raise ValueError("decision file must be frozen before audit collection.")
    return payload


def validate_audit_inputs(
    fit_validation_payload: dict,
    fit_validation_path: Path,
    decision_payload: dict,
) -> dict[int, tuple[int, ...]]:
    """Validate frozen provenance and return target physical experts by layer."""

    if fit_validation_payload.get("experiment") != decision_payload.get("experiment"):
        raise ValueError("fit/validation cache and decision experiment names differ.")
    if fit_validation_payload.get("smoke_only") is True:
        raise ValueError("audit cannot use a smoke fit/validation cache.")
    if fit_validation_payload.get("audit_collected") is True:
        raise ValueError("fit/validation cache must not contain audit statistics.")
    actual_sha = file_sha256(fit_validation_path)
    if decision_payload.get("covariance_cache_sha256") != actual_sha:
        raise ValueError("decision covariance SHA does not match the fit/validation cache.")

    expected_pairs = {
        (int(item["layer"]), int(item["expert"]))
        for item in fit_validation_payload.get("representative_experts", [])
    }
    decision_pairs = {
        (int(item["layer"]), int(item["expert"]))
        for item in decision_payload.get("decisions", [])
    }
    if not expected_pairs or len(expected_pairs) != len(decision_pairs) or decision_pairs != expected_pairs:
        raise ValueError("decision experts do not match the frozen representative expert set.")
    for item in decision_payload["decisions"]:
        if isinstance(item.get("keep_indices"), dict):
            index_sets = item["keep_indices"].items()
        else:
            index_sets = (
                (field, item.get(field))
                for field in ("ramp_keep_indices", "rms_keep_indices", "tail_keep_indices")
            )
        for field, indices in index_sets:
            if not isinstance(indices, list) or len(indices) != int(item.get("keep_count", -1)):
                raise ValueError(f"decision field {field} has an invalid channel count.")
            if len(set(int(index) for index in indices)) != len(indices):
                raise ValueError(f"decision field {field} contains duplicate channels.")

    targets: dict[int, list[int]] = {}
    for layer_idx, expert_idx in sorted(decision_pairs):
        targets.setdefault(layer_idx, []).append(expert_idx)
    return {layer_idx: tuple(experts) for layer_idx, experts in targets.items()}


def main() -> int:
    args = parse_args()
    fit_validation_payload = load_payload(args.fit_validation_cache)
    decision_payload = load_decision(args.decision_file)
    target_experts = validate_audit_inputs(
        fit_validation_payload,
        args.fit_validation_cache,
        decision_payload,
    )
    token_payload = load_payload(args.calibration_token_cache)
    if token_payload.get("input_ids_sha256") != fit_validation_payload.get("calibration_input_ids_sha256"):
        raise ValueError("calibration token SHA does not match the fit/validation artifact.")
    if file_sha256(args.calibration_token_cache) != fit_validation_payload.get("calibration_cache_file_sha256"):
        raise ValueError("calibration cache file SHA does not match the fit/validation artifact.")
    sequence_order = token_payload.get("source", {}).get("sequence_order")
    if not isinstance(sequence_order, list):
        raise ValueError("calibration cache must include source.sequence_order.")
    quotas = E1_SPLIT_QUOTAS if args.experiment == "RAMP-E1" else DEFAULT_SPLIT_QUOTAS
    splits = build_stratified_split_indices(sequence_order, seed=int(args.seed), quotas=quotas)
    expected_audit = fit_validation_payload.get("manifest", {}).get("split_indices", {}).get("audit", {})
    expected_count = expected_audit.get("count", expected_audit.get("frozen_count", -1))
    expected_sha = expected_audit.get("sha256", expected_audit.get("frozen_sha256"))
    if int(expected_count) != int(splits["audit"].numel()):
        raise ValueError("audit split count does not match the frozen manifest.")
    if expected_sha != index_tensor_sha256(splits["audit"]):
        raise ValueError("audit split SHA does not match the frozen manifest.")

    tokens, _ = load_shared_calibration_tokens(
        args.calibration_token_cache,
        required_sequence_length=int(args.sequence_length),
        model_path=str(args.model_path),
        device="cpu",
    )
    model, _ = load_supported_moe(
        str(args.model_path),
        device_map=args.device_map,
        model_family=args.model_family,
    )
    device = next(model.parameters()).device
    statistics = collect_split(
        model,
        tokens,
        splits["audit"],
        int(args.sequence_length),
        target_experts,
        device=device,
    )
    for layer_values in statistics.values():
        for values in layer_values.values():
            values.pop("down_proj", None)

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment": str(args.experiment),
        "split": "audit",
        "audit_collected": True,
        "test_metrics_used_for_selection": False,
        "model_path": str(args.model_path.resolve()),
        "model_family": str(args.model_family),
        "fit_validation_cache": str(args.fit_validation_cache.resolve()),
        "fit_validation_cache_sha256": file_sha256(args.fit_validation_cache),
        "decision_file": str(args.decision_file.resolve()),
        "decision_file_sha256": file_sha256(args.decision_file),
        "calibration_cache_file_sha256": file_sha256(args.calibration_token_cache),
        "calibration_input_ids_sha256": token_payload["input_ids_sha256"],
        "audit_indices": {
            "count": int(splits["audit"].numel()),
            "sha256": index_tensor_sha256(splits["audit"]),
            "indices": splits["audit"].tolist(),
        },
        "target_experts": {layer: list(experts) for layer, experts in target_experts.items()},
        "statistics": statistics,
    }
    torch.save(payload, args.output_cache)
    print(args.output_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())