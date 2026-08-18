from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ramp_reconstruction import normalized_output_error
from run_ramp_reconstruction_probe import file_sha256


SELECTION_FIELDS = {
    "ramp": "ramp_keep_indices",
    "rms": "rms_keep_indices",
    "tail": "tail_keep_indices",
}
DEFAULT_ALPHA_GRID = (1.0e-3, 1.0e-2, 1.0e-1, 3.0e-1, 1.0, 3.0, 10.0, 30.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a post-hoc RAMP-E0 ridge regularization diagnostic.")
    parser.add_argument("--fit-validation-cache", type=Path, required=True)
    parser.add_argument("--audit-cache", type=Path, required=True)
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=DEFAULT_ALPHA_GRID)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    fit_validation = torch.load(args.fit_validation_cache, map_location="cpu", weights_only=True)
    audit = torch.load(args.audit_cache, map_location="cpu", weights_only=True)
    decisions = _load_json(args.decision_file)
    fit_validation_sha = file_sha256(args.fit_validation_cache)
    if fit_validation.get("experiment") != "RAMP-E0":
        raise ValueError("fit/validation cache is not an RAMP-E0 artifact.")
    if audit.get("audit_collected") is not True or audit.get("split") != "audit":
        raise ValueError("audit cache is not a completed RAMP audit artifact.")
    if audit.get("fit_validation_cache_sha256") != fit_validation_sha:
        raise ValueError("audit cache does not match the fit/validation cache SHA.")
    if decisions.get("covariance_cache_sha256") != fit_validation_sha:
        raise ValueError("decision file does not match the fit/validation cache SHA.")
    if decisions.get("frozen_before_audit") is not True:
        raise ValueError("decisions must have been frozen before audit collection.")
    return fit_validation, audit, decisions


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute the median of an empty sequence.")
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), 0.5).item())


def _score_alpha_grid(
    down_proj: torch.Tensor,
    fit_covariance: torch.Tensor,
    validation_covariance: torch.Tensor,
    audit_covariance: torch.Tensor,
    keep_indices: torch.Tensor,
    alpha_grid: tuple[float, ...],
) -> tuple[float, dict[float, dict[str, float]]]:
    weights = down_proj.to(dtype=torch.float64)
    fit_covariance = fit_covariance.to(dtype=torch.float64)
    validation_covariance = validation_covariance.to(dtype=torch.float64)
    audit_covariance = audit_covariance.to(dtype=torch.float64)
    keep = keep_indices.to(dtype=torch.long)
    pruned_mask = torch.ones(weights.shape[1], dtype=torch.bool)
    pruned_mask[keep] = False
    pruned = torch.nonzero(pruned_mask, as_tuple=False).flatten()
    keep_covariance = fit_covariance.index_select(0, keep).index_select(1, keep)
    keep_covariance = 0.5 * (keep_covariance + keep_covariance.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(keep_covariance)
    target = weights.index_select(1, pruned).matmul(
        fit_covariance.index_select(0, pruned).index_select(1, keep)
    )
    rotated_target = target.matmul(eigenvectors)
    base = weights.index_select(1, keep)
    regularization_scale = float(torch.trace(fit_covariance).item()) / float(fit_covariance.shape[0])
    audit_none_error = normalized_output_error(weights, audit_covariance, keep, base)
    scores = {}
    for alpha in alpha_grid:
        inverse_scale = (eigenvalues + float(alpha) * regularization_scale).clamp_min(1.0e-30).reciprocal()
        delta = (rotated_target * inverse_scale[None, :]).matmul(eigenvectors.transpose(0, 1))
        effective = base + delta
        audit_error = normalized_output_error(weights, audit_covariance, keep, effective)
        scores[float(alpha)] = {
            "validation_error": normalized_output_error(
                weights,
                validation_covariance,
                keep,
                effective,
            ),
            "audit_error": audit_error,
            "audit_none_error": audit_none_error,
            "audit_r2_pruned": float(1.0 - audit_error / max(audit_none_error, 1.0e-12)),
            "compensation_frobenius_ratio": float(
                delta.norm().item() / base.norm().clamp_min(1.0e-12).item()
            ),
        }
    return audit_none_error, scores


def _group_summary(rows: list[dict], alpha: float) -> dict[str, float | int]:
    return {
        "experts": len(rows),
        "median_audit_error": _median([row["scores"][alpha]["audit_error"] for row in rows]),
        "median_audit_r2_pruned": _median([row["scores"][alpha]["audit_r2_pruned"] for row in rows]),
        "median_audit_validation_ratio": _median(
            [
                row["scores"][alpha]["audit_error"]
                / max(row["scores"][alpha]["validation_error"], 1.0e-12)
                for row in rows
            ]
        ),
        "positive_audit_r2_experts": sum(row["scores"][alpha]["audit_r2_pruned"] > 0.0 for row in rows),
    }


def main() -> int:
    args = parse_args()
    alpha_grid = tuple(sorted({float(alpha) for alpha in args.alpha_grid}))
    if not alpha_grid or alpha_grid[0] <= 0.0:
        raise ValueError("alpha-grid must contain positive values.")
    fit_validation, audit, decisions = _validate_inputs(args)
    rows = []
    for decision in decisions.get("decisions", []):
        layer_idx = int(decision["layer"])
        expert_idx = int(decision["expert"])
        values = fit_validation["statistics"][layer_idx][expert_idx]
        fit_stats = values["splits"]["fit"]
        validation_stats = values["splits"]["validation"]
        audit_stats = audit["statistics"][layer_idx][expert_idx]
        for selection, field in SELECTION_FIELDS.items():
            keep = torch.tensor(decision[field], dtype=torch.long)
            audit_none_error, scores = _score_alpha_grid(
                values["down_proj"],
                fit_stats["covariance"],
                validation_stats["covariance"],
                audit_stats["covariance"],
                keep,
                alpha_grid,
            )
            rows.append(
                {
                    "layer": layer_idx,
                    "expert": expert_idx,
                    "selection": selection,
                    "route_count_fit": int(fit_stats["route_count"]),
                    "route_count_validation": int(validation_stats["route_count"]),
                    "route_count_audit": int(audit_stats["route_count"]),
                    "audit_none_error": audit_none_error,
                    "scores": scores,
                }
            )

    selected_alpha = {}
    summary = {}
    for selection in SELECTION_FIELDS:
        selection_rows = [row for row in rows if row["selection"] == selection]
        median_validation_by_alpha = {
            alpha: _median([row["scores"][alpha]["validation_error"] for row in selection_rows])
            for alpha in alpha_grid
        }
        alpha = min(alpha_grid, key=lambda value: (median_validation_by_alpha[value], value))
        selected_alpha[selection] = alpha
        summary[selection] = {
            "validation_selected_alpha": alpha,
            "median_validation_error_by_alpha": median_validation_by_alpha,
            "median_validation_error": median_validation_by_alpha[alpha],
            "median_audit_error": _median([row["scores"][alpha]["audit_error"] for row in selection_rows]),
            "median_audit_r2_pruned": _median(
                [row["scores"][alpha]["audit_r2_pruned"] for row in selection_rows]
            ),
            "median_compensation_ratio": _median(
                [row["scores"][alpha]["compensation_frobenius_ratio"] for row in selection_rows]
            ),
            "positive_audit_r2_experts": sum(
                row["scores"][alpha]["audit_r2_pruned"] > 0.0 for row in selection_rows
            ),
        }

    by_key = {
        (int(row["layer"]), int(row["expert"]), str(row["selection"])): row
        for row in rows
    }
    relative_improvements = []
    wins = 0
    for decision in decisions["decisions"]:
        layer_idx = int(decision["layer"])
        expert_idx = int(decision["expert"])
        ramp_error = by_key[(layer_idx, expert_idx, "ramp")]["scores"][selected_alpha["ramp"]]["audit_error"]
        rms_error = by_key[(layer_idx, expert_idx, "rms")]["scores"][selected_alpha["rms"]]["audit_error"]
        tail_error = by_key[(layer_idx, expert_idx, "tail")]["scores"][selected_alpha["tail"]]["audit_error"]
        baseline_error = min(rms_error, tail_error)
        relative_improvements.append((baseline_error - ramp_error) / max(baseline_error, 1.0e-12))
        wins += int(ramp_error < baseline_error)

    ramp_rows = [row for row in rows if row["selection"] == "ramp"]
    ramp_alpha = selected_alpha["ramp"]
    route_order = sorted(ramp_rows, key=lambda row: int(row["route_count_fit"]))
    route_strata = {
        "low": route_order[:8],
        "medium": route_order[8:16],
        "high": route_order[16:],
    }
    stratified = {
        "by_fit_route_stratum": {
            name: {
                **_group_summary(values, ramp_alpha),
                "fit_route_count_min": min(int(row["route_count_fit"]) for row in values),
                "fit_route_count_max": max(int(row["route_count_fit"]) for row in values),
            }
            for name, values in route_strata.items()
        },
        "by_layer": {
            str(layer_idx): _group_summary(
                [row for row in ramp_rows if int(row["layer"]) == layer_idx],
                ramp_alpha,
            )
            for layer_idx in sorted({int(row["layer"]) for row in ramp_rows})
        },
    }
    comparison = {
        "median_relative_improvement_vs_best_rms_tail": _median(relative_improvements),
        "ramp_wins_vs_best_rms_tail": wins,
        "paired_experts": len(relative_improvements),
    }
    output = {
        "schema_version": 1,
        "experiment": "RAMP-E0",
        "post_hoc": True,
        "changes_preregistered_outcome": False,
        "purpose": "Diagnose whether the preregistered ridge grid ended at an active boundary.",
        "fit_validation_cache_sha256": file_sha256(args.fit_validation_cache),
        "audit_cache_sha256": file_sha256(args.audit_cache),
        "decision_file_sha256": file_sha256(args.decision_file),
        "alpha_grid": alpha_grid,
        "summary": summary,
        "comparison": comparison,
        "stratified": stratified,
        "rows": rows,
    }
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_results.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RAMP-E0 Post-hoc Regularization Diagnostic",
        "",
        "This diagnostic does not change the preregistered audit outcome. The expanded ridge grid was selected using validation error only.",
        "",
        "| selection | validation-selected alpha | median validation error | median audit error | median audit R2 | positive audit R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selection, values in summary.items():
        lines.append(
            f"| {selection} | {values['validation_selected_alpha']:.6g} | "
            f"{values['median_validation_error']:.6f} | {values['median_audit_error']:.6f} | "
            f"{values['median_audit_r2_pruned']:.6f} | {values['positive_audit_r2_experts']}/24 |"
        )
    lines.extend(
        [
            "",
            f"RAMP median relative improvement over best RMS/Tail: `{comparison['median_relative_improvement_vs_best_rms_tail']:.6f}`.",
            f"RAMP wins over best RMS/Tail: `{comparison['ramp_wins_vs_best_rms_tail']}/24`.",
            "",
            "## RAMP by Fit-route Stratum",
            "",
            "| stratum | fit route range | median audit error | median audit R2 | audit/validation ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for stratum, values in stratified["by_fit_route_stratum"].items():
        lines.append(
            f"| {stratum} | {values['fit_route_count_min']}-{values['fit_route_count_max']} | "
            f"{values['median_audit_error']:.6f} | {values['median_audit_r2_pruned']:.6f} | "
            f"{values['median_audit_validation_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## RAMP by Layer",
            "",
            "| layer | median audit error | median audit R2 | positive audit R2 |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for layer_idx, values in stratified["by_layer"].items():
        lines.append(
            f"| {layer_idx} | {values['median_audit_error']:.6f} | "
            f"{values['median_audit_r2_pruned']:.6f} | {values['positive_audit_r2_experts']}/{values['experts']} |"
        )
    args.output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_results)
    print(args.output_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())