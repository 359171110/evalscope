from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from ramp_reconstruction import (
    fit_rank_limited_compensation,
    fit_ridge_compensation,
    normalized_output_error,
)
from run_ramp_reconstruction_probe import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen RAMP-E0 decisions on audit covariance.")
    parser.add_argument("--fit-validation-cache", type=Path, required=True)
    parser.add_argument("--audit-cache", type=Path, required=True)
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--validation-results", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    fit_validation = torch.load(args.fit_validation_cache, map_location="cpu", weights_only=True)
    audit = torch.load(args.audit_cache, map_location="cpu", weights_only=True)
    decisions = _load_json(args.decision_file)
    validation = _load_json(args.validation_results)
    if audit.get("audit_collected") is not True or audit.get("split") != "audit":
        raise ValueError("audit cache is not a completed RAMP audit artifact.")
    if audit.get("fit_validation_cache_sha256") != file_sha256(args.fit_validation_cache):
        raise ValueError("audit cache does not match fit/validation cache SHA.")
    if decisions.get("covariance_cache_sha256") != file_sha256(args.fit_validation_cache):
        raise ValueError("decision file does not match fit/validation cache SHA.")
    if validation.get("covariance_cache_sha256") != file_sha256(args.fit_validation_cache):
        raise ValueError("validation result does not match fit/validation cache SHA.")
    if decisions.get("frozen_before_audit") is not True:
        raise ValueError("decisions must be frozen before audit evaluation.")
    return fit_validation, audit, decisions, validation


def _audit_stats(audit: dict, layer_idx: int, expert_idx: int) -> dict:
    stats = audit["statistics"].get(int(layer_idx), {}).get(int(expert_idx))
    if not stats or int(stats.get("route_count", 0)) <= 0:
        raise ValueError(f"missing audit stats for layer {layer_idx}, expert {expert_idx}.")
    return stats


def _decision_map(decisions: dict) -> dict[tuple[int, int], dict]:
    values = decisions.get("decisions", [])
    if len(values) != 24:
        raise ValueError("expected 24 frozen decisions.")
    result = {}
    for value in values:
        key = (int(value["layer"]), int(value["expert"]))
        if key in result:
            raise ValueError(f"duplicate decision for {key}.")
        result[key] = value
    return result


def _evaluate(
    down_proj: torch.Tensor,
    fit_covariance: torch.Tensor,
    audit_covariance: torch.Tensor,
    keep_indices: torch.Tensor,
    *,
    alpha: float,
    rank: int | None,
) -> dict[str, float]:
    regularization = float(alpha) * float(torch.trace(fit_covariance).item()) / float(fit_covariance.shape[0])
    if rank is None:
        effective, delta = fit_ridge_compensation(
            down_proj,
            fit_covariance,
            keep_indices,
            regularization=regularization,
        )
    else:
        effective, delta = fit_rank_limited_compensation(
            down_proj,
            fit_covariance,
            keep_indices,
            regularization=regularization,
            rank=int(rank),
        )
    no_compensation = down_proj.index_select(1, keep_indices)
    audit_error = normalized_output_error(down_proj, audit_covariance, keep_indices, effective)
    audit_none_error = normalized_output_error(down_proj, audit_covariance, keep_indices, no_compensation)
    validation_error = normalized_output_error(down_proj, fit_covariance, keep_indices, effective)
    return {
        "audit_error": audit_error,
        "audit_none_error": audit_none_error,
        "audit_r2_pruned": float(1.0 - audit_error / max(audit_none_error, 1.0e-12)),
        "fit_error": validation_error,
        "compensation_frobenius_ratio": float(
            delta.norm().item() / no_compensation.norm().clamp_min(1.0e-12).item()
        ),
    }


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute the median of an empty sequence.")
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), 0.5).item())


def paired_bootstrap_median_ci(
    values: list[float],
    *,
    seed: int = 42,
    iterations: int = 10_000,
) -> tuple[float, float]:
    """Return a deterministic percentile CI for an expert-paired median."""

    if not values or int(iterations) <= 0:
        raise ValueError("values and iterations must be non-empty and positive.")
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        0,
        tensor.numel(),
        (int(iterations), tensor.numel()),
        generator=generator,
    )
    bootstrap = torch.quantile(tensor[indices], 0.5, dim=1)
    lower, upper = torch.quantile(bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(lower.item()), float(upper.item())


def classify_preregistered_outcome(criteria: dict[str, object]) -> str:
    """Map frozen audit criteria to the preregistered RAMP-E0 outcome."""

    if float(criteria["median_audit_r2_pruned"]) < 0.20:
        return "NO_GO_CORE_RECONSTRUCTABILITY"
    go_fields = (
        "median_relative_improvement_at_least_15pct",
        "wins_at_least_18_of_24",
        "median_r2_at_least_50pct",
        "rank16_retention_at_least_70pct",
        "generalization_ok_at_least_20_of_24",
        "compensation_scale_within_limits",
        "bootstrap_ci_lower_above_zero",
    )
    if all(bool(criteria[field]) for field in go_fields):
        return "GO_FULL_EQUAL_WIDTH"
    return "REVISE_SELECTION_OR_COMPENSATION"


def main() -> int:
    args = parse_args()
    fit_validation, audit, decisions, validation = _validate_inputs(args)
    decision_map = _decision_map(decisions)
    validation_rows = {
        (int(row["layer"]), int(row["expert"]), str(row["selection"])): row
        for row in validation["results"]
    }
    rows = []
    for (layer_idx, expert_idx), decision in sorted(decision_map.items()):
        fit_values = fit_validation["statistics"][layer_idx][expert_idx]
        fit_stats = fit_values["splits"]["fit"]
        audit_stats = _audit_stats(audit, layer_idx, expert_idx)
        down_proj = fit_values["down_proj"].to(dtype=torch.float64)
        selections = [
            ("ramp", "ramp_keep_indices"),
            ("rms", "rms_keep_indices"),
            ("tail", "tail_keep_indices"),
        ]
        selections.extend(
            (selection, None)
            for selection in sorted(decision.get("random_keep_indices", {}))
        )
        for selection, field in selections:
            raw_indices = (
                decision[field]
                if field is not None
                else decision["random_keep_indices"][selection]
            )
            keep = torch.tensor(raw_indices, dtype=torch.long)
            alpha = float(decisions["shared_alpha_by_selection"][selection]["alpha"])
            ridge = _evaluate(
                down_proj,
                fit_stats["covariance"].to(dtype=torch.float64),
                audit_stats["covariance"].to(dtype=torch.float64),
                keep,
                alpha=alpha,
                rank=None,
            )
            row = {
                "layer": layer_idx,
                "expert": expert_idx,
                "selection": selection,
                "route_count_audit": int(audit_stats["route_count"]),
                "alpha": alpha,
                "ridge": ridge,
            }
            validation_row = validation_rows[(layer_idx, expert_idx, selection)]
            row["validation_error"] = float(validation_row["ridge"]["validation_error"])
            if selection == "ramp":
                row["rank16"] = _evaluate(
                    down_proj,
                    fit_stats["covariance"].to(dtype=torch.float64),
                    audit_stats["covariance"].to(dtype=torch.float64),
                    keep,
                    alpha=alpha,
                    rank=16,
                )
            rows.append(row)

    by_selection: dict[str, list[dict]] = {}
    for row in rows:
        by_selection.setdefault(row["selection"], []).append(row)
    summary = {}
    for selection, values in by_selection.items():
        summary[selection] = {
            "experts": len(values),
            "median_audit_error": _median([item["ridge"]["audit_error"] for item in values]),
            "median_validation_error": _median([item["validation_error"] for item in values]),
            "median_audit_r2_pruned": _median([item["ridge"]["audit_r2_pruned"] for item in values]),
            "median_compensation_ratio": _median(
                [item["ridge"]["compensation_frobenius_ratio"] for item in values]
            ),
            "audit_validation_error_ratio_median": _median(
                [
                    item["ridge"]["audit_error"] / max(item["validation_error"], 1.0e-12)
                    for item in values
                ]
            ),
        }
        if selection == "ramp":
            summary[selection]["median_rank16_audit_error"] = _median(
                [item["rank16"]["audit_error"] for item in values]
            )
            summary[selection]["rank16_error_reduction_retention"] = _median(
                [
                    (item["ridge"]["audit_none_error"] - item["rank16"]["audit_error"])
                    / max(item["ridge"]["audit_none_error"] - item["ridge"]["audit_error"], 1.0e-12)
                    for item in values
                ]
            )

    pair_ids = sorted(
        {
            (int(row["layer"]), int(row["expert"]))
            for row in rows
            if row["selection"] == "ramp"
        }
    )
    by_pair = {
        (int(row["layer"]), int(row["expert"]), str(row["selection"])): row
        for row in rows
    }
    relative_improvements = []
    wins = 0
    generalization_ok = 0
    compensation_ratios = []
    ramp_r2 = []
    rank16_retentions = []
    random_mean_errors = []
    for layer_idx, expert_idx in pair_ids:
        ramp = by_pair[(layer_idx, expert_idx, "ramp")]
        rms = by_pair[(layer_idx, expert_idx, "rms")]
        tail = by_pair[(layer_idx, expert_idx, "tail")]
        baseline_error = min(rms["ridge"]["audit_error"], tail["ridge"]["audit_error"])
        ramp_error = ramp["ridge"]["audit_error"]
        relative_improvements.append((baseline_error - ramp_error) / max(baseline_error, 1.0e-12))
        wins += int(ramp_error < baseline_error)
        generalization_ok += int(ramp_error / max(ramp["validation_error"], 1.0e-12) <= 1.5)
        compensation_ratios.append(ramp["ridge"]["compensation_frobenius_ratio"])
        ramp_r2.append(ramp["ridge"]["audit_r2_pruned"])
        full_reduction = ramp["ridge"]["audit_none_error"] - ramp_error
        if full_reduction > 1.0e-12:
            rank16_retentions.append(
                (ramp["ridge"]["audit_none_error"] - ramp["rank16"]["audit_error"])
                / full_reduction
            )
        random_errors = [
            by_pair[(layer_idx, expert_idx, selection)]["ridge"]["audit_error"]
            for selection in by_selection
            if selection.startswith("random_")
        ]
        random_mean_errors.append(float(sum(random_errors) / len(random_errors)))

    relative_ci = paired_bootstrap_median_ci(relative_improvements)
    sorted_ratios = sorted(compensation_ratios)
    p90_index = max(0, int(torch.ceil(torch.tensor(0.9 * len(sorted_ratios))).item()) - 1)
    criteria = {
        "paired_experts": len(pair_ids),
        "median_relative_improvement_vs_best_rms_tail": _median(relative_improvements),
        "paired_bootstrap_95_ci": list(relative_ci),
        "ramp_wins_vs_best_rms_tail": wins,
        "median_audit_r2_pruned": _median(ramp_r2),
        "median_rank16_error_reduction_retention": _median(rank16_retentions),
        "rank16_retention_eligible_experts": len(rank16_retentions),
        "generalization_ok_count": generalization_ok,
        "median_compensation_ratio": _median(compensation_ratios),
        "p90_compensation_ratio": float(sorted_ratios[p90_index]),
        "median_random_mean_audit_error": _median(random_mean_errors),
    }
    criteria.update(
        {
            "median_relative_improvement_at_least_15pct": criteria[
                "median_relative_improvement_vs_best_rms_tail"
            ]
            >= 0.15,
            "wins_at_least_18_of_24": wins >= 18,
            "median_r2_at_least_50pct": criteria["median_audit_r2_pruned"] >= 0.50,
            "rank16_retention_at_least_70pct": criteria[
                "median_rank16_error_reduction_retention"
            ]
            >= 0.70,
            "generalization_ok_at_least_20_of_24": generalization_ok >= 20,
            "compensation_scale_within_limits": criteria["median_compensation_ratio"] <= 0.5
            and criteria["p90_compensation_ratio"] <= 1.0,
            "bootstrap_ci_lower_above_zero": relative_ci[0] > 0.0,
            "ramp_better_than_random_mean": summary["ramp"]["median_audit_error"]
            < criteria["median_random_mean_audit_error"],
        }
    )
    outcome = classify_preregistered_outcome(criteria)

    output = {
        "schema_version": 1,
        "experiment": "RAMP-E0",
        "fit_validation_cache_sha256": file_sha256(args.fit_validation_cache),
        "audit_cache_sha256": file_sha256(args.audit_cache),
        "decision_file_sha256": file_sha256(args.decision_file),
        "outcome": outcome,
        "criteria": criteria,
        "summary": summary,
        "rows": rows,
    }
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_results.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RAMP-E0 Audit Summary",
        "",
        "The audit split was evaluated only after the channel decisions and shared validation regularization were frozen.",
        "",
        f"Preregistered outcome: **{outcome}**.",
        "",
        "| selection | median audit error | median validation error | median audit R2 | median compensation ratio | audit/validation ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selection, values in sorted(summary.items()):
        lines.append(
            f"| {selection} | {values['median_audit_error']:.6f} | {values['median_validation_error']:.6f} | "
            f"{values['median_audit_r2_pruned']:.6f} | {values['median_compensation_ratio']:.6f} | "
            f"{values['audit_validation_error_ratio_median']:.6f} |"
        )
    if "ramp" in summary:
        lines.extend(
            [
                "",
                f"RAMP rank-16 median audit error: `{summary['ramp']['median_rank16_audit_error']:.6f}`.",
                f"Rank-16 retained error reduction: `{summary['ramp']['rank16_error_reduction_retention']:.6f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Preregistered Criteria",
            "",
            f"- Median relative improvement over best RMS/Tail: `{criteria['median_relative_improvement_vs_best_rms_tail']:.6f}`.",
            f"- Paired bootstrap 95% CI: `[{relative_ci[0]:.6f}, {relative_ci[1]:.6f}]`.",
            f"- Expert wins over best RMS/Tail: `{wins}/24`.",
            f"- Median audit residual R2: `{criteria['median_audit_r2_pruned']:.6f}`.",
            f"- Rank-16 error-reduction retention: `{criteria['median_rank16_error_reduction_retention']:.6f}`.",
            f"- Audit/validation ratio <= 1.5: `{generalization_ok}/24`.",
            f"- Compensation ratio median/P90: `{criteria['median_compensation_ratio']:.6f}` / `{criteria['p90_compensation_ratio']:.6f}`.",
            f"- Median random-mean audit error: `{criteria['median_random_mean_audit_error']:.6f}`.",
        ]
    )
    args.output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "layer",
                "expert",
                "selection",
                "route_count_audit",
                "alpha",
                "validation_error",
                "audit_error",
                "audit_none_error",
                "audit_r2_pruned",
                "compensation_frobenius_ratio",
                "rank16_audit_error",
                "rank16_audit_r2_pruned",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "layer": row["layer"],
                    "expert": row["expert"],
                    "selection": row["selection"],
                    "route_count_audit": row["route_count_audit"],
                    "alpha": row["alpha"],
                    "validation_error": row["validation_error"],
                    "audit_error": row["ridge"]["audit_error"],
                    "audit_none_error": row["ridge"]["audit_none_error"],
                    "audit_r2_pruned": row["ridge"]["audit_r2_pruned"],
                    "compensation_frobenius_ratio": row["ridge"]["compensation_frobenius_ratio"],
                    "rank16_audit_error": row.get("rank16", {}).get("audit_error"),
                    "rank16_audit_r2_pruned": row.get("rank16", {}).get("audit_r2_pruned"),
                }
            )
    print(args.output_results)
    print(args.output_summary)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())