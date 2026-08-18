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
from run_ramp_e1_stage_a import file_sha256, fit_regularization


RANKS = ("rank16", "rank32", "rank64", "rank128", "full")
PRIMARY_SELECTIONS = (
    "rms",
    "tail",
    "ramp_e0",
    "pair_corr",
    "conditional_activation",
    "conditional_output",
    "conditional_stable",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen RAMP-E1 stage-A decisions on audit covariance.")
    parser.add_argument("--fit-validation-cache", type=Path, required=True)
    parser.add_argument("--audit-cache", type=Path, required=True)
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--validation-results", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    fit_validation = torch.load(args.fit_validation_cache, map_location="cpu", weights_only=True)
    audit = torch.load(args.audit_cache, map_location="cpu", weights_only=True)
    decisions = load_json(args.decision_file)
    validation = load_json(args.validation_results)
    artifacts = (fit_validation, audit, decisions, validation)
    if any(item.get("experiment") != "RAMP-E1" for item in artifacts):
        raise ValueError("all evaluator inputs must be RAMP-E1 artifacts.")
    fit_sha = file_sha256(args.fit_validation_cache)
    if audit.get("fit_validation_cache_sha256") != fit_sha:
        raise ValueError("audit cache does not match fit/validation cache SHA.")
    if decisions.get("covariance_cache_sha256") != fit_sha:
        raise ValueError("decision file does not match fit/validation cache SHA.")
    if validation.get("covariance_cache_sha256") != fit_sha:
        raise ValueError("validation result does not match fit/validation cache SHA.")
    if audit.get("decision_file_sha256") != file_sha256(args.decision_file):
        raise ValueError("audit cache does not match frozen decision SHA.")
    if decisions.get("frozen_before_audit") is not True:
        raise ValueError("decisions were not frozen before audit.")
    return fit_validation, audit, decisions, validation


def median(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float64).median())


def choose_primary_selection(summary: dict) -> str:
    eligible = ("conditional_output", "conditional_stable")
    return min(
        eligible,
        key=lambda selection: (summary[selection]["full"]["median_validation_error"], selection),
    )


def paired_bootstrap_median_ci(values: list[float], *, seed: int = 42, iterations: int = 10_000) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(0, tensor.numel(), (iterations, tensor.numel()), generator=generator)
    samples = torch.quantile(tensor[indices], 0.5, dim=1)
    bounds = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return [float(bounds[0]), float(bounds[1])]


def evaluate(
    down_proj: torch.Tensor,
    fit_covariance: torch.Tensor,
    audit_covariance: torch.Tensor,
    keep: torch.Tensor,
    *,
    alpha: float,
    rank_name: str,
) -> dict[str, float]:
    rank = None if rank_name == "full" else int(rank_name.removeprefix("rank"))
    regularization = fit_regularization(fit_covariance, alpha)
    if rank is None:
        effective, delta = fit_ridge_compensation(
            down_proj, fit_covariance, keep, regularization=regularization
        )
    else:
        effective, delta = fit_rank_limited_compensation(
            down_proj, fit_covariance, keep, regularization=regularization, rank=rank
        )
    none = down_proj.index_select(1, keep)
    audit_error = normalized_output_error(down_proj, audit_covariance, keep, effective)
    none_error = normalized_output_error(down_proj, audit_covariance, keep, none)
    return {
        "audit_error": audit_error,
        "audit_none_error": none_error,
        "audit_r2_pruned": 1.0 - audit_error / max(none_error, 1.0e-12),
        "compensation_frobenius_ratio": float(delta.norm() / none.norm().clamp_min(1.0e-12)),
    }


def main() -> int:
    args = parse_args()
    fit_validation, audit, decisions, validation = validate_inputs(args)
    validation_rows = {
        (int(row["layer"]), int(row["expert"]), row["selection"], row["rank"]): row
        for row in validation["results"]
    }
    rows = []
    for decision in decisions["decisions"]:
        layer, expert = int(decision["layer"]), int(decision["expert"])
        values = fit_validation["statistics"][layer][expert]
        fit = values["splits"]["fit"]
        validation_stats = values["splits"]["validation"]
        audit_stats = audit["statistics"][layer][expert]
        down_proj = values["down_proj"].to(dtype=torch.float64)
        fit_covariance = fit["covariance"].to(dtype=torch.float64)
        audit_covariance = audit_stats["covariance"].to(dtype=torch.float64)
        gate_square_sum = float(fit["gate_square_sum"])
        gate_fourth_sum = float(fit["gate_fourth_sum"])
        effective_n = gate_square_sum * gate_square_sum / max(gate_fourth_sum, 1.0e-30)
        for selection, raw_keep in decision["keep_indices"].items():
            keep = torch.tensor(raw_keep, dtype=torch.long)
            alpha = float(decisions["shared_alpha_by_selection"][selection]["alpha"])
            for rank_name in RANKS:
                metric = evaluate(
                    down_proj,
                    fit_covariance,
                    audit_covariance,
                    keep,
                    alpha=alpha,
                    rank_name=rank_name,
                )
                validation_metric = validation_rows[(layer, expert, selection, rank_name)]["metrics"]
                rows.append({
                    "layer": layer,
                    "expert": expert,
                    "selection": selection,
                    "rank": rank_name,
                    "alpha": alpha,
                    "route_count_fit": int(fit["route_count"]),
                    "route_count_validation": int(validation_stats["route_count"]),
                    "route_count_audit": int(audit_stats["route_count"]),
                    "effective_n_fit": effective_n,
                    "low_support": int(fit["route_count"]) < 1536 or effective_n / keep.numel() < 2.0,
                    "validation_error": float(validation_metric["validation_error"]),
                    **metric,
                })

    summary = {}
    for selection in sorted({row["selection"] for row in rows}):
        summary[selection] = {}
        for rank_name in RANKS:
            values = [row for row in rows if row["selection"] == selection and row["rank"] == rank_name]
            summary[selection][rank_name] = {
                "experts": len(values),
                "median_audit_error": median([row["audit_error"] for row in values]),
                "median_validation_error": median([row["validation_error"] for row in values]),
                "median_audit_r2_pruned": median([row["audit_r2_pruned"] for row in values]),
                "median_compensation_ratio": median([row["compensation_frobenius_ratio"] for row in values]),
                "generalization_ok": sum(
                    row["audit_error"] / max(row["validation_error"], 1.0e-12) <= 1.5 for row in values
                ),
            }

    primary = choose_primary_selection(summary)
    pair_keys = sorted({(row["layer"], row["expert"]) for row in rows if row["selection"] == primary})
    row_map = {
        (row["layer"], row["expert"], row["selection"], row["rank"]): row
        for row in rows
    }
    improvements = []
    wins = 0
    low_support_deltas = []
    rank64_retentions = []
    for layer, expert in pair_keys:
        method = row_map[(layer, expert, primary, "full")]
        baselines = [
            row_map[(layer, expert, selection, "full")]
            for selection in ("rms", "tail", "ramp_e0", "pair_corr")
        ]
        baseline = min(baselines, key=lambda row: row["audit_error"])
        improvements.append((baseline["audit_error"] - method["audit_error"]) / max(baseline["audit_error"], 1.0e-12))
        wins += int(method["audit_error"] < baseline["audit_error"])
        rank64 = row_map[(layer, expert, primary, "rank64")]
        full_reduction = method["audit_none_error"] - method["audit_error"]
        if full_reduction > 1.0e-12:
            rank64_retentions.append(
                (method["audit_none_error"] - rank64["audit_error"]) / full_reduction
            )
        if method["low_support"]:
            low_support_deltas.append(method["audit_error"] - baseline["audit_error"])

    criteria = {
        "primary_selection": primary,
        "median_relative_improvement_vs_best_baseline": median(improvements),
        "paired_bootstrap_95_ci": paired_bootstrap_median_ci(improvements),
        "wins_vs_best_baseline": wins,
        "median_audit_r2_pruned": summary[primary]["full"]["median_audit_r2_pruned"],
        "median_rank64_error_reduction_retention": median(rank64_retentions),
        "rank64_retention_eligible_experts": len(rank64_retentions),
        "generalization_ok_count": summary[primary]["full"]["generalization_ok"],
        "low_support_experts": len(low_support_deltas),
        "low_support_nonpositive_delta_count": sum(delta <= 0.0 for delta in low_support_deltas),
    }
    criteria.update({
        "improvement_at_least_8pct": criteria["median_relative_improvement_vs_best_baseline"] >= 0.08,
        "wins_at_least_18_of_24": wins >= 18,
        "bootstrap_ci_lower_above_zero": criteria["paired_bootstrap_95_ci"][0] > 0.0,
        "median_r2_at_least_20pct": criteria["median_audit_r2_pruned"] >= 0.20,
        "rank64_retention_at_least_80pct": criteria["median_rank64_error_reduction_retention"] >= 0.80,
        "generalization_ok_at_least_20_of_24": criteria["generalization_ok_count"] >= 20,
        "low_support_no_systematic_negative": not low_support_deltas or median(low_support_deltas) <= 0.0,
    })
    required = (
        "improvement_at_least_8pct",
        "wins_at_least_18_of_24",
        "bootstrap_ci_lower_above_zero",
        "median_r2_at_least_20pct",
        "rank64_retention_at_least_80pct",
        "generalization_ok_at_least_20_of_24",
        "low_support_no_systematic_negative",
    )
    if all(criteria[name] for name in required):
        outcome = "GO_STAGE_B"
    elif 0.03 <= criteria["median_relative_improvement_vs_best_baseline"] < 0.08 and criteria["bootstrap_ci_lower_above_zero"]:
        outcome = "REVISE_LOCAL_SELECTION"
    else:
        outcome = "NO_GO_STAGE_A"

    output = {
        "schema_version": 1,
        "experiment": "RAMP-E1",
        "stage": "A",
        "outcome": outcome,
        "fit_validation_cache_sha256": file_sha256(args.fit_validation_cache),
        "audit_cache_sha256": file_sha256(args.audit_cache),
        "decision_file_sha256": file_sha256(args.decision_file),
        "criteria": criteria,
        "summary": summary,
        "rows": rows,
    }
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_results.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RAMP-E1 Stage-A Audit Summary",
        "",
        f"Frozen outcome: **{outcome}**.",
        "",
        f"Primary validation-selected conditional method: `{primary}`.",
        "",
        "| selection | rank | median audit error | median validation error | median audit R2 | generalization <= 1.5 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for selection in PRIMARY_SELECTIONS:
        for rank_name in RANKS:
            values = summary[selection][rank_name]
            lines.append(
                f"| {selection} | {rank_name} | {values['median_audit_error']:.6f} | "
                f"{values['median_validation_error']:.6f} | {values['median_audit_r2_pruned']:.6f} | "
                f"{values['generalization_ok']}/24 |"
            )
    lines.extend([
        "",
        "## Stage-A Criteria",
        "",
        f"- Median relative improvement over best RMS/Tail/RAMP-E0/pair-correlation baseline: `{criteria['median_relative_improvement_vs_best_baseline']:.6f}`.",
        f"- Paired bootstrap 95% CI: `{criteria['paired_bootstrap_95_ci']}`.",
        f"- Expert wins: `{wins}/24`.",
        f"- Median audit residual R2: `{criteria['median_audit_r2_pruned']:.6f}`.",
        f"- Rank-64 error-reduction retention: `{criteria['median_rank64_error_reduction_retention']:.6f}`.",
        f"- Generalization ratio <= 1.5: `{criteria['generalization_ok_count']}/24`.",
        f"- Low-support experts: `{criteria['low_support_experts']}`.",
        "",
        "This is an expert-level mechanism result. It does not establish PPL or downstream quality.",
    ])
    args.output_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fieldnames = list(rows[0])
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output_results)
    print(args.output_summary)
    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())