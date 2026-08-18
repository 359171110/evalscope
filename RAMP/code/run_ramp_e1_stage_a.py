from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from ramp_reconstruction import (
    conditional_activation_selection,
    fit_rank_limited_compensation,
    fit_ridge_compensation,
    normalized_output_error,
    pairwise_output_correlation_selection,
    ramp_conditional_residual_selection,
    rank_rms_channels,
    rank_tail_channels,
)


ALPHA_GRID = (1.0e-3, 1.0e-2, 1.0e-1, 3.0e-1, 1.0, 3.0, 10.0, 30.0)
RANKS = (16, 32, 64, 128, None)
STABLE_ANCHOR_COUNTS = (0, 19, 38, 77)
SELECTIONS = (
    "rms",
    "tail",
    "ramp_e0",
    "pair_corr",
    "conditional_activation",
    "conditional_output",
    "conditional_stable",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze RAMP-E1 stage-A selector and compensation decisions.")
    parser.add_argument("--covariance-cache", type=Path, required=True)
    parser.add_argument("--output-decision", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--keep-count", type=int, default=384)
    parser.add_argument("--tail-lambda", type=float, default=0.5)
    parser.add_argument("--selection-regularization-alpha", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    return parser.parse_args()


def load_cache(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("experiment") != "RAMP-E1":
        raise ValueError("covariance cache must be a RAMP-E1 artifact.")
    if payload.get("smoke_only") is True or payload.get("audit_collected") is True:
        raise ValueError("stage-A decisions require formal fit/validation statistics only.")
    if payload.get("split") != "fit_validation":
        raise ValueError("stage-A decisions require a fit_validation covariance cache.")
    return payload


def split_stats(payload: dict, layer: int, expert: int, name: str) -> dict:
    stats = payload["statistics"][layer][expert]["splits"].get(name)
    if not stats or int(stats.get("route_count", 0)) <= 0:
        raise ValueError(f"missing {name} statistics for layer {layer}, expert {expert}.")
    return stats


def fit_regularization(covariance: torch.Tensor, alpha: float) -> float:
    return float(alpha) * float(torch.trace(covariance).item()) / float(covariance.shape[0])


def random_keep(channels: int, keep_count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(channels, generator=generator)[:keep_count].sort().values


def choose_alpha(
    down_proj: torch.Tensor,
    fit_covariance: torch.Tensor,
    validation_covariance: torch.Tensor,
    keep: torch.Tensor,
) -> tuple[float, dict[float, float]]:
    scores = {}
    for alpha in ALPHA_GRID:
        regularization = fit_regularization(fit_covariance, alpha)
        effective, _ = fit_ridge_compensation(
            down_proj, fit_covariance, keep, regularization=regularization
        )
        scores[alpha] = normalized_output_error(down_proj, validation_covariance, keep, effective)
    best = min(ALPHA_GRID, key=lambda alpha: (scores[alpha], alpha))
    return float(best), scores


def choose_shared_alpha(score_tables: list[dict[float, float]]) -> tuple[float, dict[float, float]]:
    aggregated = {
        alpha: float(torch.tensor([table[alpha] for table in score_tables], dtype=torch.float64).median())
        for alpha in ALPHA_GRID
    }
    best = min(ALPHA_GRID, key=lambda alpha: (aggregated[alpha], alpha))
    return float(best), aggregated


def evaluate(
    down_proj: torch.Tensor,
    fit_covariance: torch.Tensor,
    validation_covariance: torch.Tensor,
    keep: torch.Tensor,
    *,
    alpha: float | None,
    rank: int | None,
) -> dict[str, float]:
    none = down_proj.index_select(1, keep)
    if alpha is None:
        effective, delta = none, torch.zeros_like(none)
    elif rank is None:
        effective, delta = fit_ridge_compensation(
            down_proj, fit_covariance, keep, regularization=fit_regularization(fit_covariance, alpha)
        )
    else:
        effective, delta = fit_rank_limited_compensation(
            down_proj,
            fit_covariance,
            keep,
            regularization=fit_regularization(fit_covariance, alpha),
            rank=rank,
        )
    fit_error = normalized_output_error(down_proj, fit_covariance, keep, effective)
    validation_error = normalized_output_error(down_proj, validation_covariance, keep, effective)
    none_validation = normalized_output_error(down_proj, validation_covariance, keep, none)
    return {
        "fit_error": fit_error,
        "validation_error": validation_error,
        "validation_none_error": none_validation,
        "validation_r2_pruned": 1.0 - validation_error / max(none_validation, 1.0e-12),
        "compensation_frobenius_ratio": float(delta.norm() / none.norm().clamp_min(1.0e-12)),
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.keep_count <= 768:
        raise ValueError("keep-count must be in [1, 768].")
    payload = load_cache(args.covariance_cache)
    results = []
    decisions = []
    alpha_tables: dict[str, list[dict[float, float]]] = defaultdict(list)
    stable_alpha_tables: dict[int, list[dict[float, float]]] = defaultdict(list)
    for layer_key, experts in sorted(payload["statistics"].items(), key=lambda item: int(item[0])):
        layer = int(layer_key)
        for expert_key, values in sorted(experts.items(), key=lambda item: int(item[0])):
            expert = int(expert_key)
            fit = split_stats(payload, layer, expert, "fit")
            validation = split_stats(payload, layer, expert, "validation")
            down_proj = values["down_proj"].to(dtype=torch.float64)
            fit_covariance = fit["covariance"].to(dtype=torch.float64)
            validation_covariance = validation["covariance"].to(dtype=torch.float64)
            channels = int(down_proj.shape[1])
            if channels != 768:
                raise ValueError(f"expected Qwen3 expert width 768, got {channels}.")
            rms = rank_rms_channels(down_proj, fit["unweighted_square_sum"], route_count=int(fit["route_count"]))[: args.keep_count]
            tail = rank_tail_channels(
                down_proj,
                fit["unweighted_square_sum"],
                fit["max_abs"],
                route_count=int(fit["route_count"]),
                tail_lambda=args.tail_lambda,
            )[: args.keep_count]
            selector_regularization = fit_regularization(fit_covariance, args.selection_regularization_alpha)
            selections = {
                "rms": rms,
                "tail": tail,
                "ramp_e0": ramp_conditional_residual_selection(
                    down_proj, fit_covariance, keep_count=args.keep_count, anchor_count=38,
                    regularization=selector_regularization,
                ),
                "pair_corr": pairwise_output_correlation_selection(
                    down_proj, fit_covariance, keep_count=args.keep_count,
                ),
                "conditional_activation": conditional_activation_selection(
                    fit_covariance, keep_count=args.keep_count, regularization=selector_regularization,
                ),
                "conditional_output": ramp_conditional_residual_selection(
                    down_proj, fit_covariance, keep_count=args.keep_count, anchor_count=0,
                    regularization=selector_regularization,
                ),
            }
            stable_candidates = {}
            for anchor_count in STABLE_ANCHOR_COUNTS:
                keep = ramp_conditional_residual_selection(
                    down_proj,
                    fit_covariance,
                    keep_count=args.keep_count,
                    anchor_count=anchor_count,
                    regularization=selector_regularization,
                )
                alpha, scores = choose_alpha(down_proj, fit_covariance, validation_covariance, keep)
                stable_alpha_tables[anchor_count].append(scores)
                stable_candidates[str(anchor_count)] = {
                    "keep_indices": keep.tolist(),
                    "per_expert_alpha": alpha,
                    "validation_error_by_alpha": scores,
                }
            random_selections = {
                f"random_{seed}": random_keep(channels, args.keep_count, seed)
                for seed in args.random_seeds
            }
            selections.update(random_selections)
            decision = {
                "layer": layer,
                "expert": expert,
                "keep_count": args.keep_count,
                "route_count_fit": int(fit["route_count"]),
                "route_count_validation": int(validation["route_count"]),
                "gate_square_sum_fit": float(fit["gate_square_sum"]),
                "gate_fourth_sum_fit": float(fit["gate_fourth_sum"]),
                "keep_indices": {name: keep.tolist() for name, keep in selections.items()},
                "alpha_by_selection": {},
                "conditional_stable_candidates": stable_candidates,
            }
            for name, keep in selections.items():
                alpha, scores = choose_alpha(down_proj, fit_covariance, validation_covariance, keep)
                alpha_tables[name].append(scores)
                decision["alpha_by_selection"][name] = {"alpha": alpha, "scores": scores}
            decisions.append(decision)

    shared_alphas = {}
    for name, tables in alpha_tables.items():
        alpha, scores = choose_shared_alpha(tables)
        shared_alphas[name] = {"alpha": alpha, "median_validation_error_by_alpha": scores}
    stable_grid = {}
    for anchor_count, tables in stable_alpha_tables.items():
        _, median_scores = choose_shared_alpha(tables)
        stable_grid[anchor_count] = median_scores
    stable_anchor, stable_alpha = min(
        (
            (anchor_count, alpha)
            for anchor_count in STABLE_ANCHOR_COUNTS
            for alpha in ALPHA_GRID
        ),
        key=lambda item: (stable_grid[item[0]][item[1]], item[0], item[1]),
    )
    shared_alphas["conditional_stable"] = {
        "alpha": float(stable_alpha),
        "anchor_count": int(stable_anchor),
        "median_validation_error": float(stable_grid[stable_anchor][stable_alpha]),
        "median_validation_error_by_anchor_and_alpha": stable_grid,
    }
    for decision in decisions:
        candidate = decision["conditional_stable_candidates"][str(stable_anchor)]
        decision["keep_indices"]["conditional_stable"] = candidate["keep_indices"]
        decision["alpha_by_selection"]["conditional_stable"] = {
            "alpha": float(stable_alpha),
            "scores": candidate["validation_error_by_alpha"],
        }
    for decision in decisions:
        layer, expert = int(decision["layer"]), int(decision["expert"])
        values = payload["statistics"][layer][expert]
        fit = split_stats(payload, layer, expert, "fit")
        validation = split_stats(payload, layer, expert, "validation")
        down_proj = values["down_proj"].to(dtype=torch.float64)
        for name, raw_keep in decision["keep_indices"].items():
            keep = torch.tensor(raw_keep, dtype=torch.long)
            alpha = float(shared_alphas[name]["alpha"])
            for rank in (None, 16, 32, 64, 128):
                rank_name = "full" if rank is None else f"rank{rank}"
                metric = evaluate(
                    down_proj,
                    fit["covariance"].to(dtype=torch.float64),
                    validation["covariance"].to(dtype=torch.float64),
                    keep,
                    alpha=alpha,
                    rank=rank,
                )
                results.append({
                    "layer": layer,
                    "expert": expert,
                    "selection": name,
                    "rank": rank_name,
                    "alpha": alpha,
                    "metrics": metric,
                })
        decision["shared_alpha_by_selection"] = shared_alphas

    output = {
        "schema_version": 1,
        "experiment": "RAMP-E1",
        "stage": "A",
        "covariance_cache": str(args.covariance_cache.resolve()),
        "covariance_cache_sha256": file_sha256(args.covariance_cache),
        "shared_alpha_by_selection": shared_alphas,
        "results": results,
    }
    decision_output = {
        "schema_version": 1,
        "experiment": "RAMP-E1",
        "stage": "A",
        "frozen_before_audit": True,
        "covariance_cache_sha256": file_sha256(args.covariance_cache),
        "shared_alpha_by_selection": shared_alphas,
        "decisions": decisions,
        "test_metrics_used_for_selection": False,
    }
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    args.output_decision.parent.mkdir(parents=True, exist_ok=True)
    args.output_results.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_decision.write_text(json.dumps(decision_output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_results)
    print(args.output_decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())