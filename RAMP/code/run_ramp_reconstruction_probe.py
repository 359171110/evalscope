from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from ramp_reconstruction import (
    fit_rank_limited_compensation,
    fit_ridge_compensation,
    normalized_output_error,
    ramp_conditional_residual_selection,
    rank_rms_channels,
    rank_tail_channels,
)


METHODS = (
    "random_none",
    "random_ridge",
    "rms_none",
    "rms_ridge",
    "tail_none",
    "tail_ridge",
    "ramp_none",
    "ramp_rank16",
    "ramp_ridge",
)
ALPHA_GRID = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAMP-E0 offline reconstruction decisions.")
    parser.add_argument("--covariance-cache", type=Path, required=True)
    parser.add_argument("--output-decision", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--keep-count", type=int, default=384)
    parser.add_argument("--anchor-count", type=int, default=38)
    parser.add_argument("--tail-lambda", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    return parser.parse_args()


def _load_cache(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("covariance cache must contain a dictionary payload.")
    if payload.get("experiment") != "RAMP-E0":
        raise ValueError("covariance cache is not an RAMP-E0 artifact.")
    if payload.get("audit_collected") is True:
        raise ValueError("probe selection must run before audit collection.")
    if payload.get("smoke_only") is True:
        raise ValueError("smoke covariance artifacts cannot produce formal decisions.")
    if "fit" not in payload.get("split", "") or "validation" not in payload.get("split", ""):
        raise ValueError("probe requires both fit and validation statistics.")
    return payload


def _split_stats(payload: dict, layer_idx: int, expert_idx: int, split_name: str) -> dict:
    values = payload["statistics"][int(layer_idx)][int(expert_idx)]
    stats = values["splits"].get(split_name)
    if not stats or int(stats.get("route_count", 0)) <= 0:
        raise ValueError(f"missing routed statistics for layer {layer_idx}, expert {expert_idx}, split {split_name}.")
    return stats


def _fit_regularization(covariance: torch.Tensor, alpha: float) -> float:
    return float(alpha) * float(torch.trace(covariance).item()) / float(covariance.shape[0])


def _choose_alpha(
    down_proj: torch.Tensor,
    fit_covariance: torch.Tensor,
    validation_covariance: torch.Tensor,
    keep_indices: torch.Tensor,
) -> tuple[float, dict[float, float]]:
    weights = down_proj.detach().to(dtype=torch.float64, device="cpu")
    keep = keep_indices.detach().to(dtype=torch.long, device="cpu")
    pruned_mask = torch.ones(weights.shape[1], dtype=torch.bool)
    pruned_mask[keep] = False
    pruned = torch.nonzero(pruned_mask, as_tuple=False).flatten()
    cov_kk = fit_covariance.index_select(0, keep).index_select(1, keep)
    cov_kk = 0.5 * (cov_kk + cov_kk.transpose(0, 1))
    eigvals, eigvecs = torch.linalg.eigh(cov_kk)
    target = weights.index_select(1, pruned).matmul(
        fit_covariance.index_select(0, pruned).index_select(1, keep)
    )
    rotated_target = target.matmul(eigvecs)
    scores = {}
    for alpha in ALPHA_GRID:
        regularization = _fit_regularization(fit_covariance, alpha)
        inverse_scale = (eigvals + regularization).clamp_min(1.0e-30).reciprocal()
        delta = (rotated_target * inverse_scale[None, :]).matmul(eigvecs.transpose(0, 1))
        effective = weights.index_select(1, keep) + delta
        scores[alpha] = normalized_output_error(
            weights,
            validation_covariance,
            keep,
            effective,
        )
    best_alpha = min(ALPHA_GRID, key=lambda alpha: (scores[alpha], alpha))
    return float(best_alpha), scores


def choose_shared_alpha(score_tables: list[dict[float, float]]) -> tuple[float, dict[float, float]]:
    """Choose one alpha across experts using median validation error."""

    if not score_tables:
        raise ValueError("score_tables must be non-empty.")
    aggregated = {
        alpha: float(torch.tensor([table[alpha] for table in score_tables], dtype=torch.float64).median().item())
        for alpha in ALPHA_GRID
    }
    best_alpha = min(ALPHA_GRID, key=lambda alpha: (aggregated[alpha], alpha))
    return float(best_alpha), aggregated


def _random_keep(channel_count: int, keep_count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(channel_count, generator=generator)[: int(keep_count)].sort().values


def _evaluate_method(
    method: str,
    down_proj: torch.Tensor,
    fit_stats: dict,
    validation_stats: dict,
    *,
    keep_indices: torch.Tensor,
    alpha: float | None,
) -> dict[str, object]:
    fit_covariance = fit_stats["covariance"].to(dtype=torch.float64)
    validation_covariance = validation_stats["covariance"].to(dtype=torch.float64)
    fit_regularization = 0.0 if alpha is None else _fit_regularization(fit_covariance, alpha)
    fit_effective = down_proj.index_select(1, keep_indices)
    validation_effective = fit_effective
    delta = torch.zeros_like(fit_effective)
    if alpha is not None:
        validation_effective, delta = fit_ridge_compensation(
            down_proj,
            fit_covariance,
            keep_indices,
            regularization=fit_regularization,
        )
    if method == "ramp_rank16":
        validation_effective, delta = fit_rank_limited_compensation(
            down_proj,
            fit_covariance,
            keep_indices,
            regularization=fit_regularization,
            rank=16,
        )
    return {
        "method": method,
        "keep_indices": keep_indices.tolist(),
        "alpha": alpha,
        "fit_error": normalized_output_error(down_proj, fit_covariance, keep_indices, validation_effective),
        "validation_error": normalized_output_error(
            down_proj,
            validation_covariance,
            keep_indices,
            validation_effective,
        ),
        "compensation_frobenius_ratio": float(
            delta.norm().item() / down_proj.index_select(1, keep_indices).norm().clamp_min(1.0e-12).item()
        ),
    }


def main() -> int:
    args = parse_args()
    if not 0 < int(args.keep_count):
        raise ValueError("keep-count must be positive.")
    payload = _load_cache(args.covariance_cache)
    results: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    score_tables_by_selection: dict[str, list[dict[float, float]]] = defaultdict(list)
    for layer_key, experts in sorted(payload["statistics"].items(), key=lambda item: int(item[0])):
        layer_idx = int(layer_key)
        for expert_key, values in sorted(experts.items(), key=lambda item: int(item[0])):
            expert_idx = int(expert_key)
            fit_stats = _split_stats(payload, layer_idx, expert_idx, "fit")
            validation_stats = _split_stats(payload, layer_idx, expert_idx, "validation")
            down_proj = values["down_proj"].to(dtype=torch.float64)
            channel_count = int(down_proj.shape[1])
            if int(args.keep_count) > channel_count:
                raise ValueError("keep-count exceeds expert intermediate width.")
            fit_covariance = fit_stats["covariance"].to(dtype=torch.float64)
            rms_rank = rank_rms_channels(
                down_proj,
                fit_stats["unweighted_square_sum"],
                route_count=int(fit_stats["route_count"]),
            )
            tail_rank = rank_tail_channels(
                down_proj,
                fit_stats["unweighted_square_sum"],
                fit_stats["max_abs"],
                route_count=int(fit_stats["route_count"]),
                tail_lambda=float(args.tail_lambda),
            )
            ramp_keep = ramp_conditional_residual_selection(
                down_proj,
                fit_covariance,
                keep_count=int(args.keep_count),
                anchor_count=int(args.anchor_count),
                regularization=_fit_regularization(fit_covariance, 1.0e-4),
            )
            selections = {
                "rms": rms_rank[: int(args.keep_count)],
                "tail": tail_rank[: int(args.keep_count)],
                "ramp": ramp_keep,
            }
            random_selections = {
                f"random_{seed}": _random_keep(channel_count, int(args.keep_count), seed)
                for seed in args.random_seeds
            }
            alpha_by_method = {}
            for name, keep in {**selections, **random_selections}.items():
                alpha, alpha_scores = _choose_alpha(
                    down_proj,
                    fit_covariance,
                    validation_stats["covariance"].to(dtype=torch.float64),
                    keep,
                )
                alpha_by_method[name] = {"alpha": alpha, "scores": alpha_scores}
                score_tables_by_selection[name].append(alpha_scores)
                results.append(
                    {
                        "layer": layer_idx,
                        "expert": expert_idx,
                        "selection": name,
                        "route_count_fit": int(fit_stats["route_count"]),
                        "route_count_validation": int(validation_stats["route_count"]),
                        "no_compensation": _evaluate_method(
                            f"{name}_none", down_proj, fit_stats, validation_stats, keep_indices=keep, alpha=None
                        ),
                        "per_expert_ridge": _evaluate_method(
                            f"{name}_ridge", down_proj, fit_stats, validation_stats, keep_indices=keep, alpha=alpha
                        ),
                    }
                )
            decisions.append(
                {
                    "layer": layer_idx,
                    "expert": expert_idx,
                    "keep_count": int(args.keep_count),
                    "anchor_count": int(args.anchor_count),
                    "ramp_keep_indices": ramp_keep.tolist(),
                    "rms_keep_indices": selections["rms"].tolist(),
                    "tail_keep_indices": selections["tail"].tolist(),
                    "random_keep_indices": {name: keep.tolist() for name, keep in random_selections.items()},
                    "alpha_by_method": alpha_by_method,
                }
            )
    shared_alpha_by_selection = {}
    for selection, score_tables in score_tables_by_selection.items():
        alpha, median_scores = choose_shared_alpha(score_tables)
        shared_alpha_by_selection[selection] = {
            "alpha": alpha,
            "median_validation_error_by_alpha": median_scores,
        }
    for row in results:
        layer_idx = int(row["layer"])
        expert_idx = int(row["expert"])
        selection = str(row["selection"])
        values = payload["statistics"][layer_idx][expert_idx]
        fit_stats = _split_stats(payload, layer_idx, expert_idx, "fit")
        validation_stats = _split_stats(payload, layer_idx, expert_idx, "validation")
        down_proj = values["down_proj"].to(dtype=torch.float64)
        keep = torch.tensor(row["no_compensation"]["keep_indices"], dtype=torch.long)
        shared_alpha = float(shared_alpha_by_selection[selection]["alpha"])
        row["ridge"] = _evaluate_method(
            f"{selection}_ridge",
            down_proj,
            fit_stats,
            validation_stats,
            keep_indices=keep,
            alpha=shared_alpha,
        )
        if selection == "ramp":
            row["rank16"] = _evaluate_method(
                "ramp_rank16",
                down_proj,
                fit_stats,
                validation_stats,
                keep_indices=keep,
                alpha=shared_alpha,
            )
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    args.output_decision.parent.mkdir(parents=True, exist_ok=True)
    args.output_results.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "RAMP-E0",
                "covariance_cache": str(args.covariance_cache.resolve()),
                "covariance_cache_sha256": file_sha256(args.covariance_cache),
                "primary_alpha_scope": "shared_across_24_experts_per_selection",
                "shared_alpha_by_selection": shared_alpha_by_selection,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.output_decision.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "RAMP-E0",
                "frozen_before_audit": True,
                "covariance_cache_sha256": file_sha256(args.covariance_cache),
                "primary_alpha_scope": "shared_across_24_experts_per_selection",
                "shared_alpha_by_selection": shared_alpha_by_selection,
                "decisions": decisions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_results)
    print(args.output_decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())