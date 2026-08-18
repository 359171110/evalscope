from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.committee_regret import build_frontier_regret_floors
from src.static_expert_pruning import (
    allocate_static_prefix_widths,
    allocate_static_prefix_widths_per_layer,
)
from src.tail_risk import (
    build_consensus_rare_event_risk_floors,
    build_rare_event_risk_floors,
)
from src.utility_rebinding import (
    aggregate_unique_contribution_folds,
    aggregate_output_saliency_folds,
    fuse_expert_utility_with_output_saliency,
    rebind_expert_utility_to_coverage,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(table: dict, layer_id: int):
    if layer_id in table:
        return table[layer_id]
    return table[str(layer_id)]


def _coverage(payload: dict, layer_ids: list[int]) -> torch.Tensor:
    return torch.stack(
        [
            _lookup(payload["table"], layer_id)["block_coverage_scores"].float()
            for layer_id in layer_ids
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebind conditional expert utility to a tail-aware channel ranking."
    )
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--reference-channel-cache", type=Path, required=True)
    parser.add_argument("--tail-channel-cache", type=Path, required=True)
    parser.add_argument(
        "--risk-floor-cache",
        type=Path,
        default=None,
        help="Optional train-only cache supplying expert risk while tail cache supplies coverage.",
    )
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float, required=True)
    parser.add_argument(
        "--allocation-scope",
        choices=("global", "per_layer"),
        default="global",
    )
    parser.add_argument(
        "--retained-blocks-per-layer",
        type=int,
        default=None,
    )
    parser.add_argument("--risk-floor-min-width", type=int, default=0)
    parser.add_argument("--risk-floor-early-layers", type=int, default=4)
    parser.add_argument("--risk-floor-quantile", type=float, default=0.995)
    parser.add_argument("--risk-floor-relative-max", type=float, default=0.10)
    parser.add_argument(
        "--risk-floor-consensus-cache",
        action="append",
        type=Path,
        default=[],
        help="Train-only tail cache used for cross-interval risk voting; repeatable.",
    )
    parser.add_argument("--risk-floor-min-votes", type=int, default=None)
    parser.add_argument(
        "--reference-coverage-consensus-cache",
        action="append",
        type=Path,
        default=[],
        help="Optional repeated train-only RMS caches for mean coverage consensus.",
    )
    parser.add_argument(
        "--tail-coverage-consensus-cache",
        action="append",
        type=Path,
        default=[],
        help="Optional repeated train-only tail caches for mean coverage consensus.",
    )
    parser.add_argument(
        "--teacher-consensus-cache",
        action="append",
        type=Path,
        default=[],
        help="Optional repeated train-only Conditional-Dual teacher caches.",
    )
    parser.add_argument(
        "--teacher-consensus-std-penalty",
        type=float,
        default=0.0,
        help="Lower-confidence-bound penalty applied to teacher block values.",
    )
    parser.add_argument(
        "--output-saliency-cache",
        action="append",
        type=Path,
        default=[],
        help=(
            "Train-only expert output-contribution cache; repeat to build a "
            "per-fold-normalized mean consensus."
        ),
    )
    parser.add_argument("--output-saliency-beta", type=float, default=0.0)
    parser.add_argument("--output-saliency-floor-min-width", type=int, default=0)
    parser.add_argument(
        "--output-saliency-floor-quantile", type=float, default=0.995
    )
    parser.add_argument(
        "--output-saliency-floor-relative-max", type=float, default=0.10
    )
    parser.add_argument("--unique-contribution-floor-min-width", type=int, default=0)
    parser.add_argument(
        "--unique-contribution-floor-quantile", type=float, default=0.995
    )
    parser.add_argument(
        "--unique-contribution-floor-relative-max", type=float, default=0.0
    )
    parser.add_argument(
        "--unique-contribution-fold-aggregation",
        choices=("mean", "minimum"),
        default="minimum",
    )
    parser.add_argument("--frontier-reference-profile", type=Path, default=None)
    parser.add_argument(
        "--frontier-regret-cache",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--frontier-regret-floor-quantile", type=float, default=0.995
    )
    parser.add_argument("--frontier-regret-width-increment", type=int, default=1)
    parser.add_argument(
        "--frontier-regret-fold-aggregation",
        choices=("mean", "minimum"),
        default="minimum",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = float(args.target_pruning_ratio)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target-pruning-ratio must be in [0, 1].")

    teacher = torch.load(args.teacher_cache, map_location="cpu", weights_only=True)
    reference = torch.load(
        args.reference_channel_cache, map_location="cpu", weights_only=True
    )
    tail = torch.load(args.tail_channel_cache, map_location="cpu", weights_only=True)
    risk_source_path = (
        args.tail_channel_cache
        if args.risk_floor_cache is None
        else args.risk_floor_cache
    )
    risk_source = (
        tail
        if args.risk_floor_cache is None
        else torch.load(args.risk_floor_cache, map_location="cpu", weights_only=True)
    )
    sequence_length = int(teacher.get("sequence_length", -1))
    if sequence_length <= 0:
        raise ValueError("teacher cache must use a positive sequence length.")
    for name, payload in (
        ("teacher", teacher),
        ("reference channel", reference),
        ("tail channel", tail),
        ("risk floor", risk_source),
    ):
        if payload.get("split") != "train":
            raise ValueError(f"{name} cache must use the train split.")
        if int(payload.get("sequence_length", -1)) != sequence_length:
            raise ValueError(f"{name} cache sequence length does not match the teacher cache.")
    if teacher.get("test_metrics_used") is not False:
        raise ValueError("teacher cache must be independent of test metrics.")
    if teacher.get("parent_mode") != "dual":
        raise ValueError("tail-risk profile teacher must use parent_mode=dual.")

    values_table = teacher["unconditional_block_values"]
    layer_ids = sorted(int(layer_id) for layer_id in values_table)
    old_values = torch.stack(
        [_lookup(values_table, layer_id).float() for layer_id in layer_ids]
    )
    teacher_consensus = list(getattr(args, "teacher_consensus_cache", []))
    teacher_consensus_provenance = []
    if teacher_consensus:
        teacher_folds = []
        for teacher_path in teacher_consensus:
            teacher_fold = torch.load(
                teacher_path, map_location="cpu", weights_only=True
            )
            if teacher_fold.get("split") != "train":
                raise ValueError("teacher consensus caches must use the train split.")
            if teacher_fold.get("test_metrics_used") is not False:
                raise ValueError("teacher consensus caches must be test-independent.")
            if teacher_fold.get("parent_mode") != "dual":
                raise ValueError("teacher consensus caches must use parent_mode=dual.")
            fold_values = torch.stack(
                [
                    _lookup(teacher_fold["unconditional_block_values"], layer_id).float()
                    for layer_id in layer_ids
                ]
            )
            if fold_values.shape != old_values.shape:
                raise ValueError("teacher consensus cache shapes do not match.")
            teacher_folds.append(fold_values)
            teacher_consensus_provenance.append(
                {
                    "path": str(teacher_path.resolve()),
                    "sha256": file_sha256(teacher_path),
                    "calibration_token_offset": teacher_fold.get(
                        "calibration_token_offset", 0
                    ),
                    "calibration_token_end": teacher_fold.get("calibration_token_end"),
                }
            )
        stacked_teacher_values = torch.stack(teacher_folds)
        std_penalty = float(getattr(args, "teacher_consensus_std_penalty", 0.0))
        if std_penalty < 0.0:
            raise ValueError("teacher-consensus-std-penalty must be non-negative.")
        old_values = (
            stacked_teacher_values.mean(dim=0)
            - std_penalty * stacked_teacher_values.std(dim=0, unbiased=False)
        ).clamp_min(0.0)
    old_coverage = _coverage(reference, layer_ids)
    new_coverage = _coverage(tail, layer_ids)
    coverage_consensus_provenance = {"reference": [], "tail": []}
    reference_consensus = list(getattr(args, "reference_coverage_consensus_cache", []))
    tail_consensus = list(getattr(args, "tail_coverage_consensus_cache", []))
    if bool(reference_consensus) != bool(tail_consensus):
        raise ValueError(
            "reference and tail coverage consensus caches must be supplied together."
        )
    if reference_consensus:
        if len(reference_consensus) != len(tail_consensus):
            raise ValueError(
                "reference and tail coverage consensus cache counts must match."
            )
        reference_folds = []
        tail_folds = []
        for reference_path, tail_path in zip(reference_consensus, tail_consensus):
            reference_fold = torch.load(
                reference_path, map_location="cpu", weights_only=True
            )
            tail_fold = torch.load(tail_path, map_location="cpu", weights_only=True)
            for name, payload in (("reference", reference_fold), ("tail", tail_fold)):
                if payload.get("split") != "train":
                    raise ValueError(f"{name} coverage consensus cache must use train split.")
                if int(payload.get("sequence_length", -1)) != sequence_length:
                    raise ValueError(
                        f"{name} coverage consensus cache sequence length does not match."
                    )
            reference_folds.append(_coverage(reference_fold, layer_ids))
            tail_folds.append(_coverage(tail_fold, layer_ids))
            coverage_consensus_provenance["reference"].append(
                {"path": str(reference_path.resolve()), "sha256": file_sha256(reference_path)}
            )
            coverage_consensus_provenance["tail"].append(
                {"path": str(tail_path.resolve()), "sha256": file_sha256(tail_path)}
            )
        old_coverage = torch.stack(reference_folds).mean(dim=0)
        new_coverage = torch.stack(tail_folds).mean(dim=0)
    rebound_values, expert_utility = rebind_expert_utility_to_coverage(
        old_values, old_coverage, new_coverage
    )
    output_saliency_provenance = None
    output_saliency_factor = None
    output_saliency_consensus = None
    unique_contribution_score = None
    co_route_uniqueness_folds = None
    co_route_formula = None
    output_saliency_paths = list(getattr(args, "output_saliency_cache", []))
    if output_saliency_paths:
        output_saliency_folds = []
        co_route_context_folds = []
        output_saliency_fold_provenance = []
        expected_formula = None
        expected_co_route_formula = None
        for raw_output_path in output_saliency_paths:
            output_path = raw_output_path.expanduser().resolve()
            output_payload = torch.load(
                output_path, map_location="cpu", weights_only=True
            )
            if output_payload.get("split") != "train":
                raise ValueError("output saliency cache must use the train split.")
            if int(output_payload.get("sequence_length", -1)) != sequence_length:
                raise ValueError("output saliency cache sequence length does not match.")
            if output_payload.get("test_metrics_used") is not False:
                raise ValueError("output saliency cache must be test-independent.")
            formula = output_payload.get("output_saliency_formula")
            if expected_formula is None:
                expected_formula = formula
            elif formula != expected_formula:
                raise ValueError("output saliency cache formulas do not match.")
            saliency_table = output_payload.get("expert_output_saliency_mean")
            if not isinstance(saliency_table, dict) or not saliency_table:
                raise ValueError(
                    "output saliency cache is missing expert_output_saliency_mean."
                )
            fold_saliency = torch.stack(
                [_lookup(saliency_table, layer_id).float() for layer_id in layer_ids]
            )
            if output_saliency_folds and (
                fold_saliency.shape != output_saliency_folds[0].shape
            ):
                raise ValueError("output saliency cache shapes do not match.")
            output_saliency_folds.append(fold_saliency)
            co_route_table = output_payload.get("expert_co_route_context")
            if isinstance(co_route_table, dict) and co_route_table:
                fold_context = torch.stack(
                    [_lookup(co_route_table, layer_id).float() for layer_id in layer_ids]
                )
                if fold_context.shape != (
                    fold_saliency.shape[0],
                    fold_saliency.shape[1],
                    fold_saliency.shape[1],
                ):
                    raise ValueError("co-route context cache shape does not match saliency.")
                current_co_route_formula = output_payload.get("co_route_formula")
                if expected_co_route_formula is None:
                    expected_co_route_formula = current_co_route_formula
                elif current_co_route_formula != expected_co_route_formula:
                    raise ValueError("co-route context cache formulas do not match.")
                co_route_context_folds.append(fold_context)
            elif int(getattr(args, "unique_contribution_floor_min_width", 0)) > 0:
                raise ValueError(
                    "unique contribution floors require expert_co_route_context in every cache."
                )
            output_saliency_fold_provenance.append(
                {
                    "path": str(output_path),
                    "sha256": file_sha256(output_path),
                    "calibration_token_offset": output_payload.get(
                        "calibration_token_offset", 0
                    ),
                    "calibration_token_end": output_payload.get(
                        "calibration_token_end"
                    ),
                    "split": output_payload.get("split"),
                }
            )
        output_saliency = aggregate_output_saliency_folds(
            torch.stack(output_saliency_folds)
        )
        output_saliency_consensus = output_saliency
        if co_route_context_folds:
            if len(co_route_context_folds) != len(output_saliency_folds):
                raise ValueError(
                    "unique contribution requires co-route context for every saliency fold."
                )
            unique_contribution_score, co_route_uniqueness_folds = (
                aggregate_unique_contribution_folds(
                    torch.stack(output_saliency_folds),
                    torch.stack(co_route_context_folds),
                    aggregation=str(
                        getattr(
                            args,
                            "unique_contribution_fold_aggregation",
                            "minimum",
                        )
                    ),
                )
            )
            co_route_formula = expected_co_route_formula
        expert_utility, output_saliency_factor = fuse_expert_utility_with_output_saliency(
            expert_utility,
            output_saliency,
            beta=float(getattr(args, "output_saliency_beta", 0.0)),
        )
        rebound_values = expert_utility.unsqueeze(-1) * (new_coverage + 1.0e-8)
        output_saliency_provenance = {
            "aggregation": "per_fold_layer_mean_normalize_then_arithmetic_mean",
            "fold_count": len(output_saliency_folds),
            "folds": output_saliency_fold_provenance,
            "beta": float(getattr(args, "output_saliency_beta", 0.0)),
            "formula": expected_formula,
            "split": "train",
        }

    frontier_regret_min_widths = None
    frontier_committee_regret_floor = None
    frontier_committee_regret_score = None
    frontier_regret_provenance = None
    frontier_paths = list(getattr(args, "frontier_regret_cache", []))
    if frontier_paths:
        reference_profile_path = getattr(args, "frontier_reference_profile", None)
        if reference_profile_path is None:
            raise ValueError(
                "frontier regret caches require --frontier-reference-profile."
            )
        reference_profile_path = reference_profile_path.expanduser().resolve()
        reference_profile = torch.load(
            reference_profile_path, map_location="cpu", weights_only=True
        )
        if reference_profile.get("test_metrics_used_for_profile") is not False:
            raise ValueError("frontier reference profile must be test-independent.")
        reference_widths = reference_profile.get("profile_widths")
        if not isinstance(reference_widths, torch.Tensor) or reference_widths.shape != old_values.shape[:2]:
            raise ValueError("frontier reference widths do not match profile shape.")
        regret_folds = []
        fold_provenance = []
        expected_regret_formula = None
        expected_regret_approximation = None
        for raw_path in frontier_paths:
            path = raw_path.expanduser().resolve()
            cache = torch.load(path, map_location="cpu", weights_only=True)
            if cache.get("split") != "train":
                raise ValueError("frontier regret cache must use the train split.")
            if int(cache.get("sequence_length", -1)) != sequence_length:
                raise ValueError("frontier regret cache sequence length does not match.")
            if cache.get("test_metrics_used") is not False:
                raise ValueError("frontier regret cache must be test-independent.")
            table = cache.get("expert_block_committee_residual_mean")
            if not isinstance(table, dict) or not table:
                raise ValueError(
                    "frontier regret cache is missing block residual means."
                )
            fold = torch.stack(
                [_lookup(table, layer_id).float() for layer_id in layer_ids]
            )
            if fold.shape != old_values.shape:
                raise ValueError("frontier regret cache shape does not match blocks.")
            formula = cache.get("block_committee_regret_formula")
            approximation = cache.get("block_committee_regret_approximation")
            if expected_regret_formula is None:
                expected_regret_formula = formula
                expected_regret_approximation = approximation
            elif formula != expected_regret_formula or approximation != expected_regret_approximation:
                raise ValueError("frontier regret cache formulas do not match.")
            regret_folds.append(fold)
            fold_provenance.append(
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "calibration_token_offset": cache.get(
                        "calibration_token_offset", 0
                    ),
                    "calibration_token_end": cache.get("calibration_token_end"),
                }
            )
        (
            frontier_regret_min_widths,
            frontier_committee_regret_floor,
            frontier_committee_regret_score,
        ) = build_frontier_regret_floors(
            torch.stack(regret_folds),
            reference_widths,
            global_quantile=float(
                getattr(args, "frontier_regret_floor_quantile", 0.995)
            ),
            width_increment=int(
                getattr(args, "frontier_regret_width_increment", 1)
            ),
            aggregation=str(
                getattr(args, "frontier_regret_fold_aggregation", "minimum")
            ),
        )
        frontier_regret_provenance = {
            "reference_profile": {
                "path": str(reference_profile_path),
                "sha256": file_sha256(reference_profile_path),
                "profile_sha256": reference_profile.get("profile_sha256"),
                "total_blocks": reference_profile.get("total_blocks"),
            },
            "folds": fold_provenance,
            "formula": expected_regret_formula,
            "approximation": expected_regret_approximation,
            "split": "train",
        }

    maximum_blocks = int(rebound_values.numel())
    layer_maximum_blocks = int(rebound_values.shape[1] * rebound_values.shape[2])
    total_blocks_by_layer = None
    if args.allocation_scope == "per_layer":
        retained_per_layer = args.retained_blocks_per_layer
        if retained_per_layer is None:
            retained_per_layer = int(round(layer_maximum_blocks * (1.0 - target)))
        if not 0 <= int(retained_per_layer) <= layer_maximum_blocks:
            raise ValueError(
                f"retained-blocks-per-layer must be in [0, {layer_maximum_blocks}]."
            )
        total_blocks_by_layer = torch.full(
            (int(rebound_values.shape[0]),),
            int(retained_per_layer),
            dtype=torch.long,
        )
        total_blocks = int(total_blocks_by_layer.sum().item())
    else:
        if args.retained_blocks_per_layer is not None:
            raise ValueError(
                "retained-blocks-per-layer requires allocation-scope=per_layer."
            )
        total_blocks = int(round(maximum_blocks * (1.0 - target)))
    risk_floor = None
    min_widths = None
    output_saliency_risk_floor = None
    unique_contribution_risk_floor = None
    consensus_provenance = []
    if int(args.risk_floor_min_width) > 0:
        if args.risk_floor_cache is not None and args.risk_floor_consensus_cache:
            raise ValueError(
                "risk-floor-cache and risk-floor-consensus-cache are mutually exclusive."
            )
        if args.risk_floor_consensus_cache:
            risk_folds = []
            for cache_path in args.risk_floor_consensus_cache:
                cache = torch.load(cache_path, map_location="cpu", weights_only=True)
                if cache.get("split") != "train":
                    raise ValueError("risk consensus caches must use the train split.")
                risk_table = cache.get("expert_tail_risk_proxy")
                if not isinstance(risk_table, dict):
                    raise ValueError(
                        "risk consensus caches must contain expert_tail_risk_proxy."
                    )
                risk_folds.append(
                    torch.stack(
                        [_lookup(risk_table, layer_id).float() for layer_id in layer_ids]
                    )
                )
                consensus_provenance.append(
                    {
                        "path": str(cache_path.resolve()),
                        "sha256": file_sha256(cache_path),
                        "calibration_token_offset": cache.get(
                            "calibration_token_offset", 0
                        ),
                        "calibration_token_end": cache.get("calibration_token_end"),
                    }
                )
            votes = args.risk_floor_min_votes
            if votes is None:
                votes = len(risk_folds) // 2 + 1
            min_widths, risk_floor = build_consensus_rare_event_risk_floors(
                torch.stack(risk_folds),
                early_layer_count=int(args.risk_floor_early_layers),
                global_quantile=float(args.risk_floor_quantile),
                relative_to_global_max=float(args.risk_floor_relative_max),
                minimum_width=int(args.risk_floor_min_width),
                num_blocks=int(rebound_values.shape[2]),
                minimum_votes=int(votes),
            )
        else:
            risk_table = risk_source.get("expert_tail_risk_proxy")
            if not isinstance(risk_table, dict):
                raise ValueError("risk floor cache must contain expert_tail_risk_proxy.")
            expert_risk = torch.stack(
                [_lookup(risk_table, layer_id).float() for layer_id in layer_ids]
            )
            min_widths, risk_floor = build_rare_event_risk_floors(
                expert_risk,
                early_layer_count=int(args.risk_floor_early_layers),
                global_quantile=float(args.risk_floor_quantile),
                relative_to_global_max=float(args.risk_floor_relative_max),
                minimum_width=int(args.risk_floor_min_width),
                num_blocks=int(rebound_values.shape[2]),
            )
    output_floor_width = int(
        getattr(args, "output_saliency_floor_min_width", 0)
    )
    if output_floor_width > 0:
        if output_saliency_consensus is None:
            raise ValueError(
                "output saliency safety floors require output-saliency-cache."
            )
        existing_min_widths = (
            torch.zeros_like(output_saliency_consensus, dtype=torch.long)
            if min_widths is None
            else min_widths.clone()
        )
        output_min_widths, output_saliency_risk_floor = (
            build_rare_event_risk_floors(
                output_saliency_consensus,
                early_layer_count=int(output_saliency_consensus.shape[0]),
                global_quantile=float(
                    getattr(args, "output_saliency_floor_quantile", 0.995)
                ),
                relative_to_global_max=float(
                    getattr(args, "output_saliency_floor_relative_max", 0.10)
                ),
                minimum_width=output_floor_width,
                num_blocks=int(rebound_values.shape[2]),
            )
        )
        newly_constrained = output_min_widths > existing_min_widths
        output_saliency_risk_floor["newly_constrained_count"] = int(
            newly_constrained.sum().item()
        )
        output_saliency_risk_floor["newly_constrained_experts"] = [
            {
                "layer": int(layer),
                "expert": int(expert),
                "saliency": float(output_saliency_consensus[layer, expert]),
                "previous_min_width": int(existing_min_widths[layer, expert]),
                "min_width": int(output_min_widths[layer, expert]),
            }
            for layer, expert in torch.nonzero(newly_constrained).tolist()
        ]
        min_widths = torch.maximum(existing_min_widths, output_min_widths)
    unique_floor_width = int(
        getattr(args, "unique_contribution_floor_min_width", 0)
    )
    if unique_floor_width > 0:
        if unique_contribution_score is None:
            raise ValueError(
                "unique contribution safety floors require output saliency and co-route caches."
            )
        existing_min_widths = (
            torch.zeros_like(unique_contribution_score, dtype=torch.long)
            if min_widths is None
            else min_widths.clone()
        )
        unique_min_widths, unique_contribution_risk_floor = (
            build_rare_event_risk_floors(
                unique_contribution_score,
                early_layer_count=int(unique_contribution_score.shape[0]),
                global_quantile=float(
                    getattr(args, "unique_contribution_floor_quantile", 0.995)
                ),
                relative_to_global_max=float(
                    getattr(args, "unique_contribution_floor_relative_max", 0.0)
                ),
                minimum_width=unique_floor_width,
                num_blocks=int(rebound_values.shape[2]),
            )
        )
        newly_constrained = unique_min_widths > existing_min_widths
        unique_contribution_risk_floor["newly_constrained_count"] = int(
            newly_constrained.sum().item()
        )
        unique_contribution_risk_floor["newly_constrained_experts"] = [
            {
                "layer": int(layer),
                "expert": int(expert),
                "unique_contribution": float(
                    unique_contribution_score[layer, expert]
                ),
                "previous_min_width": int(existing_min_widths[layer, expert]),
                "min_width": int(unique_min_widths[layer, expert]),
            }
            for layer, expert in torch.nonzero(newly_constrained).tolist()
        ]
        min_widths = torch.maximum(existing_min_widths, unique_min_widths)
    if frontier_regret_min_widths is not None:
        existing_min_widths = (
            torch.zeros_like(frontier_regret_min_widths, dtype=torch.long)
            if min_widths is None
            else min_widths.clone()
        )
        newly_constrained = frontier_regret_min_widths > existing_min_widths
        frontier_committee_regret_floor["newly_constrained_count"] = int(
            newly_constrained.sum().item()
        )
        frontier_committee_regret_floor["newly_constrained_experts"] = [
            {
                "layer": int(layer),
                "expert": int(expert),
                "score": float(frontier_committee_regret_score[layer, expert]),
                "previous_min_width": int(existing_min_widths[layer, expert]),
                "min_width": int(frontier_regret_min_widths[layer, expert]),
            }
            for layer, expert in torch.nonzero(newly_constrained).tolist()
        ]
        min_widths = torch.maximum(
            existing_min_widths, frontier_regret_min_widths
        )
    if total_blocks_by_layer is None:
        widths = allocate_static_prefix_widths(
            rebound_values,
            total_blocks=total_blocks,
            min_widths=min_widths,
        )
    else:
        widths = allocate_static_prefix_widths_per_layer(
            rebound_values,
            total_blocks_by_layer=total_blocks_by_layer,
            min_widths=min_widths,
        )
    profile_digest = hashlib.sha256(widths.numpy().tobytes()).hexdigest()
    tail_lambda = float(tail.get("tail_lambda", 0.0))
    tail_label = f"{tail_lambda:.2f}".replace(".", "p")
    mode = f"conditional_dual_tail_{tail_label}"
    if (
        output_saliency_provenance is not None
        and float(args.output_saliency_beta) > 0.0
    ):
        beta_label = f"{float(args.output_saliency_beta):.2f}".replace(".", "p")
        mode = f"{mode}_output_saliency_b{beta_label}"
    if output_saliency_risk_floor is not None:
        mode = f"{mode}_output_floor_w{output_floor_width}"
    if unique_contribution_risk_floor is not None:
        mode = f"{mode}_unique_floor_w{unique_floor_width}"
    if frontier_committee_regret_floor is not None:
        quantile_label = f"{float(args.frontier_regret_floor_quantile):.3f}".replace(
            ".", "p"
        )
        mode = f"{mode}_frontier_regret_q{quantile_label}"
    if risk_floor is not None:
        mode = f"{mode}_risk_floor_w{int(args.risk_floor_min_width)}"
        if consensus_provenance:
            mode = (
                f"{mode}_consensus{risk_floor['minimum_votes']}of"
                f"{risk_floor['fold_count']}"
            )
    payload = {
        "schema_version": 1,
        "method": f"static_expert_{mode}",
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": teacher.get("model_path"),
        "dataset": teacher.get("dataset"),
        "calibration_split": "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": int(widths.shape[0]),
        "num_experts": int(widths.shape[1]),
        "num_blocks": int(rebound_values.shape[2]),
        "channel_block_size": int(tail.get("block_size", 0)),
        "target_pruning_ratio": target,
        "allocation_scope": args.allocation_scope,
        "target_blocks_by_layer": (
            None if total_blocks_by_layer is None else total_blocks_by_layer.tolist()
        ),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "actual_structural_pruning_ratio": 1.0
        - int(widths.sum().item()) / maximum_blocks,
        "total_blocks": int(widths.sum().item()),
        "maximum_blocks": maximum_blocks,
        "min_blocks_per_expert": 0,
        "profile_widths": widths.cpu(),
        "profile_sha256": profile_digest,
        "tail_lambda": tail_lambda,
        "tail_score_mode": tail.get("score_mode"),
        "risk_floor": risk_floor,
        "risk_consensus_provenance": consensus_provenance,
        "coverage_consensus_provenance": coverage_consensus_provenance,
        "teacher_consensus_provenance": teacher_consensus_provenance,
        "teacher_consensus_std_penalty": float(
            getattr(args, "teacher_consensus_std_penalty", 0.0)
        ),
        "output_saliency_provenance": output_saliency_provenance,
        "output_saliency_factor": None
        if output_saliency_factor is None
        else output_saliency_factor.cpu(),
        "output_saliency_risk_floor": output_saliency_risk_floor,
        "unique_contribution_risk_floor": unique_contribution_risk_floor,
        "unique_contribution_score": None
        if unique_contribution_score is None
        else unique_contribution_score.cpu(),
        "co_route_uniqueness_folds": None
        if co_route_uniqueness_folds is None
        else co_route_uniqueness_folds.cpu(),
        "co_route_formula": co_route_formula,
        "unique_contribution_fold_aggregation": str(
            getattr(args, "unique_contribution_fold_aggregation", "minimum")
        ),
        "frontier_committee_regret_floor": frontier_committee_regret_floor,
        "frontier_committee_regret_score": frontier_committee_regret_score,
        "frontier_regret_provenance": frontier_regret_provenance,
        "expert_utility": expert_utility.cpu(),
        "teacher_value_key": "unconditional_block_values",
        "cache_provenance": {
            "calibration": {
                "sha256": teacher.get("calibration_cache_file_sha256"),
                "input_ids_sha256": teacher.get("calibration_input_ids_sha256"),
                "protocol_name": teacher.get("calibration_source", {}).get("protocol_name"),
                "split": teacher.get("split"),
                "sequence_length": teacher.get("sequence_length"),
                "calibration_sequences": teacher.get("calibration_sequences"),
                "calibration_tokens": teacher.get("calibration_tokens"),
            },
            "channel": {
                "path": str(args.tail_channel_cache.resolve()),
                "sha256": file_sha256(args.tail_channel_cache),
                "score_mode": tail.get("score_mode"),
                "dataset": tail.get("dataset"),
                "split": tail.get("split"),
                "sequence_length": tail.get("sequence_length"),
                "calibration_sequences": tail.get("calibration_sequences"),
                "calibration_tokens": tail.get("calibration_tokens"),
                "calibration_token_offset": tail.get("calibration_token_offset", 0),
                "calibration_token_end": tail.get("calibration_token_end"),
                "calibration_source": tail.get("calibration_source"),
            },
            "reference_channel": {
                "path": str(args.reference_channel_cache.resolve()),
                "sha256": file_sha256(args.reference_channel_cache),
                "score_mode": reference.get("score_mode"),
            },
            "risk_floor": {
                "path": str(risk_source_path.resolve()),
                "sha256": file_sha256(risk_source_path),
                "dataset": risk_source.get("dataset"),
                "dataset_config": risk_source.get("dataset_config"),
                "split": risk_source.get("split"),
                "sequence_length": risk_source.get("sequence_length"),
                "calibration_sequences": risk_source.get("calibration_sequences"),
                "calibration_tokens": risk_source.get("calibration_tokens"),
                "calibration_token_offset": risk_source.get(
                    "calibration_token_offset", 0
                ),
                "calibration_token_end": risk_source.get("calibration_token_end"),
                "calibration_source": risk_source.get("calibration_source"),
            },
            "conditional_dual_teacher": {
                "path": str(args.teacher_cache.resolve()),
                "sha256": file_sha256(args.teacher_cache),
                "parent_mode": teacher.get("parent_mode"),
                "dataset": teacher.get("dataset"),
                "dataset_config": teacher.get("dataset_config"),
                "split": teacher.get("split"),
                "calibration_tokens": teacher.get("calibration_tokens"),
                "calibration_token_offset": teacher.get(
                    "calibration_token_offset", 0
                ),
                "calibration_token_end": teacher.get("calibration_token_end"),
                "calibration_source": teacher.get("calibration_source"),
            },
        },
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_profile)
    unique, counts = torch.unique(widths, return_counts=True)
    summary = {
        key: value
        for key, value in payload.items()
        if key not in {
            "profile_widths",
            "expert_utility",
            "output_saliency_factor",
            "unique_contribution_score",
            "co_route_uniqueness_folds",
            "frontier_committee_regret_score",
        }
    }
    summary["width_histogram"] = {
        str(int(width)): int(count)
        for width, count in zip(unique.tolist(), counts.tolist())
    }
    summary["output_profile"] = str(args.output_profile.resolve())
    summary_path = args.output_profile.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output_profile.resolve())
    print(summary_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
