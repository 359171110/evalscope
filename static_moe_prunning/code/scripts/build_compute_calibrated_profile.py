from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.static_expert_pruning import (
    aggregate_route_count_folds,
    allocate_compute_calibrated_prefix_widths,
    allocate_fold_constrained_prefix_widths,
    allocate_route_envelope_constrained_prefix_widths,
    build_layer_routing_entropy_prior,
    build_static_block_values,
    validate_static_profile_payload,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an exact-structure static expert profile at a train-only "
            "expected routed-compute anchor."
        )
    )
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--target-routed-pruning-ratio", type=float, required=True)
    parser.add_argument(
        "--compute-route-cache",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional train-only route-count cache used only for the compute "
            "constraint; repeat to average normalized route distributions."
        ),
    )
    parser.add_argument("--search-iterations", type=int, default=64)
    parser.add_argument(
        "--compute-noninferiority-reference-profile",
        type=Path,
        default=None,
        help=(
            "Optional frozen train-only reference profile. When provided, "
            "every compute-route fold becomes a separate retained-cost upper "
            "bound instead of being collapsed into one scalar anchor."
        ),
    )
    parser.add_argument("--fold-dual-iterations", type=int, default=512)
    parser.add_argument("--fold-dual-step-size", type=float, default=2.0)
    parser.add_argument("--fold-relative-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--route-envelope-expansion",
        type=float,
        default=None,
        help=(
            "Optional reference-centered coordinate-envelope constraint. "
            "The lower/upper train-fold route envelope is padded by this "
            "multiple of its observed range; zero enables the unpadded envelope."
        ),
    )
    parser.add_argument(
        "--route-aggregation",
        choices=("mean", "worst_case", "cvar"),
        default="mean",
        help="Train-fold route-cost aggregation used by the compute constraint.",
    )
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=0.75,
        help="Upper-tail confidence level for route-aggregation=cvar.",
    )
    parser.add_argument(
        "--layer-entropy-gamma",
        type=float,
        default=0.0,
        help="Train routing-entropy placement prior; positive protects dispersed layers.",
    )
    return parser.parse_args()


def _risk_floors(source: dict, shape: tuple[int, int]) -> torch.Tensor | None:
    floors = torch.zeros(shape, dtype=torch.long)
    found = False
    for key in (
        "risk_floor",
        "output_saliency_risk_floor",
        "unique_contribution_risk_floor",
        "frontier_committee_regret_floor",
    ):
        risk = source.get(key)
        if not isinstance(risk, dict):
            continue
        selected = risk.get("selected_experts")
        if not isinstance(selected, list):
            continue
        for item in selected:
            layer = int(item["layer"])
            expert = int(item["expert"])
            width = int(item["min_width"])
            if not 0 <= layer < shape[0] or not 0 <= expert < shape[1]:
                raise ValueError(f"{key} expert index is out of bounds.")
            floors[layer, expert] = max(int(floors[layer, expert]), width)
            found = True
    return floors if found else None


def _source_values(
    source: dict,
    coverage: torch.Tensor,
    route_counts: torch.Tensor,
) -> torch.Tensor:
    mode = str(source.get("mode"))
    if mode == "route_rms":
        return build_static_block_values(
            coverage,
            route_counts=route_counts,
            mode="route_rms",
        )
    if mode.startswith("conditional_dual_tail_"):
        utility = source.get("expert_utility")
        if not isinstance(utility, torch.Tensor) or utility.shape != coverage.shape[:2]:
            raise ValueError("Tail-Risk source profile must contain expert_utility.")
        return utility.float().unsqueeze(-1) * (coverage.float() + 1.0e-8)
    raise ValueError(f"unsupported source profile mode for compute calibration: {mode}")


def _route_counts_from_cache(path: Path, layer_ids: list[int]) -> tuple[torch.Tensor, dict]:
    resolved = path.expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if payload.get("split") != "train":
        raise ValueError("compute route-count caches must use the train split.")
    route_table = payload.get("route_counts")
    if not isinstance(route_table, dict):
        raise ValueError("compute route-count cache must contain route_counts.")
    counts = torch.stack(
        [_lookup(route_table, layer_id).float() for layer_id in layer_ids]
    )
    if float(counts.sum().item()) <= 0.0:
        raise ValueError("compute route-count cache must contain positive routed mass.")
    return counts, {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "dataset": payload.get("dataset"),
        "split": payload.get("split"),
        "calibration_token_offset": payload.get("calibration_token_offset", 0),
        "calibration_token_end": payload.get("calibration_token_end"),
    }


def main() -> int:
    args = parse_args()
    source_path = args.source_profile.expanduser().resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    source_widths = validate_static_profile_payload(source)
    channel_info = source.get("cache_provenance", {}).get("channel", {})
    channel_path = Path(str(channel_info.get("path", ""))).expanduser().resolve()
    if not channel_path.is_file():
        raise FileNotFoundError("source profile channel cache does not exist.")
    expected_channel_sha = channel_info.get("sha256")
    actual_channel_sha = file_sha256(channel_path)
    if expected_channel_sha != actual_channel_sha:
        raise ValueError("source profile channel cache SHA256 mismatch.")
    channel = torch.load(channel_path, map_location="cpu", weights_only=True)
    if channel.get("split") != "train":
        raise ValueError("compute calibration requires a train-only channel cache.")
    layer_ids = [int(layer_id) for layer_id in source["layer_ids"]]
    coverage = torch.stack(
        [
            _lookup(channel["table"], layer_id)["block_coverage_scores"].float()
            for layer_id in layer_ids
        ]
    )
    source_route_counts = torch.stack(
        [_lookup(channel["route_counts"], layer_id).float() for layer_id in layer_ids]
    )
    if coverage.shape[:2] != source_widths.shape or source_route_counts.shape != source_widths.shape:
        raise ValueError("source profile and channel cache shapes do not match.")
    values = _source_values(source, coverage, source_route_counts)
    compute_paths = list(args.compute_route_cache) or [channel_path]
    compute_folds = []
    compute_provenance = []
    for compute_path in compute_paths:
        counts, provenance = _route_counts_from_cache(compute_path, layer_ids)
        if counts.shape != source_widths.shape:
            raise ValueError("compute route-count cache shape does not match profile.")
        compute_folds.append(counts.double())
        compute_provenance.append(provenance)
    fold_tensor = torch.stack(compute_folds)
    aggregate_distribution, aggregation_audit = aggregate_route_count_folds(
        fold_tensor,
        aggregation=str(getattr(args, "route_aggregation", "mean")),
        cvar_alpha=float(getattr(args, "cvar_alpha", 0.75)),
    )
    compute_route_counts = aggregate_distribution * compute_folds[0].sum()
    entropy_prior, entropy_audit = build_layer_routing_entropy_prior(
        aggregate_distribution,
        gamma=float(getattr(args, "layer_entropy_gamma", 0.0)),
    )
    values = values * entropy_prior.to(values.device, values.dtype).view(-1, 1, 1)
    floors = _risk_floors(source, tuple(source_widths.shape))
    noninferiority_path = getattr(
        args, "compute_noninferiority_reference_profile", None
    )
    noninferiority_provenance = None
    if noninferiority_path is None:
        widths, compute_audit = allocate_compute_calibrated_prefix_widths(
            values,
            compute_route_counts,
            total_blocks=int(source["total_blocks"]),
            target_routed_pruning_ratio=float(args.target_routed_pruning_ratio),
            min_blocks_per_expert=int(source.get("min_blocks_per_expert", 0)),
            min_widths=floors,
            search_iterations=int(args.search_iterations),
        )
        method_suffix = "compute_calibrated"
    else:
        reference_path = Path(noninferiority_path).expanduser().resolve()
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        reference_widths = validate_static_profile_payload(reference)
        if reference.get("test_metrics_used_for_profile") is not False:
            raise ValueError("compute reference profile must be test-independent.")
        if reference_widths.shape != source_widths.shape:
            raise ValueError("compute reference profile shape does not match source.")
        if int(reference_widths.sum().item()) != int(source["total_blocks"]):
            raise ValueError(
                "compute reference profile must use the same structural budget."
            )
        envelope_expansion = getattr(args, "route_envelope_expansion", None)
        allocator = allocate_fold_constrained_prefix_widths
        allocator_kwargs = {}
        if envelope_expansion is not None:
            allocator = allocate_route_envelope_constrained_prefix_widths
            allocator_kwargs["envelope_expansion"] = float(envelope_expansion)
        widths, fold_audit = allocator(
            values,
            fold_tensor,
            reference_widths,
            total_blocks=int(source["total_blocks"]),
            min_blocks_per_expert=int(source.get("min_blocks_per_expert", 0)),
            min_widths=floors,
            dual_iterations=int(getattr(args, "fold_dual_iterations", 512)),
            dual_step_size=float(getattr(args, "fold_dual_step_size", 2.0)),
            relative_tolerance=float(
                getattr(args, "fold_relative_tolerance", 1.0e-10)
            ),
            **allocator_kwargs,
        )
        if fold_audit["all_fold_constraints_satisfied"] is not True:
            raise RuntimeError(
                "per-fold compute non-inferiority is infeasible or the dual "
                "search did not find a feasible exact-prefix profile; "
                f"maximum relative violation="
                f"{fold_audit['maximum_relative_fold_violation']}."
            )
        maximum_routed_blocks = float(compute_route_counts.sum().item()) * float(
            coverage.shape[-1]
        )
        reference_retained = float(
            (
                compute_route_counts
                * reference_widths.to(compute_route_counts.dtype)
            ).sum().item()
        )
        achieved_retained = float(
            (compute_route_counts * widths.to(compute_route_counts.dtype)).sum().item()
        )
        target_pruning = 1.0 - reference_retained / maximum_routed_blocks
        achieved_pruning = 1.0 - achieved_retained / maximum_routed_blocks
        compute_audit = {
            "target_routed_pruning_ratio": target_pruning,
            "achieved_train_routed_pruning_ratio": achieved_pruning,
            "absolute_train_compute_error": abs(
                achieved_pruning - target_pruning
            ),
            "maximum_train_routed_blocks": maximum_routed_blocks,
            "target_train_retained_blocks": reference_retained,
            "achieved_train_retained_blocks": achieved_retained,
            "per_fold_noninferiority": fold_audit,
        }
        noninferiority_provenance = {
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
            "profile_sha256": reference.get("profile_sha256"),
            "total_blocks": int(reference_widths.sum().item()),
        }
        method_suffix = (
            "fold_route_envelope_compute_noninferior"
            if envelope_expansion is not None
            else "fold_compute_noninferior"
        )
    profile_digest = hashlib.sha256(widths.numpy().tobytes()).hexdigest()
    payload = {
        **source,
        "method": f"{source['method']}_{method_suffix}",
        "mode": f"{source['mode']}_{method_suffix}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_widths": widths.cpu(),
        "profile_sha256": profile_digest,
        "total_blocks": int(widths.sum().item()),
        "actual_structural_pruning_ratio": 1.0
        - int(widths.sum().item()) / int(source["maximum_blocks"]),
        "compute_calibration": {
            **compute_audit,
            "split": "train",
            "route_count_cache_path": str(channel_path),
            "route_count_cache_sha256": actual_channel_sha,
            "route_distribution_aggregation": aggregation_audit["aggregation"],
            "route_distribution_fold_count": len(compute_folds),
            "route_distribution_aggregation_requested": aggregation_audit["aggregation"],
            "route_distribution_cvar_alpha": aggregation_audit["cvar_alpha"],
            "route_distribution_mass_before_normalization": aggregation_audit[
                "aggregate_mass_before_normalization"
            ],
            "layer_entropy_gamma": entropy_audit["gamma"],
            "layer_entropy_mean": entropy_audit["mean_normalized_entropy"],
            "layer_entropy_min": entropy_audit["minimum_normalized_entropy"],
            "layer_entropy_max": entropy_audit["maximum_normalized_entropy"],
            "route_distribution_provenance": compute_provenance,
            "source_profile_path": str(source_path),
            "source_profile_sha256": file_sha256(source_path),
            "compute_noninferiority_reference_profile": noninferiority_provenance,
            "test_metrics_used": False,
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
    args.output_profile.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
