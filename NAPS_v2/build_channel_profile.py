from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_naps_v2_artifacts import file_sha256


def _layer_table(rankings: dict[str, Any], layer_id: int) -> dict[str, Any]:
    table = rankings["table"].get(layer_id, rankings["table"].get(str(layer_id)))
    if table is None:
        raise KeyError(f"Missing CHANNEL ranking table for layer {layer_id}")
    return table


def validate_nested_rankings(rankings: dict[str, Any]) -> tuple[list[int], int, int, tuple[int, ...], int]:
    if int(rankings.get("schema_version", -1)) != 4:
        raise ValueError("CHANNEL rankings must use schema version 4")
    if not rankings.get("ranking_is_nested", False):
        raise ValueError("CHANNEL rankings must declare nested channel orders")
    layer_ids = sorted(int(layer_id) for layer_id in rankings["table"])
    if layer_ids != list(range(len(layer_ids))):
        raise ValueError("CHANNEL layer IDs must be contiguous and zero-based")
    source_width = int(rankings["source_intermediate_size"])
    block_size = int(rankings["channel_alignment"])
    width_options = tuple(int(width) for width in rankings["width_options"])
    if not width_options or any(width <= 0 or width > source_width for width in width_options):
        raise ValueError("CHANNEL width options must fit inside the source width")
    if source_width % block_size or any(width % block_size for width in width_options):
        raise ValueError("CHANNEL source and candidate widths must be block-aligned")

    expected_channels = torch.arange(source_width)
    num_experts = -1
    for layer_id in layer_ids:
        table = _layer_table(rankings, layer_id)
        layer_widths = tuple(int(width) for width in table["width_options"].tolist())
        if layer_widths != width_options:
            raise ValueError(f"Layer {layer_id} width options do not match the artifact")
        orders = table["ranked_indices_by_width"].to(torch.long)
        if orders.ndim != 3 or orders.shape[1:] != (len(width_options), source_width):
            raise ValueError(f"Layer {layer_id} CHANNEL ranking shape is invalid")
        if num_experts < 0:
            num_experts = int(orders.shape[0])
        elif int(orders.shape[0]) != num_experts:
            raise ValueError("CHANNEL expert count differs across layers")
        sorted_orders = torch.sort(orders, dim=-1).values
        expected = expected_channels.expand_as(sorted_orders)
        if not torch.equal(sorted_orders.cpu(), expected):
            raise ValueError(f"Layer {layer_id} contains a non-permutation channel order")
        if not torch.equal(orders, orders[:, :1].expand_as(orders)):
            raise ValueError(f"Layer {layer_id} candidate widths do not share one nested order")
        if "ranked_indices" in table and not torch.equal(
            table["ranked_indices"].to(torch.long), orders[:, 0]
        ):
            raise ValueError(f"Layer {layer_id} canonical ranking differs from width rankings")
    return layer_ids, num_experts, source_width, width_options, block_size


def build_uniform_profile(
    rankings: dict[str, Any],
    uniform_width: int,
    padded_width: int | None = None,
) -> dict[str, Any]:
    layer_ids, num_experts, source_width, width_options, block_size = validate_nested_rankings(rankings)
    uniform_width = int(uniform_width)
    if uniform_width not in width_options:
        raise ValueError(f"Uniform width {uniform_width} is not present in CHANNEL width options {width_options}")
    padded_width = uniform_width if padded_width is None else int(padded_width)
    if padded_width < uniform_width or padded_width > source_width or padded_width % block_size:
        raise ValueError("Padded width must be aligned and between the uniform and source widths")
    widths = torch.full(
        (len(layer_ids), num_experts),
        uniform_width // block_size,
        dtype=torch.long,
    )
    return {
        "schema_version": 4,
        "method": "channel_calibrated_nested_mask",
        "mode": "uniform_padded_homogeneous",
        "model_path": rankings["model_path"],
        "model_family": rankings["model_family"],
        "profile_construction": "label_free_real_token_calibration",
        "test_metrics_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": len(layer_ids),
        "num_experts": num_experts,
        "source_intermediate_size": source_width,
        "intermediate_size": padded_width,
        "padded_intermediate_size": padded_width,
        "num_blocks": padded_width // block_size,
        "channel_block_size": block_size,
        "width_options": torch.tensor(width_options, dtype=torch.long),
        "profile_widths": widths,
        "allocation_scope": "per_expert_uniform_channel",
        "uniform_width": uniform_width,
        "total_blocks": int(widths.sum().item()),
        "maximum_blocks": int(widths.numel() * padded_width // block_size),
        "padding_is_structural_zero": True,
        "ranking_is_nested": True,
        "capture_path": rankings.get("capture_path"),
        "capture_sha256": rankings.get("capture_sha256"),
        "calibration": rankings.get("calibration"),
        "model_provenance": rankings.get("model_provenance"),
    }


def _balanced_width_assignment(
    shrink_costs: torch.Tensor,
    expand_gains: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    if shrink_costs.ndim != 1 or expand_gains.shape != shrink_costs.shape:
        raise ValueError("Shrink costs and expand gains must be aligned one-dimensional tensors")
    if eligible.shape != shrink_costs.shape or eligible.dtype != torch.bool:
        raise ValueError("Eligibility mask must be boolean and expert-aligned")
    if not bool(torch.isfinite(shrink_costs[eligible]).all()) or not bool(
        torch.isfinite(expand_gains[eligible]).all()
    ):
        raise ValueError("Eligible width-transfer utilities must be finite")
    if bool((shrink_costs[eligible] < 0).any()) or bool((expand_gains[eligible] < 0).any()):
        raise ValueError("Width-transfer utilities must be non-negative")

    # State keys are large-minus-small counts. Medium is considered first and
    # wins exact ties, which keeps the no-transfer solution deterministic.
    states: dict[int, tuple[float, int, tuple[int, ...]]] = {0: (0.0, 0, ())}
    for expert_id in range(shrink_costs.numel()):
        choices = [(0, 0.0, 0)]
        if bool(eligible[expert_id]):
            choices.extend((
                (-1, -float(shrink_costs[expert_id]), -1),
                (1, float(expand_gains[expert_id]), 1),
            ))
        next_states: dict[int, tuple[float, int, tuple[int, ...]]] = {}
        for balance, (objective, changed, assignments) in states.items():
            for delta, value, assignment in choices:
                candidate = (
                    objective + value,
                    changed + int(assignment != 0),
                    assignments + (assignment,),
                )
                next_balance = balance + delta
                current = next_states.get(next_balance)
                if current is None or candidate[0] > current[0] + 1.0e-12 or (
                    abs(candidate[0] - current[0]) <= 1.0e-12
                    and (candidate[1], candidate[2]) < (current[1], current[2])
                ):
                    next_states[next_balance] = candidate
        states = next_states
    objective, _, assignments = states[0]
    return torch.tensor(assignments, dtype=torch.long), float(objective)


def build_budgeted_profile(
    rankings: dict[str, Any],
    small_width: int,
    medium_width: int,
    large_width: int,
    padded_width: int | None = None,
    min_coverage_confidence: float = 1.0,
) -> dict[str, Any]:
    layer_ids, num_experts, source_width, width_options, block_size = validate_nested_rankings(rankings)
    small_width, medium_width, large_width = map(int, (small_width, medium_width, large_width))
    if (small_width, medium_width, large_width) != tuple(
        sorted((small_width, medium_width, large_width))
    ) or not small_width < medium_width < large_width:
        raise ValueError("Budgeted widths must be strictly increasing")
    if medium_width - small_width != large_width - medium_width:
        raise ValueError("Budgeted widths must be symmetric around the medium width")
    if any(width not in width_options for width in (small_width, medium_width, large_width)):
        raise ValueError("All budgeted widths must be present in the CHANNEL ranking artifact")
    if not 0.0 <= min_coverage_confidence <= 1.0:
        raise ValueError("Minimum coverage confidence must be between zero and one")
    padded_width = large_width if padded_width is None else int(padded_width)
    if padded_width < large_width or padded_width > source_width or padded_width % block_size:
        raise ValueError("Padded width must be aligned and between the large and source widths")

    small_slot = width_options.index(small_width)
    medium_slot = width_options.index(medium_width)
    large_slot = width_options.index(large_width)
    widths_by_layer = []
    allocation_diagnostics = []
    for layer_id in layer_ids:
        table = _layer_table(rankings, layer_id)
        order = table["ranked_indices"].to(torch.long)
        response_energy = table["route_weighted_response_energy"].float()
        down_energy = table["down_channel_energy"].float()
        if response_energy.shape != order.shape or down_energy.shape != order.shape:
            raise ValueError(f"Layer {layer_id} utility components are not expert-channel aligned")
        ordered_utility = torch.gather(response_energy * down_energy, 1, order)
        shrink_costs = ordered_utility[:, small_width:medium_width].sum(1)
        expand_gains = ordered_utility[:, medium_width:large_width].sum(1)
        coverage = table["coverage_confidence"].float()
        score_sources = table["score_sources"]
        if coverage.shape != (num_experts,) or len(score_sources) != num_experts:
            raise ValueError(f"Layer {layer_id} coverage metadata is not expert-aligned")
        eligible = coverage >= float(min_coverage_confidence)
        eligible &= torch.tensor(
            [source == "real_token_route_weighted" for source in score_sources],
            dtype=torch.bool,
        )
        assignments, objective_gain = _balanced_width_assignment(
            shrink_costs,
            expand_gains,
            eligible,
        )
        widths = torch.full((num_experts,), medium_width, dtype=torch.long)
        widths[assignments < 0] = small_width
        widths[assignments > 0] = large_width
        small_count = int((assignments < 0).sum().item())
        large_count = int((assignments > 0).sum().item())
        if small_count != large_count or int(widths.sum().item()) != num_experts * medium_width:
            raise RuntimeError(f"Layer {layer_id} CHANNEL allocation violated the exact budget")
        widths_by_layer.append(widths // block_size)
        allocation_diagnostics.append({
            "layer_id": layer_id,
            "eligible_experts": int(eligible.sum().item()),
            "small_experts": small_count,
            "medium_experts": int((assignments == 0).sum().item()),
            "large_experts": large_count,
            "fit_objective_gain": objective_gain,
            "small_expert_ids": torch.where(assignments < 0)[0].tolist(),
            "large_expert_ids": torch.where(assignments > 0)[0].tolist(),
            "width_slots": {
                "small": small_slot,
                "medium": medium_slot,
                "large": large_slot,
            },
        })
    profile_widths = torch.stack(widths_by_layer)
    return {
        "schema_version": 4,
        "method": "channel_calibrated_nested_mask",
        "mode": "fit_utility_budgeted_heterogeneous",
        "model_path": rankings["model_path"],
        "model_family": rankings["model_family"],
        "profile_construction": "label_free_fit_route_weighted_marginal_utility",
        "test_metrics_used_for_profile": False,
        "holdout_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": len(layer_ids),
        "num_experts": num_experts,
        "source_intermediate_size": source_width,
        "intermediate_size": padded_width,
        "padded_intermediate_size": padded_width,
        "num_blocks": padded_width // block_size,
        "channel_block_size": block_size,
        "width_options": torch.tensor(width_options, dtype=torch.long),
        "profile_widths": profile_widths,
        "allocation_scope": "per_layer_balanced_expert_width",
        "small_width": small_width,
        "medium_width": medium_width,
        "large_width": large_width,
        "budget_reference_width": medium_width,
        "min_coverage_confidence": float(min_coverage_confidence),
        "total_blocks": int(profile_widths.sum().item()),
        "maximum_blocks": int(profile_widths.numel() * padded_width // block_size),
        "padding_is_structural_zero": True,
        "ranking_is_nested": True,
        "allocation_diagnostics": allocation_diagnostics,
        "capture_path": rankings.get("capture_path"),
        "capture_sha256": rankings.get("capture_sha256"),
        "calibration": rankings.get("calibration"),
        "model_provenance": rankings.get("model_provenance"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an exporter-compatible CHANNEL profile.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--uniform-width", type=int)
    mode.add_argument("--budgeted-widths", type=int, nargs=3, metavar=("SMALL", "MEDIUM", "LARGE"))
    parser.add_argument("--padded-width", type=int)
    parser.add_argument("--min-coverage-confidence", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    rankings_path = artifact_dir / "rankings.pt"
    profile_path = artifact_dir / "profile.pt"
    summary_path = artifact_dir / "profile_summary.json"
    if profile_path.exists() and not args.force:
        raise FileExistsError(f"CHANNEL profile already exists: {profile_path}")
    rankings = torch.load(rankings_path, map_location="cpu", weights_only=True)
    if args.uniform_width is not None:
        profile = build_uniform_profile(rankings, args.uniform_width, args.padded_width)
    else:
        profile = build_budgeted_profile(
            rankings,
            *args.budgeted_widths,
            padded_width=args.padded_width,
            min_coverage_confidence=args.min_coverage_confidence,
        )
    profile["cache_provenance"] = {"channel_sha256": file_sha256(rankings_path)}
    torch.save(profile, profile_path)
    summary = {
        key: value
        for key, value in profile.items()
        if key not in {"profile_widths", "width_options"}
    }
    summary["width_options"] = profile["width_options"].tolist()
    summary["profile_shape"] = list(profile["profile_widths"].shape)
    summary["assigned_widths"] = sorted(
        set((profile["profile_widths"] * profile["channel_block_size"]).flatten().tolist())
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())