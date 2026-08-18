from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_channel_artifacts import nested_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute a NAPS-v2-Mask ranking against a CHANNEL teacher artifact."
    )
    parser.add_argument("--mask-artifact", type=Path, required=True)
    parser.add_argument("--mask-diagnostics", type=Path, required=True)
    parser.add_argument("--channel-artifact", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def table_row(table: dict[Any, Any], layer_id: int) -> dict[str, Any]:
    row = table.get(layer_id)
    if row is None:
        row = table.get(str(layer_id))
    if row is None:
        raise KeyError(f"Missing layer {layer_id} in ranking table")
    return row


def rank_percentiles(scores: torch.Tensor, zero_mask: torch.Tensor) -> torch.Tensor:
    order = nested_order(scores, zero_mask)
    percentiles = torch.empty(scores.numel(), dtype=torch.float32)
    percentiles[order.cpu()] = torch.linspace(1.0, 0.0, scores.numel())
    return percentiles


def fusion_name(aimer_weight: float) -> str:
    return f"aimer_{aimer_weight:.2f}_pseudo_output_{1.0 - aimer_weight:.2f}"


def selection_metrics(
    selected: torch.Tensor,
    reference_selected: torch.Tensor,
    reference_scores: torch.Tensor,
) -> dict[str, float]:
    selected_mask = torch.zeros(reference_scores.numel(), dtype=torch.bool)
    teacher_mask = torch.zeros_like(selected_mask)
    selected_mask[selected.to(torch.long)] = True
    teacher_mask[reference_selected.to(torch.long)] = True
    overlap = int((selected_mask & teacher_mask).sum().item())

    utility = reference_scores.float().clone()
    utility[~torch.isfinite(utility)] = 0.0
    utility = utility.clamp_min(0.0)
    denominator = utility[teacher_mask].sum().clamp_min(torch.finfo(torch.float32).eps)
    return {
        "top_width_overlap": overlap / int(reference_selected.numel()),
        "utility_recall": float((utility[selected_mask].sum() / denominator).item()),
    }


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main() -> int:
    args = parse_args()
    mask = torch.load(args.mask_artifact, map_location="cpu", weights_only=False)
    channel = torch.load(args.channel_artifact, map_location="cpu", weights_only=False)
    diagnostics = json.loads(args.mask_diagnostics.read_text(encoding="utf-8"))

    records = {
        (int(record["layer_id"]), int(record["expert_id"])): record
        for record in diagnostics["records"]
    }
    layer_ids = sorted(int(layer_id) for layer_id in channel["table"])
    fusion_weights = tuple(index / 10.0 for index in range(11))
    candidate_rows: dict[str, list[dict[str, float]]] = {
        "aimer_only": [],
        "pseudo_output_energy_only": [],
        "down_energy_only": [],
        "final_mask": [],
    }
    candidate_rows.update({fusion_name(weight): [] for weight in fusion_weights})
    layer_rows = []
    swap_count = 0
    beneficial_overlap_swaps = 0
    harmful_overlap_swaps = 0
    neutral_overlap_swaps = 0
    positive_utility_swaps = 0
    negative_utility_swaps = 0
    utility_delta_sum = 0.0
    normalized_utility_delta_sum = 0.0

    for layer_id in layer_ids:
        mask_layer = table_row(mask["table"], layer_id)
        channel_layer = table_row(channel["table"], layer_id)
        layer_candidate_rows = {name: [] for name in candidate_rows}
        for expert_id in range(int(channel_layer["ranked_indices"].shape[0])):
            teacher_order = channel_layer["ranked_indices"][expert_id].to(torch.long)
            teacher_selected = teacher_order[:args.width]
            teacher_scores = channel_layer["channel_scores"][expert_id]
            zero_mask = channel_layer["effective_zero_masks"][expert_id]
            aimer_scores = channel_layer["stable_aimer_scores"][expert_id]
            structural_scores = channel_layer["structural_channel_scores"][expert_id]
            down_scores = channel_layer["down_channel_energy"][expert_id]
            holdout_scores = (
                channel_layer["holdout_route_weighted_response_energy"][expert_id].float()
                * down_scores.float()
            )
            holdout_order = nested_order(holdout_scores, zero_mask, aimer_scores)
            holdout_selected = holdout_order[:args.width]

            aimer_order = nested_order(aimer_scores, zero_mask)
            structural_order = nested_order(structural_scores, zero_mask, aimer_scores)
            down_order = nested_order(down_scores, zero_mask, aimer_scores)
            aimer_percentiles = rank_percentiles(aimer_scores, zero_mask)
            structural_percentiles = rank_percentiles(structural_scores, zero_mask)
            final_order = mask_layer["ranked_indices"][expert_id].to(torch.long)

            selections = {
                "aimer_only": aimer_order[:args.width],
                "pseudo_output_energy_only": structural_order[:args.width],
                "down_energy_only": down_order[:args.width],
                "final_mask": final_order[:args.width],
            }
            for weight in fusion_weights:
                fusion_scores = weight * aimer_percentiles + (1.0 - weight) * structural_percentiles
                selections[fusion_name(weight)] = nested_order(
                    fusion_scores, zero_mask, aimer_scores
                )[:args.width]
            for name, selected in selections.items():
                metrics = {
                    **prefixed_metrics(
                        "fit", selection_metrics(selected, teacher_selected, teacher_scores)
                    ),
                    **prefixed_metrics(
                        "holdout", selection_metrics(selected, holdout_selected, holdout_scores)
                    ),
                }
                candidate_rows[name].append(metrics)
                layer_candidate_rows[name].append(metrics)

            record = records[(layer_id, expert_id)]["mask"]
            swap_in = torch.tensor(record["swap_in_channels"], dtype=torch.long)
            swap_out = torch.tensor(record["swap_out_channels"], dtype=torch.long)
            teacher_mask = torch.zeros(teacher_scores.numel(), dtype=torch.bool)
            teacher_mask[teacher_selected] = True
            utility = teacher_scores.float().clone()
            utility[~torch.isfinite(utility)] = 0.0
            utility_scale = utility.abs().mean().clamp_min(torch.finfo(torch.float32).eps)
            for incoming, outgoing in zip(swap_in.tolist(), swap_out.tolist()):
                swap_count += 1
                overlap_delta = int(teacher_mask[incoming]) - int(teacher_mask[outgoing])
                beneficial_overlap_swaps += overlap_delta > 0
                harmful_overlap_swaps += overlap_delta < 0
                neutral_overlap_swaps += overlap_delta == 0
                utility_delta = float(utility[incoming].item() - utility[outgoing].item())
                utility_delta_sum += utility_delta
                normalized_utility_delta_sum += utility_delta / float(utility_scale.item())
                positive_utility_swaps += utility_delta > 0.0
                negative_utility_swaps += utility_delta < 0.0

        layer_rows.append({
            "layer_id": layer_id,
            "candidates": {
                name: mean_metrics(rows) for name, rows in layer_candidate_rows.items()
            },
        })

    candidate_summary = {name: mean_metrics(rows) for name, rows in candidate_rows.items()}
    random_overlap = args.width / int(channel["source_intermediate_size"])
    aimer_overlap = candidate_summary["aimer_only"]["fit_top_width_overlap"]
    final_overlap = candidate_summary["final_mask"]["fit_top_width_overlap"]
    aimer_recall = candidate_summary["aimer_only"]["fit_utility_recall"]
    final_recall = candidate_summary["final_mask"]["fit_utility_recall"]
    improved_layers = sum(
        row["candidates"]["final_mask"]["holdout_utility_recall"]
        > row["candidates"]["aimer_only"]["holdout_utility_recall"]
        for row in layer_rows
    )
    degraded_layers = sum(
        row["candidates"]["final_mask"]["holdout_utility_recall"]
        < row["candidates"]["aimer_only"]["holdout_utility_recall"]
        for row in layer_rows
    )
    result = {
        "schema_version": 2,
        "teacher": "CHANNEL real-token ranking; used for attribution only",
        "width": args.width,
        "experts": len(records),
        "random_top_width_overlap": random_overlap,
        "candidate_definitions": {
            "aimer_only": "Stable-AIMER weight-only ranking used by the mask baseline.",
            "pseudo_output_energy_only": (
                "Data-free pseudo-probe response squared times down-projection column energy; "
                "this is stronger than and not identical to the current PP swap score."
            ),
            "final_mask": "Stable-AIMER baseline after the current bounded PP rescue swaps.",
        },
        "candidates": candidate_summary,
        "best_rank_fusion_by_fit_utility": max(
            (fusion_name(weight) for weight in fusion_weights),
            key=lambda name: candidate_summary[name]["fit_utility_recall"],
        ),
        "best_rank_fusion_by_holdout_utility": max(
            (fusion_name(weight) for weight in fusion_weights),
            key=lambda name: candidate_summary[name]["holdout_utility_recall"],
        ),
        "pp_swap_effect": {
            "overlap_delta_vs_aimer": final_overlap - aimer_overlap,
            "teacher_utility_recall_delta_vs_aimer": final_recall - aimer_recall,
            "total_swaps": swap_count,
            "beneficial_overlap_swaps": beneficial_overlap_swaps,
            "harmful_overlap_swaps": harmful_overlap_swaps,
            "neutral_overlap_swaps": neutral_overlap_swaps,
            "positive_teacher_utility_swaps": positive_utility_swaps,
            "negative_teacher_utility_swaps": negative_utility_swaps,
            "positive_teacher_utility_swap_fraction": positive_utility_swaps / max(1, swap_count),
            "negative_teacher_utility_swap_fraction": negative_utility_swaps / max(1, swap_count),
            "mean_teacher_utility_delta_per_swap": utility_delta_sum / max(1, swap_count),
            "mean_normalized_teacher_utility_delta_per_swap": (
                normalized_utility_delta_sum / max(1, swap_count)
            ),
            "layers_improved_by_holdout_utility": improved_layers,
            "layers_degraded_by_holdout_utility": degraded_layers,
        },
        "interpretation_guardrail": (
            "Teacher agreement diagnoses this checkpoint and screens data-free candidates; "
            "it does not replace holdout reconstruction and autoregressive evaluation."
        ),
        "layers": layer_rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())