from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.analyze_mask_channel_attribution import mean_metrics, table_row
from NAPS_v2.build_channel_artifacts import nested_order
from NAPS_v2.build_naps_v2_artifacts import iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen data-free channel metrics against a frozen CHANNEL teacher.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--channel-artifact", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--expert-limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(torch.float32).eps
    result = numerator.float() / denominator.float().clamp_min(epsilon)
    result[~torch.isfinite(result)] = 0.0
    return result


def weight_metrics(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> dict[str, torch.Tensor]:
    gate = gate.float()
    up = up.float()
    down = down.float()
    gate_norm_sq = gate.square().sum(1)
    up_norm_sq = up.square().sum(1)
    down_norm_sq = down.square().sum(0)
    gate_up_inner = (gate * up).sum(1)
    gate_up_product = gate_norm_sq * up_norm_sq
    alignment = safe_ratio(gate_up_inner.abs(), gate_up_product.sqrt())
    return {
        "path_norm": (gate_up_product * down_norm_sq).clamp_min(0.0).pow(1.0 / 6.0),
        "triad_energy": down_norm_sq * (gate_up_product + gate_up_inner.square()),
        "gate_up_product": gate_up_product,
        "gate_up_alignment": alignment,
        "weight_l2_sum": gate_norm_sq + up_norm_sq + down_norm_sq,
    }


def batched_rank_percentiles(scores: torch.Tensor, zero_mask: torch.Tensor) -> torch.Tensor:
    finite = scores.float().clone()
    finite[~torch.isfinite(finite)] = -torch.inf
    finite[:, zero_mask] = -torch.inf
    orders = torch.argsort(finite, dim=1, descending=True, stable=True)
    percentiles = torch.empty_like(finite)
    values = torch.linspace(1.0, 0.0, finite.shape[1], device=finite.device).expand_as(finite)
    percentiles.scatter_(1, orders, values)
    return percentiles


def batched_order(
    scores: torch.Tensor,
    zero_mask: torch.Tensor,
    tie_break_scores: torch.Tensor,
) -> torch.Tensor:
    finite = scores.float().clone()
    finite[~torch.isfinite(finite)] = -torch.inf
    finite[:, zero_mask] = -torch.inf
    tie_break = tie_break_scores.float().clone()
    tie_break[~torch.isfinite(tie_break)] = -torch.inf
    tie_order = torch.argsort(tie_break, descending=True, stable=True)
    tie_order = tie_order.unsqueeze(0).expand(finite.shape[0], -1)
    ordered_scores = finite.gather(1, tie_order)
    score_positions = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)
    return tie_order.gather(1, score_positions)


def batched_candidate_metrics(
    scores: torch.Tensor,
    zero_mask: torch.Tensor,
    tie_break_scores: torch.Tensor,
    width: int,
    fit_selected: torch.Tensor,
    fit_scores: torch.Tensor,
    holdout_selected: torch.Tensor,
    holdout_scores: torch.Tensor,
) -> dict[str, torch.Tensor]:
    selected = batched_order(scores, zero_mask, tie_break_scores)[:, :width]
    selected_masks = torch.zeros_like(scores, dtype=torch.bool)
    selected_masks.scatter_(1, selected, True)

    def reference_metrics(reference_selected: torch.Tensor, reference_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reference_mask = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
        reference_mask[reference_selected] = True
        overlap = (selected_masks & reference_mask).sum(1).float() / width
        utility = reference_scores.double().clone()
        utility[~torch.isfinite(utility)] = 0.0
        utility.clamp_min_(0.0)
        denominator = utility[reference_mask].sum().clamp_min(torch.finfo(torch.float64).eps)
        recall = (selected_masks.double() * utility.unsqueeze(0)).sum(1) / denominator
        return overlap, recall

    candidate_ranks = batched_rank_percentiles(scores, zero_mask)
    fit_ranks = batched_rank_percentiles(fit_scores.unsqueeze(0), zero_mask)[0]
    holdout_ranks = batched_rank_percentiles(holdout_scores.unsqueeze(0), zero_mask)[0]

    def rank_correlation(reference_ranks: torch.Tensor) -> torch.Tensor:
        left = candidate_ranks - candidate_ranks.mean(1, keepdim=True)
        right = reference_ranks - reference_ranks.mean()
        denominator = left.square().sum(1).sqrt() * right.square().sum().sqrt()
        return (left * right.unsqueeze(0)).sum(1) / denominator.clamp_min(torch.finfo(torch.float32).eps)

    fit_overlap, fit_recall = reference_metrics(fit_selected, fit_scores)
    holdout_overlap, holdout_recall = reference_metrics(holdout_selected, holdout_scores)
    return {
        "fit_top_width_overlap": fit_overlap,
        "fit_utility_recall": fit_recall,
        "holdout_top_width_overlap": holdout_overlap,
        "holdout_utility_recall": holdout_recall,
        "fit_spearman": rank_correlation(fit_ranks),
        "holdout_spearman": rank_correlation(holdout_ranks),
    }


def fusion_candidates(percentiles: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    candidates: dict[str, torch.Tensor] = {}
    names = sorted(percentiles)
    for left, right in itertools.combinations(names, 2):
        for left_weight in (0.25, 0.5, 0.75):
            name = f"fusion__{left}_{left_weight:.2f}__{right}_{1.0 - left_weight:.2f}"
            candidates[name] = left_weight * percentiles[left] + (1.0 - left_weight) * percentiles[right]
    for first, second, third in itertools.combinations(names, 3):
        name = f"fusion3__{first}__{second}__{third}"
        candidates[name] = (percentiles[first] + percentiles[second] + percentiles[third]) / 3.0
    return candidates


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    channel = torch.load(args.channel_artifact, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    weight_device = torch.device("cpu")
    layer_ids = args.layers or sorted(int(layer_id) for layer_id in channel["table"])
    all_rows: dict[str, list[dict[str, float]]] = {}
    layer_rows: list[dict[str, Any]] = []
    expert_count = 0

    for layer_id in layer_ids:
        layer = table_row(channel["table"], layer_id)
        layer_metrics: dict[str, list[dict[str, float]]] = {}
        for expert_id, gate, up, down in iter_expert_weights(
            model_path, weight_map, adapter, layer_id, weight_device
        ):
            if args.expert_limit is not None and expert_id >= args.expert_limit:
                break
            zero_mask = layer["effective_zero_masks"][expert_id].to(device)
            aimer = layer["stable_aimer_scores"][expert_id].to(device)
            down_energy = layer["down_channel_energy"][expert_id].float().to(device)
            pseudo_output = layer["structural_channel_scores"][expert_id].float().to(device)
            pseudo_activation = safe_ratio(pseudo_output, down_energy)
            base_scores = {
                "aimer": aimer,
                "down_energy": down_energy,
                "pseudo_activation": pseudo_activation,
                "pseudo_output": pseudo_output,
                **{
                    name: scores.to(device)
                    for name, scores in weight_metrics(gate, up, down).items()
                },
            }
            base_names = list(base_scores)
            base_matrix = torch.stack([base_scores[name] for name in base_names])
            base_percentiles = batched_rank_percentiles(base_matrix, zero_mask)
            percentiles = {name: base_percentiles[index] for index, name in enumerate(base_names)}
            candidate_scores = {**base_scores, **fusion_candidates(percentiles)}

            candidate_names = list(candidate_scores)
            candidate_matrix = torch.stack([candidate_scores[name] for name in candidate_names])
            fit_scores = layer["channel_scores"][expert_id].float().to(device)
            fit_selected = layer["ranked_indices"][expert_id, :args.width].to(device=device, dtype=torch.long)
            holdout_scores = layer["holdout_route_weighted_response_energy"][expert_id].float().to(device) * down_energy
            holdout_selected = batched_order(
                holdout_scores.unsqueeze(0), zero_mask, aimer
            )[0, :args.width]
            metric_tensors = batched_candidate_metrics(
                candidate_matrix, zero_mask, aimer, args.width,
                fit_selected, fit_scores, holdout_selected, holdout_scores
            )
            for candidate_index, name in enumerate(candidate_names):
                metrics = {
                    metric_name: float(values[candidate_index].item())
                    for metric_name, values in metric_tensors.items()
                }
                all_rows.setdefault(name, []).append(metrics)
                layer_metrics.setdefault(name, []).append(metrics)
            expert_count += 1
        layer_rows.append({
            "layer_id": layer_id,
            "candidates": {name: mean_metrics(rows) for name, rows in layer_metrics.items()},
        })

    summary = {name: mean_metrics(rows) for name, rows in all_rows.items()}
    ranked_by_holdout = sorted(
        summary,
        key=lambda name: (
            summary[name]["holdout_utility_recall"],
            summary[name]["holdout_spearman"],
        ),
        reverse=True,
    )
    result = {
        "schema_version": 1,
        "teacher": "CHANNEL fit and independent holdout utility; screening only",
        "candidate_runtime": "data_free",
        "width": args.width,
        "layers": layer_ids,
        "experts": expert_count,
        "candidates_evaluated": len(summary),
        "top_candidates_by_holdout": [
            {"name": name, **summary[name]} for name in ranked_by_holdout[:25]
        ],
        "single_metrics": {
            name: summary[name] for name in sorted(summary) if not name.startswith("fusion")
        },
        "all_candidates": summary,
        "layer_results": layer_rows,
        "guardrail": (
            "CHANNEL is used only to screen fixed data-free formulas. Promotion requires a physical checkpoint "
            "and autoregressive evaluation; this report is not benchmark evidence."
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
