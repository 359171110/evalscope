from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from NAPS_v2.analyze_mask_channel_attribution import mean_metrics, rank_percentiles, selection_metrics, table_row
from NAPS_v2.analyze_weight_space_similarity import functional_kernel
from NAPS_v2.build_channel_artifacts import nested_order
from NAPS_v2.build_naps_v2_artifacts import iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import NapsV2Config, build_probe_sets, stable_concat_score, swiglu_response


PROTECTION_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cosine-diverse channel selection against CHANNEL.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mask-artifact-dir", type=Path, required=True)
    parser.add_argument("--channel-artifact", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--aimer-weight", type=float, default=0.5)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--expert-limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def output_similarity(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    responses: torch.Tensor | None,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    down_normalized = F.normalize(down.float(), dim=0, eps=epsilon)
    down_cosine = (down_normalized.transpose(0, 1) @ down_normalized).abs()
    functional = functional_kernel(gate, up, gate, up, epsilon).abs() * down_cosine
    result = {"functional_output": functional.clamp(0.0, 1.0)}
    if responses is not None:
        response_normalized = F.normalize(responses.float(), dim=0, eps=epsilon)
        response_cosine = (response_normalized.transpose(0, 1) @ response_normalized).abs()
        result["pseudo_output"] = (response_cosine * down_cosine).clamp(0.0, 1.0)
    for matrix in result.values():
        matrix.fill_diagonal_(1.0)
    return result


def diverse_order(
    similarity: torch.Tensor,
    importance_order: torch.Tensor,
    zero_mask: torch.Tensor,
    width: int,
    protection_ratio: float,
) -> torch.Tensor:
    channel_count = int(similarity.shape[0])
    protected = min(width, round(width * protection_ratio))
    seed_count = max(1, protected)
    selected = importance_order[:seed_count].to(device=similarity.device, dtype=torch.long).tolist()
    selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=similarity.device)
    selected_mask[torch.tensor(selected, device=similarity.device)] = True
    importance_rank = torch.empty(channel_count, dtype=torch.long, device=similarity.device)
    importance_rank[importance_order.to(similarity.device)] = torch.arange(channel_count, device=similarity.device)
    maximum_similarity = similarity[selected].amax(0)
    while len(selected) < width:
        novelty = 1.0 - maximum_similarity
        novelty[selected_mask | zero_mask.to(similarity.device)] = -torch.inf
        maximum = novelty.max()
        tied = torch.where(novelty == maximum)[0]
        chosen = tied[importance_rank[tied].argmin()]
        selected.append(int(chosen.item()))
        selected_mask[chosen] = True
        maximum_similarity = torch.maximum(maximum_similarity, similarity[chosen])
    selected_tensor = torch.tensor(selected, dtype=torch.long, device=importance_order.device)
    remaining = importance_order[~selected_mask.to(importance_order.device)[importance_order]]
    return torch.cat((selected_tensor, remaining))


def diversity_metrics(similarity: torch.Tensor, selected: torch.Tensor) -> dict[str, float]:
    selected_similarity = similarity.index_select(0, selected).index_select(1, selected).clone()
    selected_similarity.fill_diagonal_(-torch.inf)
    nearest = selected_similarity.amax(1)
    pairs = selected_similarity[torch.triu(torch.ones_like(selected_similarity, dtype=torch.bool), diagonal=1)]
    return {
        "mean_nearest_similarity": float(nearest.mean().item()),
        "p95_nearest_similarity": float(torch.quantile(nearest, 0.95).item()),
        "mean_pair_similarity": float(pairs.mean().item()),
    }


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    artifact_dir = args.mask_artifact_dir.expanduser().resolve()
    device = torch.device(args.device)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    channel = torch.load(args.channel_artifact, map_location="cpu", weights_only=False)
    audit = torch.load(artifact_dir / "routing_audit.pt", map_location="cpu", weights_only=True)
    layer_ids = args.layers or sorted(int(layer_id) for layer_id in channel["table"])
    config = NapsV2Config()
    epsilon = 1.0e-12
    candidate_rows: dict[str, list[dict[str, float]]] = {}
    layer_rows: list[dict[str, Any]] = []
    experts = 0

    for layer_id in layer_ids:
        channel_layer = table_row(channel["table"], layer_id)
        layer_audit = table_row(audit["layers"], layer_id)
        probes = layer_audit["expert_probes"].to(device)
        selected_experts = layer_audit["selected_experts"].to(device)
        selected_weights = layer_audit["selected_weights"].to(device)
        layer_candidates: dict[str, list[dict[str, float]]] = {}
        for expert_id, gate, up, down in iter_expert_weights(model_path, weight_map, adapter, layer_id, device):
            if args.expert_limit is not None and expert_id >= args.expert_limit:
                break
            zero_mask = channel_layer["effective_zero_masks"][expert_id].to(device)
            aimer = stable_concat_score(gate, up, down, config)
            gate_up_product = gate.float().square().sum(1) * up.float().square().sum(1)
            importance_scores = (
                args.aimer_weight * rank_percentiles(aimer.cpu(), zero_mask.cpu()).to(device)
                + (1.0 - args.aimer_weight)
                * rank_percentiles(gate_up_product.cpu(), zero_mask.cpu()).to(device)
            )
            importance_order = nested_order(importance_scores, zero_mask, aimer)
            probe_sets = build_probe_sets(probes, selected_experts, selected_weights, expert_id)
            responses = swiglu_response(
                probe_sets["coverage_probes"], gate, up, activation=adapter.channel_architecture.activation
            )
            similarities = output_similarity(gate, up, down, responses, epsilon)
            fit_scores = channel_layer["channel_scores"][expert_id].to(device).float()
            fit_selected = channel_layer["ranked_indices"][expert_id, :args.width].to(device)
            down_energy = channel_layer["down_channel_energy"][expert_id].to(device).float()
            holdout_scores = (
                channel_layer["holdout_route_weighted_response_energy"][expert_id].to(device).float()
                * down_energy
            )
            holdout_selected = nested_order(holdout_scores, zero_mask, aimer)[:args.width]
            selections = {"importance_baseline": importance_order[:args.width]}
            for similarity_name, similarity in similarities.items():
                for ratio in PROTECTION_RATIOS:
                    name = f"{similarity_name}_protect_{ratio:.2f}"
                    selections[name] = diverse_order(
                        similarity, importance_order, zero_mask, args.width, ratio
                    )[:args.width]
            for name, selected in selections.items():
                similarity_name = "functional_output" if name == "importance_baseline" else name.split("_protect_")[0]
                metrics = {
                    **prefixed_metrics("fit", selection_metrics(selected, fit_selected, fit_scores)),
                    **prefixed_metrics("holdout", selection_metrics(selected, holdout_selected, holdout_scores)),
                    **diversity_metrics(similarities[similarity_name], selected),
                }
                candidate_rows.setdefault(name, []).append(metrics)
                layer_candidates.setdefault(name, []).append(metrics)
            experts += 1
        layer_rows.append({
            "layer_id": layer_id,
            "candidates": {name: mean_metrics(rows) for name, rows in layer_candidates.items()},
        })

    summary = {name: mean_metrics(rows) for name, rows in candidate_rows.items()}
    result = {
        "schema_version": 1,
        "width": args.width,
        "experts": experts,
        "protection_ratios": list(PROTECTION_RATIOS),
        "candidate_definitions": {
            "functional_output": "absolute analytic gate/up functional cosine times absolute down-column cosine",
            "pseudo_output": "absolute pseudo-response cosine times absolute down-column cosine",
        },
        "candidates": summary,
        "layers": layer_rows,
        "guardrail": (
            "Diversity is screened against CHANNEL fit and independent holdout utility. Pure novelty may reduce "
            "redundancy while discarding important channels; promotion requires both diversity and utility retention."
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
