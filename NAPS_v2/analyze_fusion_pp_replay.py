from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.analyze_mask_channel_attribution import mean_metrics, rank_percentiles, selection_metrics, table_row
from NAPS_v2.build_channel_artifacts import nested_order
from NAPS_v2.build_naps_v2_artifacts import build_artifact_payload, iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import NapsV2Config, build_probe_sets, select_v2_mask, stable_concat_score, swiglu_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay bounded PP swaps on an AIMER/gate-up-product fusion baseline.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mask-artifact-dir", type=Path, required=True)
    parser.add_argument("--channel-artifact", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--aimer-weight", type=float, default=0.5)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--expert-limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-artifact-dir", type=Path)
    return parser.parse_args()


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def evaluate(
    selected: torch.Tensor,
    fit_selected: torch.Tensor,
    fit_scores: torch.Tensor,
    holdout_selected: torch.Tensor,
    holdout_scores: torch.Tensor,
) -> dict[str, float]:
    return {
        **prefixed_metrics("fit", selection_metrics(selected, fit_selected, fit_scores)),
        **prefixed_metrics("holdout", selection_metrics(selected, holdout_selected, holdout_scores)),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.aimer_weight <= 1.0:
        raise ValueError("aimer-weight must be in [0, 1]")
    model_path = args.model_path.expanduser().resolve()
    artifact_dir = args.mask_artifact_dir.expanduser().resolve()
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    channel = torch.load(args.channel_artifact, map_location="cpu", weights_only=False)
    mask = torch.load(artifact_dir / "rankings.pt", map_location="cpu", weights_only=False)
    audit = torch.load(artifact_dir / "routing_audit.pt", map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    config = NapsV2Config()
    activation = adapter.channel_architecture.activation
    layer_ids = args.layers or sorted(int(layer_id) for layer_id in channel["table"])
    candidate_rows: dict[str, list[dict[str, float]]] = {
        "aimer_baseline": [],
        "aimer_pp_replay": [],
        "fusion_baseline": [],
        "fusion_pp_replay": [],
    }
    layer_rows: list[dict[str, Any]] = []
    exact_replay_matches = 0
    experts = 0
    swap_count = 0
    positive_fit_swaps = 0
    negative_fit_swaps = 0
    positive_holdout_swaps = 0
    negative_holdout_swaps = 0
    fusion_pp_layer_orders: list[torch.Tensor] = []

    for layer_id in layer_ids:
        channel_layer = table_row(channel["table"], layer_id)
        mask_layer = table_row(mask["table"], layer_id)
        layer_audit = table_row(audit["layers"], layer_id)
        probes = layer_audit["expert_probes"].to(device)
        selected_experts = layer_audit["selected_experts"].to(device)
        selected_weights = layer_audit["selected_weights"].to(device)
        layer_candidates = {name: [] for name in candidate_rows}
        layer_fusion_pp_orders: list[torch.Tensor] = []
        for expert_id, gate, up, down in iter_expert_weights(model_path, weight_map, adapter, layer_id, device):
            if args.expert_limit is not None and expert_id >= args.expert_limit:
                break
            zero_mask = channel_layer["effective_zero_masks"][expert_id].to(device)
            aimer_scores = stable_concat_score(gate, up, down, config)
            aimer_order = nested_order(aimer_scores, zero_mask)
            gate_up_product = gate.float().square().sum(1) * up.float().square().sum(1)
            fusion_scores = (
                args.aimer_weight * rank_percentiles(aimer_scores.cpu(), zero_mask.cpu()).to(device)
                + (1.0 - args.aimer_weight)
                * rank_percentiles(gate_up_product.cpu(), zero_mask.cpu()).to(device)
            )
            fusion_order = nested_order(fusion_scores, zero_mask, aimer_scores)
            probe_sets = build_probe_sets(probes, selected_experts, selected_weights, expert_id)
            responses = swiglu_response(probe_sets["coverage_probes"], gate, up, activation=activation)
            aimer_pp_order, _ = select_v2_mask(
                aimer_order,
                aimer_scores,
                responses,
                zero_mask,
                args.width,
                int(probe_sets["native_rows"].numel()),
                config,
            )
            fusion_pp_order, fusion_mask_info = select_v2_mask(
                fusion_order,
                fusion_scores,
                responses,
                zero_mask,
                args.width,
                int(probe_sets["native_rows"].numel()),
                config,
            )
            layer_fusion_pp_orders.append(fusion_pp_order.detach().cpu())
            stored_order = mask_layer["ranked_indices"][expert_id].to(device=device, dtype=torch.long)
            exact_replay_matches += bool(torch.equal(aimer_pp_order, stored_order))

            fit_scores = channel_layer["channel_scores"][expert_id].to(device).float()
            fit_selected = channel_layer["ranked_indices"][expert_id, :args.width].to(device=device, dtype=torch.long)
            down_energy = channel_layer["down_channel_energy"][expert_id].to(device).float()
            holdout_scores = (
                channel_layer["holdout_route_weighted_response_energy"][expert_id].to(device).float()
                * down_energy
            )
            holdout_selected = nested_order(holdout_scores, zero_mask, aimer_scores)[:args.width]
            selections = {
                "aimer_baseline": aimer_order[:args.width],
                "aimer_pp_replay": aimer_pp_order[:args.width],
                "fusion_baseline": fusion_order[:args.width],
                "fusion_pp_replay": fusion_pp_order[:args.width],
            }
            for name, selection in selections.items():
                metrics = evaluate(selection, fit_selected, fit_scores, holdout_selected, holdout_scores)
                candidate_rows[name].append(metrics)
                layer_candidates[name].append(metrics)

            utility_fit = fit_scores.double()
            utility_holdout = holdout_scores.double()
            for incoming, outgoing in zip(
                fusion_mask_info["swap_in_channels"].tolist(),
                fusion_mask_info["swap_out_channels"].tolist(),
            ):
                swap_count += 1
                fit_delta = float(utility_fit[incoming].item() - utility_fit[outgoing].item())
                holdout_delta = float(utility_holdout[incoming].item() - utility_holdout[outgoing].item())
                positive_fit_swaps += fit_delta > 0.0
                negative_fit_swaps += fit_delta < 0.0
                positive_holdout_swaps += holdout_delta > 0.0
                negative_holdout_swaps += holdout_delta < 0.0
            experts += 1
        fusion_pp_layer_orders.append(torch.stack(layer_fusion_pp_orders))
        layer_rows.append({
            "layer_id": layer_id,
            "candidates": {name: mean_metrics(rows) for name, rows in layer_candidates.items()},
        })

    summary = {name: mean_metrics(rows) for name, rows in candidate_rows.items()}
    result = {
        "schema_version": 1,
        "width": args.width,
        "aimer_weight": args.aimer_weight,
        "experts": experts,
        "exact_aimer_pp_replay_matches": exact_replay_matches,
        "exact_aimer_pp_replay_fraction": exact_replay_matches / max(1, experts),
        "candidates": summary,
        "fusion_pp_delta": {
            key: summary["fusion_pp_replay"][key] - summary["fusion_baseline"][key]
            for key in summary["fusion_baseline"]
        },
        "fusion_pp_swap_effect": {
            "total_swaps": swap_count,
            "positive_fit_swaps": positive_fit_swaps,
            "negative_fit_swaps": negative_fit_swaps,
            "positive_holdout_swaps": positive_holdout_swaps,
            "negative_holdout_swaps": negative_holdout_swaps,
            "positive_fit_fraction": positive_fit_swaps / max(1, swap_count),
            "positive_holdout_fraction": positive_holdout_swaps / max(1, swap_count),
        },
        "layers": layer_rows,
        "guardrail": (
            "This replays the existing bounded PP rule on a new baseline. It is an attribution result, "
            "not evidence from a materialized checkpoint or autoregressive benchmark."
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    output_artifact_dir = None
    if args.output_artifact_dir is not None:
        output_artifact_dir = args.output_artifact_dir.expanduser().resolve()
        if output_artifact_dir.exists() and any(output_artifact_dir.iterdir()):
            raise FileExistsError(f"Output artifact directory is not empty: {output_artifact_dir}")
        output_artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if output_artifact_dir is not None:
        rankings, profile = build_artifact_payload(
            torch.stack(fusion_pp_layer_orders),
            model_path,
            args.width,
            32,
            {
                "backbone": "stable_aimer_gate_up_product_rank_fusion",
                "aimer_weight": args.aimer_weight,
                "gate_up_product_weight": 1.0 - args.aimer_weight,
                "refinement": "dynamic_pp_swap",
                "data_free": True,
                "channel_teacher_used_for_ranking": False,
                "test_metrics_used": False,
            },
        )
        torch.save(rankings, output_artifact_dir / "rankings.pt")
        torch.save(profile, output_artifact_dir / "profile.pt")
        shutil.copy2(artifact_dir / "compensation_plan.pt", output_artifact_dir / "compensation_plan.pt")
        (output_artifact_dir / "construction_report.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
