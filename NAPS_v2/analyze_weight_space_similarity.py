from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from NAPS_v2.build_naps_v2_artifacts import iter_expert_weights, load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter


THRESHOLDS = (0.9, 0.8, 0.7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze weight-only SwiGLU channel similarity for a NAPS-v2 mask.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    return parser.parse_args()


def retained_table(cache: dict[str, Any], retained_channels: int) -> dict[tuple[int, int], torch.Tensor]:
    return {
        (int(layer_id), expert_id): row[:retained_channels].to(torch.long)
        for layer_id, values in cache["table"].items()
        for expert_id, row in enumerate(values["ranked_indices"])
    }


def summarize(values: torch.Tensor) -> dict[str, float | int]:
    values = values.float().cpu()
    if not values.numel():
        return {"count": 0, "p25": 0.0, "median": 0.0, "p75": 0.0, **{f"gt_{t:.1f}": 0.0 for t in THRESHOLDS}}
    quantiles = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75]))
    summary: dict[str, float | int] = {
        "count": int(values.numel()),
        "p25": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "p75": float(quantiles[2].item()),
    }
    summary.update({f"gt_{threshold:.1f}": float((values > threshold).float().mean().item()) for threshold in THRESHOLDS})
    return summary


def functional_kernel(
    gate_left: torch.Tensor,
    up_left: torch.Tensor,
    gate_right: torch.Tensor,
    up_right: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    left_gate_up = (gate_left * up_left).sum(1)
    right_gate_up = (gate_right * up_right).sum(1)
    kernel = (
        left_gate_up[:, None] * right_gate_up[None, :]
        + (gate_left @ gate_right.transpose(0, 1)) * (up_left @ up_right.transpose(0, 1))
        + (gate_left @ up_right.transpose(0, 1)) * (up_left @ gate_right.transpose(0, 1))
    )
    left_self = gate_left.square().sum(1) * up_left.square().sum(1) + 2.0 * left_gate_up.square()
    right_self = gate_right.square().sum(1) * up_right.square().sum(1) + 2.0 * right_gate_up.square()
    denominator = torch.sqrt(left_self[:, None].clamp_min(epsilon) * right_self[None, :].clamp_min(epsilon))
    return kernel / denominator


def maximum_similarities(
    gate: torch.Tensor,
    up: torch.Tensor,
    retained: torch.Tensor,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    channel_count = gate.shape[0]
    retained = retained.to(device=gate.device, dtype=torch.long)
    retained_mask = torch.zeros(channel_count, dtype=torch.bool, device=gate.device)
    retained_mask[retained] = True
    pruned = torch.where(~retained_mask)[0]
    gate_pruned = gate.float().index_select(0, pruned)
    up_pruned = up.float().index_select(0, pruned)
    gate_retained = gate.float().index_select(0, retained)
    up_retained = up.float().index_select(0, retained)

    gate_cosine = F.normalize(gate_pruned, dim=1, eps=epsilon) @ F.normalize(
        gate_retained, dim=1, eps=epsilon
    ).transpose(0, 1)
    up_cosine = F.normalize(up_pruned, dim=1, eps=epsilon) @ F.normalize(
        up_retained, dim=1, eps=epsilon
    ).transpose(0, 1)
    gu_similarity = gate_cosine * up_cosine.abs()
    functional_similarity = functional_kernel(
        gate_pruned, up_pruned, gate_retained, up_retained, epsilon
    )
    gu_values, gu_positions = gu_similarity.max(dim=1)
    functional_values, functional_positions = functional_similarity.max(dim=1)
    return {
        "pruned": pruned,
        "gu_values": gu_values,
        "gu_representatives": retained.index_select(0, gu_positions),
        "functional_values": functional_values,
        "functional_representatives": retained.index_select(0, functional_positions),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    cache = torch.load(artifact_dir / "rankings.pt", map_location="cpu", weights_only=True)
    retained_channels = int(cache["naps"]["retained_channels"])
    retained_by_expert = retained_table(cache, retained_channels)
    expert_rows: list[dict[str, Any]] = []
    layer_values: dict[int, dict[str, list[torch.Tensor]]] = {}
    overall_values = {"gu": [], "functional": []}
    channel_path = output_dir / "pruned_channel_max_similarity.jsonl"

    with channel_path.open("w", encoding="utf-8") as channel_file:
        for layer_id in range(adapter.num_layers):
            layer_values[layer_id] = {"gu": [], "functional": []}
            for expert_id, gate, up, _ in iter_expert_weights(
                model_path, weight_map, adapter, layer_id, device
            ):
                retained = retained_by_expert[(layer_id, expert_id)]
                result = maximum_similarities(gate, up, retained, args.epsilon)
                gu_values = result["gu_values"].detach().cpu()
                functional_values = result["functional_values"].detach().cpu()
                layer_values[layer_id]["gu"].append(gu_values)
                layer_values[layer_id]["functional"].append(functional_values)
                overall_values["gu"].append(gu_values)
                overall_values["functional"].append(functional_values)
                for metric, values in (("gu", gu_values), ("functional", functional_values)):
                    expert_rows.append({
                        "model_family": adapter.model_family,
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "metric": metric,
                        **summarize(values),
                    })
                for position, channel in enumerate(result["pruned"].tolist()):
                    channel_file.write(json.dumps({
                        "model_family": adapter.model_family,
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "pruned_channel": channel,
                        "gu_max_similarity": float(gu_values[position].item()),
                        "gu_representative": int(result["gu_representatives"][position].item()),
                        "functional_max_similarity": float(functional_values[position].item()),
                        "functional_representative": int(result["functional_representatives"][position].item()),
                    }, ensure_ascii=False) + "\n")
            print(f"Analyzed layer {layer_id + 1}/{adapter.num_layers}", flush=True)

    layer_rows = []
    for layer_id, metrics in layer_values.items():
        for metric, chunks in metrics.items():
            layer_rows.append({
                "model_family": adapter.model_family,
                "layer_id": layer_id,
                "metric": metric,
                **summarize(torch.cat(chunks)),
            })
    summary = {
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
        "model_family": adapter.model_family,
        "source_width": adapter.intermediate_size,
        "retained_channels": retained_channels,
        "pruned_channels_per_expert": adapter.intermediate_size - retained_channels,
        "num_layers": adapter.num_layers,
        "num_experts": adapter.num_experts,
        "metrics": {
            metric: summarize(torch.cat(chunks))
            for metric, chunks in overall_values.items()
        },
    }
    write_csv(output_dir / "expert_summary.csv", expert_rows)
    write_csv(output_dir / "layer_summary.csv", layer_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())